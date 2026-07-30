# diag_intercept.py: interception of noise limit orders by MM quotes

import statistics
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T/dt), seed=31415)
value_fn = path_value(path, dt)

eng = MatchingEngine()
for i in range(5):
    eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
    eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")
noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=1001,
                        value_fn=value_fn, reference_price=62000.0)
informed = InformedTraderFlow(seed=2002)
mm = MarketMaker(horizon=T)


def residual_touch(engine):
    """Best bid/ask ignoring the MM's own quotes"""
    bb = ba = None
    for o in engine.orders.values():
        if str(o.agent_id) == "MM":
            continue
        if o.side == "buy":
            bb = o.price if bb is None else max(bb, o.price)
        else:
            ba = o.price if ba is None else min(ba, o.price)
    return bb, ba


n_limit = 0
n_intercepted = 0
n_crossed_anyway = 0
n_rested = 0
intercepted_dist = []

mm_fills_from = {"noise_limit_intercepted": 0, "noise_limit_other": 0,
                 "noise_market": 0, "informed_take": 0, "informed_post": 0}

t = 0.0
while t < T:
    t += 1.0
    v = value_fn(t)
    bb_f, ba_f = eng.best_bid(), eng.best_ask()
    bb_r, ba_r = residual_touch(eng)

    for r in noise.run_until(eng, t):
        hit_mm = sum(1 for f in r.fills if f.counterparty_id == "MM")
        if r.fills:
            mm.on_fills(r.fills, r.side)
        if r.order_type != "limit":
            mm_fills_from["noise_market"] += hit_mm
            continue
        n_limit += 1
        if r.side == "buy":
            mkt_full = ba_f is not None and r.price >= ba_f
            mkt_resid = ba_r is not None and r.price >= ba_r
        else:
            mkt_full = bb_f is not None and r.price <= bb_f
            mkt_resid = bb_r is not None and r.price <= bb_r
        if mkt_full and not mkt_resid:
            n_intercepted += 1
            intercepted_dist.append(abs(r.price - v))
            mm_fills_from["noise_limit_intercepted"] += hit_mm
        elif mkt_full:
            n_crossed_anyway += 1
            mm_fills_from["noise_limit_other"] += hit_mm
        else:
            n_rested += 1
            mm_fills_from["noise_limit_other"] += hit_mm

    for r in informed.run_until(eng, t, value_fn):
        hit_mm = sum(1 for f in r.fills if f.counterparty_id == "MM")
        key = "informed_take" if r.action == "take" else "informed_post"
        mm_fills_from[key] += hit_mm
        if r.fills:
            mm.on_fills(r.fills, r.side)

    mm.requote(eng, t)

print("Q1; interception of noise limit orders by THE MM (with-MM run)")
print("")
print("  noise limit orders submitted            : %d" % n_limit)
print("  intercepted (would have rested, MM ate) : %d (%.1f%%)"
      % (n_intercepted, 100.0*n_intercepted/n_limit))
print("  marketable anyway (MM irrelevant)       : %d (%.1f%%)"
      % (n_crossed_anyway, 100.0*n_crossed_anyway/n_limit))
print("  passive either way (rested)             : %d (%.1f%%)"
      % (n_rested, 100.0*n_rested/n_limit))
print("")
if intercepted_dist:
    d = sorted(intercepted_dist)
    print("  how near V were the intercepted orders? |price - V|:")
    print("    median $%.2f   mean $%.2f   p25 $%.2f   p75 $%.2f   max $%.2f"
          % (d[len(d)//2], statistics.mean(d), d[len(d)//4], d[3*len(d)//4],
             d[-1]))
    for band in (5.0, 15.0, 30.0, 60.0):
        k = sum(1 for x in d if x <= band)
        print("    within $%-5.0f of V: %d (%.1f%% of intercepted)"
              % (band, k, 100.0*k/len(d)))
print("")
passive = sum(mm_fills_from.values())
total = mm.n_buy_fills + mm.n_sell_fills
print("  MM fill attribution (fill legs)")
print("    total MM fill legs                : %d" % total)
print("    passive (someone crossed into MM) : %d (%.1f%%)"
      % (passive, 100.0*passive/total))
print("    MM's own aggressive crossings     : %d (%.1f%%)"
      % (total-passive, 100.0*(total-passive)/total))
for k, vv in sorted(mm_fills_from.items(), key=lambda kv: -kv[1]):
    print("      %-28s %6d  (%.1f%% of all MM fills)"
          % (k, vv, 100.0*vv/total))
