# diag_final.py: 3-day run
import statistics
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

MULT, PM, LIFE, DISP = 5, 0.02, 180.0, 0.0020
T, dt = 3*86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T/dt), seed=31415)
value_fn = path_value(path, dt)
eng = MatchingEngine()
for i in range(5):
    eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
    eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")
noise = NoiseTraderFlow(lam=LAM_NOISE*MULT, p_market=PM, seed=1001,
                        value_fn=value_fn, reference_price=62000.0,
                        mean_lifetime=LIFE, disp=DISP)
informed = InformedTraderFlow(seed=2002)
mm = MarketMaker(horizon=T)

birth = {oid: 0.0 for oid in eng.orders}
rows, ages = [], []
n_tr = n_lim = n_lim_mkt = n_buy = n_sell = 0
t = 0.0
while t < T:
    t += 1.0
    for r in noise.run_until(eng, t):
        if r.side == "buy": n_buy += 1
        else: n_sell += 1
        if r.order_type == "limit":
            n_lim += 1
            if r.fills: n_lim_mkt += 1
        if r.order_id is not None: birth.setdefault(r.order_id, t)
        if r.fills:
            n_tr += 1; mm.on_fills(r.fills, r.side)
    for r in informed.run_until(eng, t, value_fn):
        if getattr(r, "order_id", None) is not None:
            birth.setdefault(r.order_id, t)
        if r.fills:
            n_tr += 1; mm.on_fills(r.fills, r.side)
    mm.requote(eng, t)
    bo = ao = None
    for o in eng.orders.values():
        if str(o.agent_id) == "MM": continue
        if o.side == "buy":
            if bo is None or o.price > bo.price: bo = o
        else:
            if ao is None or o.price < ao.price: ao = o
    if bo: ages.append(t-birth.get(bo.id, t))
    if ao: ages.append(t-birth.get(ao.id, t))
    rows.append((eng.best_bid(), eng.best_ask(), eng.mid(), value_fn(t),
                 bo is not None and ao is not None))

print("3-day with-MM, C2b @ mult=%d p_market=%.2f life=%.0fs disp=%.4f"
      % (MULT, PM, LIFE, DISP))
print("")
for d in range(3):
    seg = rows[d*86400:(d+1)*86400]
    u = [m-v for (_b, _a, m, v, _r) in seg if m is not None]
    tw = sum(1 for r in seg if r[0] is not None and r[1] is not None)
    r2 = sum(1 for r in seg if r[4])
    print("  day %d: mean|mid-V| $%6.2f  signed $%+8.2f  two-sided %.2f%%  "
          "resid2sd %.1f%%" % (d+1, statistics.mean(map(abs, u)),
                               statistics.mean(u), 100.0*tw/len(seg),
                               100.0*r2/len(seg)))
allu = [m-v for (_b, _a, m, v, _r) in rows if m is not None]
tw = sum(1 for r in rows if r[0] is not None and r[1] is not None)
r2 = sum(1 for r in rows if r[4])
ages.sort()
print("")
print("  whole 3 days : mean|mid-V| $%.2f   signed $%+.2f"
      % (statistics.mean(map(abs, allu)), statistics.mean(allu)))
print("  trade rate   : %.4f /s      [Phase 3 target 0.0451/s]" % (n_tr/T))
print("  marketable noise limits: %.1f%% of %d" % (100.0*n_lim_mkt/n_lim, n_lim))
print("  two-sided: full %.2f%%   non-MM %.1f%%"
      % (100.0*tw/len(rows), 100.0*r2/len(rows)))
print("  touch age: median %.0fs  mean %.0fs  p90 %.0fs"
      % (ages[len(ages)//2], statistics.mean(ages), ages[int(0.9*len(ages))]))
print("  noise buy %.2f%% / sell %.2f%%  (n=%d)"
      % (100.0*n_buy/(n_buy+n_sell), 100.0*n_sell/(n_buy+n_sell), n_buy+n_sell))
print("  MM: fills=%d  q at 3 days %+.5f BTC   [pre-B2 -1.85951, post-B2 -2.61573]"
      % (mm.n_buy_fills+mm.n_sell_fills, mm.q))
print("  B2 cancels: scheduled=%d cancelled=%d already-filled=%d"
      % (noise.n_scheduled, noise.n_cancelled, noise.n_already_gone))
sp = sorted(a-b for (b, a, _m, _v, _r) in rows if b is not None and a is not None)
print("  median touch spread $%.2f (%.2f bp)"
      % (sp[len(sp)//2], 1e4*sp[len(sp)//2]/path[-1]))
