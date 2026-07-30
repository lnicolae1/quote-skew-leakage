# diag_anchor.py: share of steps where observe_mid() sees a real residual mid vs its own stale last_mid
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

real, fallback = 0, 0
resid_dev, mm_dev = [], []
t = 0.0
while t < T:
    t += 1.0
    for r in noise.run_until(eng, t):
        if r.fills:
            mm.on_fills(r.fills, r.side)
    for r in informed.run_until(eng, t, value_fn):
        if r.fills:
            mm.on_fills(r.fills, r.side)
    mm.cancel_quotes(eng)
    resid = eng.mid()
    if resid is None:
        fallback += 1
    else:
        real += 1
        resid_dev.append(abs(resid - value_fn(t)))
    mm.bid_order_id = mm.ask_order_id = None
    mm.requote(eng, t)
    m = eng.mid()
    if m is not None:
        mm_dev.append(abs(m - value_fn(t)))

n = real + fallback
print("MM anchor after cancelling own quotes")
print("  real residual mid       : %d/%d (%.1f%%)" % (real, n, 100.0*real/n))
print("  fallback to own last_mid: %d/%d (%.1f%%)"
      % (fallback, n, 100.0*fallback/n))
print("")
print("  mean |residual mid - V|    = $%.2f   [when available]"
      % statistics.mean(resid_dev))
print("  mean |posted book mid - V| = $%.2f" % statistics.mean(mm_dev))
