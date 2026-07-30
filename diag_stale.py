# diag_stale.py: stale resting liquidity and the touch

import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T / dt), seed=31415)
value_fn = path_value(path, dt)

eng = MatchingEngine()
for i in range(5):
    eng.add_limit_order("buy", round(62000 - (i + 1) * 0.5, 2), 0.02, "seed")
    eng.add_limit_order("sell", round(62000 + (i + 1) * 0.5, 2), 0.02, "seed")

noise = NoiseTraderFlow(lam=LAM_NOISE * 20, p_market=0.05, value_fn=value_fn,
                        disp=0.0002, seed=1001, reference_price=62000.0)
informed = InformedTraderFlow(seed=2002)

run_max_v, run_min_v = path[0], path[0]
err_to_v_bid, err_to_runmax = [], []
err_to_v_ask, err_to_runmin = [], []
rows = []
t = 0.0
while t < T:
    t += 1.0
    v = value_fn(t)
    run_max_v = max(run_max_v, v)
    run_min_v = min(run_min_v, v)
    noise.run_until(eng, t)
    informed.run_until(eng, t, value_fn)
    bb, ba = eng.best_bid(), eng.best_ask()
    if bb is not None:
        err_to_v_bid.append(abs(bb - v))
        err_to_runmax.append(abs(bb - run_max_v))
    if ba is not None:
        err_to_v_ask.append(abs(ba - v))
        err_to_runmin.append(abs(ba - run_min_v))
    if int(t) % 17280 == 0:
        rows.append((t, bb, ba, v, run_max_v, run_min_v))

print("Dense config: lam=20x, p_market=0.05, disp=0.0002, no cancellation")
print("resting orders at end: %d" % len(eng.orders))
print("")
print("%-8s %11s %11s %11s %12s %12s"
      % ("t(s)", "best_bid", "best_ask", "V", "runmax V", "runmin V"))
for (tt, bb, ba, v, mx, mn) in rows:
    print("%-8.0f %11s %11s %11.2f %12.2f %12.2f"
          % (tt, "%.2f" % bb if bb else "None",
             "%.2f" % ba if ba else "None", v, mx, mn))
print("")
print("which does the touch track; current V, or the historical envelope?")
print("  mean |best_bid - V|            = $%.2f" % statistics.mean(err_to_v_bid))
print("  mean |best_bid - running max V|= $%.2f  <-- smaller => tracks the "
      "envelope" % statistics.mean(err_to_runmax))
print("")
print("  mean |best_ask - V|            = $%.2f" % statistics.mean(err_to_v_ask))
print("  mean |best_ask - running min V|= $%.2f  <-- smaller => tracks the "
      "envelope" % statistics.mean(err_to_runmin))
print("")
print("V path range over the run: %.2f .. %.2f (span $%.2f)"
      % (min(path), max(path), max(path) - min(path)))
