# diag_pin.py: why the touch does not move: two hypotheses

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

noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=1001,
                        reference_price=62000.0)
informed = InformedTraderFlow(seed=2002)

bid_hist, n_orders_hist = [], []
t = 0.0
while t < T:
    t += 1.0
    noise.run_until(eng, t)
    informed.run_until(eng, t, value_fn)
    bb = eng.best_bid()
    if bb is not None:
        bid_hist.append(bb)
    n_orders_hist.append(len(eng.orders))

near = sum(1 for b in bid_hist if abs(b - 62000.0) < 1.0)
print("H2; is the touch pinned at the start price?")
print("  samples with best_bid within $1 of 62000: %d/%d (%.2f%%)"
      % (near, len(bid_hist), 100.0 * near / len(bid_hist)))
print("  samples with best_bid within $10 of 62000: %d/%d (%.2f%%)"
      % (sum(1 for b in bid_hist if abs(b - 62000.0) < 10.0), len(bid_hist),
         100.0 * sum(1 for b in bid_hist if abs(b - 62000.0) < 10.0)
         / len(bid_hist)))
print("  best_bid excursions above 62100 : %d samples"
      % sum(1 for b in bid_hist if b > 62100.0))
print("  => excursions happen but do not persist: the touch returns to the "
      "stale level underneath.")
print("")

print("  resting order count over run: min=%d max=%d final=%d"
      % (min(n_orders_hist), max(n_orders_hist), n_orders_hist[-1]))
print("  (no agent in the simulator ever cancels: noise_traders and "
      "informed_traders never call engine.cancel_order)")
print("")

print("feasibility; can this many informed events move the price that far?")
v_move = abs(path[-1] - path[0])
ticks_needed = v_move / 0.01
print("  V moved $%.2f = %.0f ticks on a $0.01 grid" % (v_move, ticks_needed))
print("  informed arrivals available in the whole session: %d"
      % informed.n_arrivals)
print("  => price improvement required per informed event: $%.2f (%.0f ticks)"
      % (v_move / max(1, informed.n_arrivals),
         ticks_needed / max(1, informed.n_arrivals)))
print("  a one-tick-improvement rule would deliver at most $%.2f of total "
      "movement." % (informed.n_arrivals * 0.01))
print("")

print("SIZE; can one informed order clear a price level?")
print("  informed size          : %.5f BTC" % informed.size)
print("  mean noise size (P3)   : 0.00347 BTC (%.1fx informed)"
      % (0.00347 / informed.size))
print("  seeded level size      : 0.02000 BTC (%.1fx informed)"
      % (0.02 / informed.size))
