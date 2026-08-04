# clipped_pmarket_sweep.py: clipped placement, phase 2: p_market sweep

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import measure, run, LAM, TARGET, T

TICK_BP = 1e4 * 0.01 / 62000.0


def line(r, label, extra=""):
    return ("  %-22s %9.4f %9.4f %8.1f %8.1f %8.4f %9.4f %8.2fx%s"
            % (label, r["median_bp"], r["mean_bp"], r["orders_per_side"],
               r["levels_per_side"], r["ev_per_s"], r["agg_per_s"],
               r["agg_per_s"] / TARGET, extra))


HDR = ("  %-22s %9s %9s %8s %8s %8s %9s %9s"
       % ("", "median bp", "mean bp", "ord/sd", "lvl/sd", "ev/s", "aggTr/s",
          "vs .0451"))

print("=" * 104)
print("(A) Is the 'TOUCH' mode result a one-tick pin?")
print("=" * 104)
print("  One tick ($0.01) at the $62,000 reference = %.4f bp." % TICK_BP)
print("  Real-market median touch = 0.29bp = ~%.0f ticks. A book pinned at one"
      % (0.29 / TICK_BP))
print("  tick is ~180x tighter than reality, which is a degenerate pin, not a")
print("  calibration success. noise_traders.py's C2 block warns of exactly")
print("  this: a touch-anchored rule is self-referential.")
print("")
x = run(0, 0.0020, 180.0, "touch")
sp = sorted(x["spreads_bp"])
n = len(sp)
print("  TOUCH, disp=0.0020 life=180, 1 seed, spread distribution in bp:")
for p in (5, 25, 50, 75, 95, 99):
    print("    p%-3d %9.4f bp   (= %.1f ticks)"
          % (p, sp[int(p / 100.0 * n)], sp[int(p / 100.0 * n)] / TICK_BP))
at_tick = sum(1 for s in sp if s <= TICK_BP * 1.001)
print("    fraction of seconds pinned at exactly one tick: %.2f%%"
      % (100.0 * at_tick / n))

print("")
print("=" * 104)
print("(B) Restoring the trade rate under JOIN; p_market grid, disp=0.0020 "
      "life=180, 1 seed")
print("=" * 104)
print(HDR)
best = None
for pm in (0.02, 0.05, 0.08, 0.11, 0.15, 0.20):
    r = measure(0.0020, 180.0, "join", p_market=pm)
    mark = ""
    if best is None or abs(r["agg_per_s"] - TARGET) < abs(best[1]["agg_per_s"] - TARGET):
        best = (pm, r)
    print(line(r, "p_market=%.3f" % pm, mark))
PM_STAR = best[0]
print("")
print("  closest to target: p_market=%.3f at %.4f aggTrades/s (%.2fx)"
      % (PM_STAR, best[1]["agg_per_s"], best[1]["agg_per_s"] / TARGET))

print("")
print("=" * 104)
print("(C) DISP sweep at p_market=%.3f, JOIN, life=180, 3 seeds; how tight "
      "before the rate breaches?" % PM_STAR)
print("=" * 104)
print(HDR)
for disp in (0.0020, 0.0010, 0.0005, 0.00025, 0.00010):
    r = measure(disp, 180.0, "join", p_market=PM_STAR)
    flag = ""
    if r["agg_per_s"] > TARGET:
        flag = "  <== BREACHES target"
    if r["median_bp"] <= 0.29:
        flag += "  <== reaches 0.29bp"
    print(line(r, "disp=%.5f" % disp, flag))

print("")
print("  Reference: Phase 3 real-market median touch 0.29bp; near-touch depth")
print("  0.0452 BTC / 2.15 levels within 1bp on the bid; ~666 total bid")
print("  levels/side. Target trade rate %.4f aggTrades/s." % TARGET)
