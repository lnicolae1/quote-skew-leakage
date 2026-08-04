# clipped_window.py: clipped placement, phase 5c: target bands, verdict and margin helpers

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import TARGET
from clipped_depth_sweep import (run, SIGMA, SPY, LAM, REAL_MEDIAN_BP,
                                 REAL_MEAN_BP, REAL_LEVELS, REAL_NEAR_BTC,
                                 REAL_NEAR_LEVELS, REAL_TOUCH_AGE, TOL_WIDTH,
                                 TOL_LEVELS, TOL_RATE)

T = 86400.0
SEEDS = list(range(10))
LIFE = 720.0
P_MARKET = 0.0120
Z = 1.96

REAL_SHAPE = REAL_MEAN_BP / REAL_MEDIAN_BP
TOL_SHAPE = 0.50

DISPS = (0.0050, 0.0055, 0.0060, 0.0065)

TARGETS = [("median width bp", "median_bp", REAL_MEDIAN_BP, TOL_WIDTH),
           ("levels/side", "levels_per_side", REAL_LEVELS, TOL_LEVELS),
           ("aggTrades/s", "agg_per_s", TARGET, TOL_RATE),
           ("mean/median shape", "shape", REAL_SHAPE, TOL_SHAPE)]


def measure(disp, p_market, seeds=SEEDS):
    per = {k: [] for k in ("median_bp", "mean_bp", "shape", "levels_per_side",
                           "orders_per_side", "agg_per_s", "ev_per_s",
                           "near_btc", "near_lvl", "touch_age", "clipped_pct")}
    t0 = time.time()
    for s in seeds:
        x = run(s, LAM, p_market, LIFE, disp)
        sp = sorted(x["spreads_bp"])
        med = sp[len(sp) // 2]
        mean = statistics.mean(sp)
        per["median_bp"].append(med)
        per["mean_bp"].append(mean)
        per["shape"].append(mean / med if med else float("nan"))
        for k in ("levels_per_side", "orders_per_side", "agg_per_s",
                  "ev_per_s", "near_btc", "near_lvl", "touch_age",
                  "clipped_pct"):
            per[k].append(x[k])
    n = len(seeds)
    out = {"disp": disp, "pm": p_market, "n": n,
           "coh": SIGMA * math.sqrt(LIFE / SPY) / disp,
           "secs": time.time() - t0}
    for k, v in per.items():
        out[k] = statistics.mean(v)
        out[k + "_se"] = (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0
    return out


def band(target, tol):
    return target * (1 - tol), target * (1 + tol)


def verdict(point, se, target, tol):
    lo_b, hi_b = band(target, tol)
    lo, hi = point - Z * se, point + Z * se
    if lo >= lo_b and hi <= hi_b:
        return "MET"
    if hi < lo_b or lo > hi_b:
        return "MISSED"
    return "UNRESOLVED"


def margin_se(point, se, target, tol):
    """Distance from the point estimate to the nearest band edge, in SE"""
    lo_b, hi_b = band(target, tol)
    d = min(point - lo_b, hi_b - point)
    return d / se if se > 0 else float("inf")


def linfit(xs, ys):
    """Least-squares slope/intercept"""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx if sxx else float("nan")
    return m, my - m * mx


def solve_for(m, b, y):
    """disp at which the fitted line reaches y"""
    return (y - b) / m if m else float("nan")


def main():
    print("=" * 128)
    print("Phase 5c; is there a disp window at L=720 where width and levels "
          "are both in band?")
    print("=" * 128)
    print("JOIN clipping, MM absent. lam=%.4f, L=%.0fs, p_market=%.4f held. "
          "%d seeds x %.0fs per cell." % (LAM, LIFE, P_MARKET, len(SEEDS), T))
    print("p_market is pinned because Phase 5b showed it moves the rate at "
          "3.7 sigma while moving")
    print("width by -0.0037 +/- 0.024 bp, i.e. not at all. disp is swept alone.")
    print("")
    hdr = ("  %-7s %17s %19s %15s %8s %8s %8s %7s %7s"
           % ("disp", "median bp", "aggTr/s", "lvl/sd", "mn/md", "nearBTC",
              "nearLv", "tchAge", "clip%"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for disp in DISPS:
        r = measure(disp, P_MARKET)
        rows.append(r)
        print("  %-7.4f %9.4f+/-%-6.4f %10.5f+/-%-7.5f %8.1f+/-%-5.1f %8.2f "
              "%8.4f %8.2f %7.0f %6.1f%%"
              % (disp, r["median_bp"], r["median_bp_se"], r["agg_per_s"],
                 r["agg_per_s_se"], r["levels_per_side"],
                 r["levels_per_side_se"], r["shape"], r["near_btc"],
                 r["near_lvl"], r["touch_age"], r["clipped_pct"]), flush=True)

    print("")
    print("=" * 128)
    print("Verdicts  (95% interval vs tolerance band)")
    print("=" * 128)
    met_count = {}
    for r in rows:
        print("  disp=%.4f" % r["disp"])
        met = 0
        for name, key, tgt, tol in TARGETS:
            pt, se = r[key], r[key + "_se"]
            v = verdict(pt, se, tgt, tol)
            met += (v == "MET")
            lo_b, hi_b = band(tgt, tol)
            print("    %-20s %10.5f +/- %-9.5f  95%% [%9.5f, %9.5f]  band "
                  "[%9.5f, %9.5f]  %s"
                  % (name, pt, se, pt - Z * se, pt + Z * se, lo_b, hi_b, v))
        met_count[r["disp"]] = met
        print("    -> %d/4 MET" % met)
        print("")

    winners = [r for r in rows if met_count[r["disp"]] == 4]

    if winners:
        print("=" * 128)
        print("Margin report for 4/4 cells; distance from point estimate to "
              "the nearest band edge, in SE")
        print("=" * 128)
        for r in winners:
            print("  disp=%.4f" % r["disp"])
            worst = None
            for name, key, tgt, tol in TARGETS:
                m = margin_se(r[key], r[key + "_se"], tgt, tol)
                lo_b, hi_b = band(tgt, tol)
                near = "lower" if (r[key] - lo_b) < (hi_b - r[key]) else "upper"
                print("    %-20s %8.2f SE from the %s edge" % (name, m, near))
                if worst is None or m < worst[1]:
                    worst = (name, m)
            print("    tightest: %s at %.2f SE." % (worst[0], worst[1]))
            if worst[1] < 1.0:
                print("    -> This is NOT a calibration. The binding target is "
                      "inside one SE of the edge;")
                print("       a different seed set could push it out.")
            elif worst[1] < 2.0:
                print("    -> Marginal. Usable as a working point, but the "
                      "binding target has under 2 SE")
                print("       of room and should be re-checked at more seeds.")
            else:
                print("    -> Comfortable: every target is at least 2 SE "
                      "inside its band.")
            print("")

    print("=" * 128)
    print("Window geometry; (a) empty window, or (b) unresolved window?")
    print("=" * 128)
    xs = [r["disp"] for r in rows]
    wm, wb = linfit(xs, [r["median_bp"] for r in rows])
    lm, lb = linfit(xs, [r["levels_per_side"] for r in rows])
    w_lo_b, w_hi_b = band(REAL_MEDIAN_BP, TOL_WIDTH)
    l_lo_b, l_hi_b = band(REAL_LEVELS, TOL_LEVELS)

    print("  fits over the %d measured cells (both near-linear in disp here):"
          % len(rows))
    print("    width  = %+.3f * disp %+.5f      (%.1f bp per unit disp)"
          % (wm, wb, wm))
    print("    levels = %+.1f * disp %+.1f" % (lm, lb))
    print("")
    d_w_lo, d_w_hi = solve_for(wm, wb, w_lo_b), solve_for(wm, wb, w_hi_b)
    d_l_lo = solve_for(lm, lb, l_lo_b)
    print("  width in band  for disp in [%.5f, %.5f]" % (d_w_lo, d_w_hi))
    print("  levels in band for disp >= %.5f  (upper edge %.1f is far above "
          "anything reachable here)" % (d_l_lo, l_hi_b))
    lo, hi = max(d_w_lo, d_l_lo), d_w_hi
    print("")
    if hi > lo:
        print("  intersection: disp in [%.5f, %.5f]; width %.5f wide. "
              "The window exists." % (lo, hi, hi - lo))
        print("  => Failure mode (b) if no cell hit 4/4: the window is real "
              "but narrow, and the")
        print("     confidence intervals at %d seeds straddle an edge inside "
              "it." % len(SEEDS))
    else:
        print("  No intersection: width needs disp <= %.5f, levels need disp "
              ">= %.5f." % (d_w_hi, d_l_lo))
        print("  => Failure mode (a): the window is empty at L=%.0f. More "
              "seeds cannot fix this." % LIFE)
        print("     Width and levels cannot both be in band at any disp with "
              "L and lam at these values.")

    print("")
    if winners:
        print("  Result: %d cell(s) hit 4/4; see the margin report above for "
              "whether that is a calibration." % len(winners))
    else:
        print("  Result: no cell hit 4/4. The diagnosis above says which "
              "failure mode applies.")
    print("  total wall clock %.0fs" % sum(r["secs"] for r in rows))


if __name__ == "__main__":
    main()
