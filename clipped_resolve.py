# clipped_resolve.py: clipped placement, phase 5b: resolve the width column of the phase 5 grid

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
Z = 1.96

REAL_SHAPE = REAL_MEAN_BP / REAL_MEDIAN_BP
TOL_SHAPE = 0.50

CELLS = [(0.0070, 0.0110), (0.0070, 0.0117), (0.0070, 0.0125),
         (0.0065, 0.0117), (0.0075, 0.0117)]

TARGETS = [("median width bp", "median_bp", REAL_MEDIAN_BP, TOL_WIDTH),
           ("levels/side", "levels_per_side", REAL_LEVELS, TOL_LEVELS),
           ("aggTrades/s", "agg_per_s", TARGET, TOL_RATE),
           ("mean/median shape", "shape", REAL_SHAPE, TOL_SHAPE)]


def measure(disp, p_market, seeds=SEEDS):
    """Per-seed statistics, so the across-seed spread is a real standard"""
    per = {k: [] for k in ("median_bp", "mean_bp", "shape", "levels_per_side",
                           "orders_per_side", "agg_per_s", "ev_per_s",
                           "near_btc", "near_lvl", "touch_age", "clipped_pct",
                           "agg_over_ev")}
    pooled = []
    t0 = time.time()
    for s in seeds:
        x = run(s, LAM, p_market, LIFE, disp)
        sp = sorted(x["spreads_bp"])
        med = sp[len(sp) // 2]
        mean = statistics.mean(sp)
        pooled += sp
        per["median_bp"].append(med)
        per["mean_bp"].append(mean)
        per["shape"].append(mean / med if med else float("nan"))
        per["agg_over_ev"].append(x["agg_per_s"] / x["ev_per_s"]
                                  if x["ev_per_s"] else float("nan"))
        for k in ("levels_per_side", "orders_per_side", "agg_per_s",
                  "ev_per_s", "near_btc", "near_lvl", "touch_age",
                  "clipped_pct"):
            per[k].append(x[k])
    pooled.sort()
    n = len(seeds)
    out = {"disp": disp, "pm": p_market, "n": n,
           "pooled_median_bp": pooled[len(pooled) // 2],
           "coh": SIGMA * math.sqrt(LIFE / SPY) / disp,
           "secs": time.time() - t0}
    for k, v in per.items():
        out[k] = statistics.mean(v)
        out[k + "_se"] = (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0
    return out


def verdict(point, se, target, tol):
    """MET / MISSED / UNRESOLVED against a +/-tol band around target"""
    lo_b, hi_b = target * (1 - tol), target * (1 + tol)
    lo, hi = point - Z * se, point + Z * se
    if lo >= lo_b and hi <= hi_b:
        return "MET"
    if hi < lo_b or lo > hi_b:
        return "MISSED"
    return "UNRESOLVED"


def main():
    print("=" * 132)
    print("Phase 5b; resolve the frontier cell at %d seeds, with standard "
          "errors" % len(SEEDS))
    print("=" * 132)
    print("JOIN clipping, MM absent. lam=%.4f, mean_lifetime=%.0fs held. "
          "%d seeds x %.0fs per cell." % (LAM, LIFE, len(SEEDS), T))
    print("Point estimates are means over per-seed statistics; +/- is 1 SE. "
          "Intervals below are +/- %.2f SE (95%%)." % Z)
    print("")

    hdr = ("  %-7s %-7s %17s %17s %8s %9s %17s %8s %7s %7s"
           % ("disp", "p_mkt", "median bp", "aggTr/s", "lvl/sd", "mn/md",
              "mean bp", "nearBTC", "nearLv", "tchAge"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = {}
    for disp, pm in CELLS:
        r = measure(disp, pm)
        rows[(disp, pm)] = r
        print("  %-7.4f %-7.4f %9.4f+/-%-6.4f %9.5f+/-%-6.5f %8.1f %9.2f "
              "%9.4f+/-%-6.4f %8.4f %7.2f %7.0f"
              % (disp, pm, r["median_bp"], r["median_bp_se"],
                 r["agg_per_s"], r["agg_per_s_se"], r["levels_per_side"],
                 r["shape"], r["mean_bp"], r["mean_bp_se"], r["near_btc"],
                 r["near_lvl"], r["touch_age"]), flush=True)

    print("")
    print("=" * 132)
    print("Verdict per target  (95% interval vs tolerance band; UNRESOLVED "
          "means the interval straddles an edge; NOT a hit)")
    print("=" * 132)
    n_met = {}
    for disp, pm in CELLS:
        r = rows[(disp, pm)]
        print("  disp=%.4f p_market=%.4f" % (disp, pm))
        met = 0
        for name, key, tgt, tol in TARGETS:
            pt, se = r[key], r[key + "_se"]
            v = verdict(pt, se, tgt, tol)
            met += (v == "MET")
            print("    %-20s %10.5f +/- %-9.5f  95%% [%9.5f, %9.5f]  band "
                  "[%9.5f, %9.5f]  %s"
                  % (name, pt, se, pt - Z * se, pt + Z * se,
                     tgt * (1 - tol), tgt * (1 + tol), v))
        n_met[(disp, pm)] = met
        print("    -> %d/4 MET" % met)
        print("")

    best = max(CELLS, key=lambda c: n_met[c])
    print("  Best cell: disp=%.4f p_market=%.4f with %d/4 targets MET."
          % (best[0], best[1], n_met[best]))
    if n_met[best] == 4:
        print("  A four-way hit is confirmed at 10 seeds.")
    else:
        unres = [n for n, k, t, tl in TARGETS
                 if verdict(rows[best][k], rows[best][k + "_se"], t, tl)
                 == "UNRESOLVED"]
        miss = [n for n, k, t, tl in TARGETS
                if verdict(rows[best][k], rows[best][k + "_se"], t, tl)
                == "MISSED"]
        print("  No four-way hit. Unresolved: %s. Missed outright: %s."
              % (", ".join(unres) or "none", ", ".join(miss) or "none"))

    print("")
    print("=" * 132)
    print("Monotonicity re-check at %d seeds" % len(SEEDS))
    print("=" * 132)
    print("Physics: width must rise with p_market at fixed disp (market orders")
    print("consume the touch), and rise with disp at fixed p_market (wider")
    print("draws sit further out). A violation larger than the combined SE")
    print("would mean something other than sampling noise.")
    print("")

    def check(axis, cells, label):
        print("  %s" % label)
        ok = True
        prev = None
        for c in cells:
            r = rows[c]
            tag = "%.4f" % (c[1] if axis == "pm" else c[0])
            delta = ""
            if prev is not None:
                d = r["median_bp"] - prev["median_bp"]
                cse = math.sqrt(r["median_bp_se"] ** 2
                                + prev["median_bp_se"] ** 2)
                sig = abs(d) > Z * cse
                if d < 0:
                    ok = False
                    delta = ("   DOWN %.4f  (%s)"
                             % (d, "SIGNIFICANT -- not noise" if sig
                                else "within noise"))
                else:
                    delta = ("   up  +%.4f  (%s)"
                             % (d, "significant" if sig else "within noise"))
            print("    %s=%s  median %.4f +/- %.4f%s"
                  % (axis, tag, r["median_bp"], r["median_bp_se"], delta))
            prev = r
        print("    -> %s" % ("monotone as expected" if ok
                             else "non-monotone; see note above"))
        print("")
        return ok

    a = check("pm", [(0.0070, 0.0110), (0.0070, 0.0117), (0.0070, 0.0125)],
              "width vs p_market at disp=0.0070 (expect rising)")
    b = check("disp", [(0.0065, 0.0117), (0.0070, 0.0117), (0.0075, 0.0117)],
              "width vs disp at p_market=0.0117 (expect rising)")
    if a and b:
        print("  Both axes monotone at %d seeds. The Phase 5 inversions were "
              "sampling noise." % len(SEEDS))
    else:
        print("  At least one axis is still non-monotone at %d seeds. If the "
              "violation is flagged" % len(SEEDS))
        print("  significant above, sampling noise does not explain it and the "
              "mechanism needs a look.")

    print("")
    print("  pooled medians (Phase 3-5 convention, for continuity): " +
          "  ".join("(%.4f,%.4f)=%.4f" % (d, p, rows[(d, p)]["pooled_median_bp"])
                    for d, p in CELLS))
    print("  total wall clock %.0fs" % sum(rows[c]["secs"] for c in CELLS))


if __name__ == "__main__":
    main()
