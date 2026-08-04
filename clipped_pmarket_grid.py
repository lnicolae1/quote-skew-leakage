# clipped_pmarket_grid.py: clipped placement, phase 5: (p_market, mean_lifetime, disp) grid

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import TARGET
from clipped_depth_sweep import (run, SIGMA, SPY, LAM, LAM_CEILING,
                                 REAL_MEDIAN_BP, REAL_MEAN_BP, REAL_LEVELS,
                                 REAL_NEAR_BTC, REAL_NEAR_LEVELS,
                                 REAL_TOUCH_AGE, TOL_WIDTH, TOL_LEVELS,
                                 TOL_RATE)

T = 86400.0
SEEDS = [0, 1, 2]
FOLLOWUP_SEEDS = [0, 1, 2, 3, 4]

REAL_SHAPE = REAL_MEAN_BP / REAL_MEDIAN_BP
TOL_SHAPE = 0.50

PMARKETS = (0.0100, 0.0117, 0.0130)
GRID = [(720.0, (0.0070, 0.0080, 0.0090)),
        (540.0, (0.0050, 0.0060, 0.0070))]


def measure(life, disp, p_market, seeds=SEEDS):
    pooled = []
    keys = ["orders_per_side", "levels_per_side", "near_btc", "near_lvl",
            "touch_age", "all_age", "ev_per_s", "agg_per_s", "clipped_pct",
            "two_sided_pct"]
    acc = {k: [] for k in keys}
    t0 = time.time()
    for s in seeds:
        x = run(s, LAM, p_market, life, disp)
        pooled += x["spreads_bp"]
        for k in keys:
            acc[k].append(x[k])
    pooled.sort()
    out = {k: statistics.mean(v) for k, v in acc.items()}
    med = pooled[len(pooled) // 2]
    mean = statistics.mean(pooled)
    out.update(median_bp=med, mean_bp=mean,
               shape=mean / med if med else float("nan"),
               life=life, disp=disp, pm=p_market,
               coh=SIGMA * math.sqrt(life / SPY) / disp,
               n_seeds=len(seeds), secs=time.time() - t0)
    out["agg_over_ev"] = (out["agg_per_s"] / out["ev_per_s"]
                          if out["ev_per_s"] else float("nan"))
    return out


def hits(r):
    return (abs(r["median_bp"] - REAL_MEDIAN_BP) <= TOL_WIDTH * REAL_MEDIAN_BP,
            abs(r["levels_per_side"] - REAL_LEVELS) <= TOL_LEVELS * REAL_LEVELS,
            abs(r["agg_per_s"] - TARGET) <= TOL_RATE * TARGET,
            abs(r["shape"] - REAL_SHAPE) <= TOL_SHAPE * REAL_SHAPE)


FIELDS = [
    ("life",    "%-6s", lambda r: "%-6.0f" % r["life"]),
    ("disp",    "%-7s", lambda r: "%-7.4f" % r["disp"]),
    ("p_mkt",   "%-7s", lambda r: "%-7.4f" % r["pm"]),
    ("med bp",  "%8s",  lambda r: "%8.4f" % r["median_bp"]),
    ("mean bp", "%8s",  lambda r: "%8.4f" % r["mean_bp"]),
    ("mn/md",   "%6s",  lambda r: "%6.2f" % r["shape"]),
    ("ord/sd",  "%8s",  lambda r: "%8.1f" % r["orders_per_side"]),
    ("lvl/sd",  "%8s",  lambda r: "%8.1f" % r["levels_per_side"]),
    ("nearBTC", "%8s",  lambda r: "%8.4f" % r["near_btc"]),
    ("nearLv",  "%6s",  lambda r: "%6.2f" % r["near_lvl"]),
    ("ev/s",    "%7s",  lambda r: "%7.4f" % r["ev_per_s"]),
    ("aggTr/s", "%8s",  lambda r: "%8.4f" % r["agg_per_s"]),
    ("vs.0451", "%7s",  lambda r: "%6.2fx" % (r["agg_per_s"] / TARGET)),
    ("agg/ev",  "%6s",  lambda r: "%6.3f" % r["agg_over_ev"]),
    ("clip%",   "%6s",  lambda r: "%5.1f%%" % r["clipped_pct"]),
    ("tchAge",  "%6s",  lambda r: "%6.0f" % r["touch_age"]),
    ("coh",     "%6s",  lambda r: "%6.3f" % r["coh"]),
]


def header():
    return "  " + " ".join(f % n for n, f, _ in FIELDS) + "  flags"


def row(r):
    w, l, a, s = hits(r)
    flags = (("W" if w else ".") + ("L" if l else ".")
             + ("R" if a else ".") + ("S" if s else "."))
    return "  " + " ".join(g(r) for _, _, g in FIELDS) + "  " + flags


def main():
    print("=" * 150)
    print("Phase 5; p_market unpinned: fine grid around the two Phase 4 "
          "compromise cells")
    print("=" * 150)
    print("JOIN clipping, MM absent. lam=%.4f held (%.0f%% of the ~%.2f IDs/s "
          "real update-rate ceiling)."
          % (LAM, 100 * LAM / LAM_CEILING, LAM_CEILING))
    print("%d seeds x %.0fs per cell." % (len(SEEDS), T))
    print("Targets: median %.4fbp (+/-%.0f%%), levels/side %.1f (+/-%.0f%%), "
          "%.4f aggTr/s (+/-%.0f%%), mean/median %.3f (+/-%.0f%%)."
          % (REAL_MEDIAN_BP, 100 * TOL_WIDTH, REAL_LEVELS, 100 * TOL_LEVELS,
             TARGET, 100 * TOL_RATE, REAL_SHAPE, 100 * TOL_SHAPE))
    print("Flags: W=width  L=levels  R=rate  S=shape.  '....' = none.")
    print("")
    print(header())
    print("  " + "-" * (len(header()) - 2))

    rows = []
    for life, disps in GRID:
        for disp in disps:
            for pm in PMARKETS:
                r = measure(life, disp, pm)
                rows.append(r)
                print(row(r), flush=True)
            print("", flush=True)
        print("", flush=True)

    print("=" * 150)
    winners = [r for r in rows if all(hits(r))]
    if winners:
        print("Cells hitting all four:")
        for r in winners:
            print("  L=%.0f disp=%.4f p_market=%.4f -> median %.4fbp (%.2fx), "
                  "levels %.1f (%.2fx), %.4f aggTr/s (%.2fx), shape %.2f (%.2fx)"
                  % (r["life"], r["disp"], r["pm"], r["median_bp"],
                     r["median_bp"] / REAL_MEDIAN_BP, r["levels_per_side"],
                     r["levels_per_side"] / REAL_LEVELS, r["agg_per_s"],
                     r["agg_per_s"] / TARGET, r["shape"],
                     r["shape"] / REAL_SHAPE))
    else:
        print("no cell hit all four. Best by count of targets met, then by "
              "total relative error:")
        def score(r):
            h = hits(r)
            err = (abs(r["median_bp"] - REAL_MEDIAN_BP) / REAL_MEDIAN_BP
                   + abs(r["levels_per_side"] - REAL_LEVELS) / REAL_LEVELS
                   + abs(r["agg_per_s"] - TARGET) / TARGET
                   + abs(r["shape"] - REAL_SHAPE) / REAL_SHAPE)
            return (-sum(h), err)
        for r in sorted(rows, key=score)[:5]:
            w, l, a, s = hits(r)
            print("  L=%-5.0f disp=%.4f pm=%.4f  %d/4  med %.4f (%.2fx) lvl "
                  "%.1f (%.2fx) rate %.4f (%.2fx) shape %.2f (%.2fx)"
                  % (r["life"], r["disp"], r["pm"], sum(hits(r)),
                     r["median_bp"], r["median_bp"] / REAL_MEDIAN_BP,
                     r["levels_per_side"], r["levels_per_side"] / REAL_LEVELS,
                     r["agg_per_s"], r["agg_per_s"] / TARGET,
                     r["shape"], r["shape"] / REAL_SHAPE))

    print("")
    print("age / levels tradeoff; does L=540 beat L=720?")
    print("  (the L=540 bet: levels above 530 with touch age under 450s)")
    print("  %-6s %10s %10s %10s" % ("life", "tchAge", "lvl/sd", "verdict"))
    for life, _ in GRID:
        sel = [r for r in rows if r["life"] == life]
        ta = statistics.mean(r["touch_age"] for r in sel)
        best_lvl = max(r["levels_per_side"] for r in sel)
        ok = "bet holds" if (best_lvl >= 530 and ta < 450) else "bet fails"
        print("  %-6.0f %9.0fs %10.1f %10s  (age %.2fx the %.0fs reference)"
              % (life, ta, best_lvl, ok, ta / REAL_TOUCH_AGE, REAL_TOUCH_AGE))

    if not winners:
        print("")
        print("follow-up skipped; no cell hit all four.")
        return
    best = min(winners, key=lambda r: abs(r["median_bp"] - REAL_MEDIAN_BP))
    print("")
    print("=" * 150)
    print("Follow-up; L=%.0f disp=%.4f p_market=%.4f re-run at %d seeds"
          % (best["life"], best["disp"], best["pm"], len(FOLLOWUP_SEEDS)))
    print("=" * 150)
    r5 = measure(best["life"], best["disp"], best["pm"], FOLLOWUP_SEEDS)
    pairs = [("median touch bp", r5["median_bp"], REAL_MEDIAN_BP),
             ("mean touch bp", r5["mean_bp"], REAL_MEAN_BP),
             ("mean/median shape", r5["shape"], REAL_SHAPE),
             ("levels/side (total)", r5["levels_per_side"], REAL_LEVELS),
             ("near-touch BTC (<=1bp)", r5["near_btc"], REAL_NEAR_BTC),
             ("near-touch levels (<=1bp)", r5["near_lvl"], REAL_NEAR_LEVELS),
             ("aggTrades/s", r5["agg_per_s"], TARGET),
             ("touch-order age (s)", r5["touch_age"], REAL_TOUCH_AGE)]
    print("  %-28s %12s %12s %9s" % ("", "sim (5 seeds)", "real/target", "ratio"))
    for name, sim, real in pairs:
        print("  %-28s %12.4f %12.4f %8.2fx" % (name, sim, real, sim / real))
    print("")
    print("  orders/side %.1f   two-sided %.2f%%   clipped %.1f%%   agg/ev "
          "%.3f   coherence ratio %.3f"
          % (r5["orders_per_side"], r5["two_sided_pct"], r5["clipped_pct"],
             r5["agg_over_ev"], r5["coh"]))


if __name__ == "__main__":
    main()
