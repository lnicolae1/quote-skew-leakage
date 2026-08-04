# clipped_confirm.py: phase 5d confirmation run at the calibrated working point
# - lam 1.804, p_market 0.0126, mean_lifetime 720 s, disp 0.0055, JOIN clipping, MM absent, informed present
# - p_market raised from 0.0120 (phase 5c: 3/4, rate UNRESOLVED at 1.73 SE) to 0.0126 (projected rate ~0.0445)
# - 15 seeds x 86400 s; per-target margin to the nearest band edge, in SE

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import TARGET
from clipped_depth_sweep import (LAM, REAL_MEDIAN_BP, REAL_MEAN_BP,
                                 REAL_LEVELS, REAL_NEAR_BTC, REAL_NEAR_LEVELS,
                                 REAL_TOUCH_AGE)
from clipped_window import (measure, verdict, margin_se, band, TARGETS, Z,
                            LIFE, REAL_SHAPE)

DISP = 0.0055
P_MARKET = 0.0126
SEEDS = list(range(15))


def main():
    print("=" * 122)
    print("Phase 5d; confirmation run at the working point")
    print("=" * 122)
    print("lam=%.4f  p_market=%.4f  mean_lifetime=%.0fs  disp=%.4f  "
          "JOIN clipping, MM absent." % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x 86400s. Point estimates are means over per-seed "
          "statistics; +/- is 1 SE." % len(SEEDS))
    print("")

    r = measure(DISP, P_MARKET, SEEDS)

    print("Verdicts  (95% interval vs tolerance band)")
    print("-" * 122)
    met = 0
    margins = []
    for name, key, tgt, tol in TARGETS:
        pt, se = r[key], r[key + "_se"]
        v = verdict(pt, se, tgt, tol)
        met += (v == "MET")
        lo_b, hi_b = band(tgt, tol)
        print("  %-20s %10.5f +/- %-9.5f  95%% [%9.5f, %9.5f]  band "
              "[%9.5f, %9.5f]  %s"
              % (name, pt, se, pt - Z * se, pt + Z * se, lo_b, hi_b, v))
        margins.append((name, margin_se(pt, se, tgt, tol), lo_b, hi_b, pt))
    print("")
    print("  -> %d/4 MET" % met)

    print("")
    print("margin report; distance from point estimate to the nearest band "
          "edge, in SE")
    print("-" * 122)
    for name, m, lo_b, hi_b, pt in margins:
        near = "lower" if (pt - lo_b) < (hi_b - pt) else "upper"
        print("  %-20s %8.2f SE from the %s edge" % (name, m, near))
    tight = min(margins, key=lambda z: z[1])
    print("")
    print("  tightest: %s at %.2f SE." % (tight[0], tight[1]))
    if met < 4:
        print("  NOT a four-way hit; see the verdicts above.")
    elif tight[1] < 1.0:
        print("  4/4, but not a calibration: the binding target is inside "
              "one SE of its edge.")
    elif tight[1] < 2.0:
        print("  4/4 but marginal: the binding target has under 2 SE of room.")
    else:
        print("  4/4 with every target at least 2 SE inside its band.")

    print("")
    print("Unflagged diagnostics (not targets; not controlled by the knobs above)")
    print("-" * 122)
    for label, sim, real, key in (
            ("near-touch BTC (<=1bp)", r["near_btc"], REAL_NEAR_BTC, "near_btc"),
            ("near-touch levels (<=1bp)", r["near_lvl"], REAL_NEAR_LEVELS, "near_lvl"),
            ("touch-order age (s)", r["touch_age"], REAL_TOUCH_AGE, "touch_age")):
        print("  %-28s %10.4f +/- %-8.4f  vs %9.4f   %.2fx"
              % (label, sim, r[key + "_se"], real, sim / real))
    print("")
    print("  Standing problem: size per near-touch level")
    print("  is too high; no knob in this investigation addresses it.")
    print("")
    print("  orders/side %.1f +/- %.1f   levels/side %.1f +/- %.1f   "
          "mean bp %.4f   clipped %.1f%%   coherence %.3f"
          % (r["orders_per_side"], r["orders_per_side_se"],
             r["levels_per_side"], r["levels_per_side_se"], r["mean_bp"],
             r["clipped_pct"], r["coh"]))
    print("  wall clock %.0fs" % r["secs"])


if __name__ == "__main__":
    main()
