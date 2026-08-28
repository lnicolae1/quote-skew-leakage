# quant_repro.py: does QUANT at HIGH reproduce the committed arm B run with the committed w_q instead of the measured-sd(q) one?
# - A: mm_override with the committed W_Q_LOW
# - B: no override (the incumbent path)
# - C: mm_override with phase8_gamma's measured width (control)
# - nothing scored; no committed default touched

import os
import sys

_DESKTOP = r"C:\Users\lnico\Desktop"
if _DESKTOP not in sys.path:
    sys.path.insert(0, _DESKTOP)

import phase7_blind as BLIND
import phase7_nobs as NOBS

from market_maker import MarketMaker
from phase7_blind import QuantizedMM, mean_se

GAMMA_HIGH = NOBS.GAMMA_CAP
SEEDS = list(NOBS.SEEDS)

# width phase8_gamma ran at HIGH: 0.34 * measured 8-seed sd(q) (0.046035)
SD_Q_MEASURED_HIGH = 0.046035
W_Q_MEASURED = BLIND.W_OVER_SD_LOW * SD_Q_MEASURED_HIGH

# committed arm B values
COMMITTED_M1 = (0.846980, 0.091807)
COMMITTED_CLIMB = (-0.006231, 0.007353)

assert issubclass(QuantizedMM, MarketMaker)
assert NOBS.GAMMA_CAP == 9.4e-6, "GAMMA_CAP moved"
assert BLIND.W_Q_LOW == BLIND.W_OVER_SD_LOW * NOBS.SD_Q_REF, \
    "W_Q_LOW is not 0.34 * SD_Q_REF any more"


def arm_override(w_q):
    return [NOBS.run(s, "QUANT", gamma=GAMMA_HIGH,
                     mm_override=QuantizedMM(w_q, horizon=NOBS.T, k=NOBS.K,
                                             gamma=GAMMA_HIGH,
                                             quote_size=NOBS.QUOTE_SIZE))
            for s in SEEDS]


def arm_incumbent():
    """No override: the incumbent path."""
    return [NOBS.run(s, "QUANT") for s in SEEDS]


def report(label, per_seed, w_q):
    m1, se1 = mean_se(NOBS.curve_values(per_seed, NOBS.M_CONTROL))
    cm, cse, cmult = NOBS.paired_between(per_seed, NOBS.M_CONTROL, NOBS.M_TOP)
    print("  %-34s w_q=%.9f" % (label, w_q))
    print("      M=1    %+.6f +/- %.6f   (committed %+.6f +/- %.6f)"
          % (m1, se1, COMMITTED_M1[0], COMMITTED_M1[1]))
    print("      climb  %+.6f +/- %.6f   (committed %+.6f +/- %.6f)  %.2f SE"
          % (cm, cse, COMMITTED_CLIMB[0], COMMITTED_CLIMB[1], cmult))
    exact_m1 = (abs(m1 - COMMITTED_M1[0]) < 5e-7
                and abs(se1 - COMMITTED_M1[1]) < 5e-7)
    exact_cl = (abs(cm - COMMITTED_CLIMB[0]) < 5e-7
                and abs(cse - COMMITTED_CLIMB[1]) < 5e-7)
    print("      reproduces 3625d32 to printed precision: M=1 %s, climb %s"
          % ("yes" if exact_m1 else "no", "yes" if exact_cl else "no"))
    print("")
    return {"m1": (m1, se1), "climb": (cm, cse), "curve": per_seed}


def main():
    bar = "=" * 100
    print(bar)
    print("QUANT at HIGH: does the committed w_q reproduce the committed run?")
    print(bar)
    print("gamma=%.2e, %d seeds x %.0fs, k=%.5f, quote_size=%.3f. "
          "Nothing scored, nothing re-run that was committed."
          % (GAMMA_HIGH, len(SEEDS), NOBS.T, NOBS.K, NOBS.QUOTE_SIZE))
    print("committed W_Q_LOW      = %.9f  (0.34 * SD_Q_REF = %.5f)"
          % (BLIND.W_Q_LOW, NOBS.SD_Q_REF))
    print("phase8 measured w_q    = %.9f  (0.34 * %.6f)"
          % (W_Q_MEASURED, SD_Q_MEASURED_HIGH))
    print("relative width change  = %.6f%%"
          % (100.0 * (W_Q_MEASURED - BLIND.W_Q_LOW) / BLIND.W_Q_LOW))
    print("")

    a = report("A  override + COMMITTED w_q", arm_override(BLIND.W_Q_LOW),
               BLIND.W_Q_LOW)
    b = report("B  no override (incumbent path)", arm_incumbent(),
               BLIND.W_Q_LOW)
    c = report("C  override + measured w_q", arm_override(W_Q_MEASURED),
               W_Q_MEASURED)

    print(bar)
    print("attribution")
    print(bar)
    same_ab = all(x["ma_curve"] == y["ma_curve"]
                  for x, y in zip(a["curve"], b["curve"]))
    same_ac = all(x["ma_curve"] == y["ma_curve"]
                  for x, y in zip(a["curve"], c["curve"]))
    print("  A vs B  (override path at the SAME width) identical curves: %s"
          % ("yes" if same_ab else "no"))
    print("  A vs C  (same path, different width)       identical curves: %s"
          % ("yes" if same_ac else "no"))
    print("")
    if same_ab and not same_ac:
        print("  -> Override path inert; width is the whole")
        print("     difference. The phase8 HIGH gap is the 0.011% width.")
    elif not same_ab:
        print("  -> Override path differs at the same width;")
        print("     width is not the explanation and phase8_gamma.py's HIGH")
        print("     arm has an unidentified defect.")
    else:
        print("  -> Width made no difference to the curves; the phase8 gap")
        print("     is neither the width nor the override path.")
    print("")
    print("  per-seed M=1:")
    print("    seed   A (committed w)      B (no override)     C (measured w)")
    for i, s in enumerate(SEEDS):
        print("    %-6d %+.9f  %+.9f  %+.9f"
              % (s, a["curve"][i]["r2_unsmoothed"],
                 b["curve"][i]["r2_unsmoothed"],
                 c["curve"][i]["r2_unsmoothed"]))
    print("")


if __name__ == "__main__":
    main()
