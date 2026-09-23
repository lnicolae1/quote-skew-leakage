# cv_pilot.py: control-variate pilot: one seed, maker size 0.12, control arm

import math
import os
import statistics
import sys

sys.path.insert(0, r"C:\Users\lnico\Desktop")
os.chdir(r"C:\Users\lnico\Desktop")

import extraction_diag as ED
from clipped_placement import make_world

SEED = 0
SIZE = 0.12
Z = 1.96

TMB_M = {60: -0.1121, 300: -0.5573, 900: -1.5313, 3600: -5.2016, 10800: -0.6760}
TMB_SE = {60: 0.0416, 300: 0.2037, 900: 0.6218, 3600: 2.3827, 10800: 5.3301}
COST, COST_SE = 1.4704, 0.0100
COST_LO, COST_HI = COST - Z * COST_SE, COST + Z * COST_SE

print("=" * 110)
print("Control-variate pilot; seed %d, maker size %.2f, control arm only" %
      (SEED, SIZE))
print("=" * 110)

out, cap = ED.run_diag(SEED, False, ED.THETA, ED.HOLD, SIZE)
samples = cap["samples"]
print("run_diag: %d samples at 60s (control arm), sd(q) = %.5f" %
      (len(samples), out["sd_q"]))

p1, vf, *_ = make_world(SEED, ED.T, ED.LAM, ED.P_MARKET, ED.LIFE, ED.DISP,
                        "join")
p2, _vf2, *_ = make_world(SEED, ED.T, ED.LAM, ED.P_MARKET, ED.LIFE, ED.DISP,
                          "join")
assert p1 == p2, "V path is not deterministic in the seed"
print("V path regenerated twice, identical (%d steps)." % len(p1))

mis = [1e4 * (m - vf(t)) / m for t, (_q, m, _c) in samples.items()]
print("mispricing (mid - V) in bp of mid: mean %+.3f, sd %.3f, max|.| %.3f" %
      (statistics.fmean(mis), statistics.stdev(mis), max(abs(x) for x in mis)))
print("")


def incs(h, overlapping):
    dm, dv = [], []
    step = ED.SAMPLE if overlapping else h
    t = ED.SAMPLE
    while t + h <= ED.T:
        a, b = samples.get(t), samples.get(t + h)
        if a is not None and b is not None:
            m0, m1 = a[1], b[1]
            dm.append(1e4 * (m1 - m0) / m0)
            dv.append(1e4 * (vf(t + h) - vf(t)) / m0)
        t += step
    return dm, dv


print("%-7s %-7s %-12s %-14s %-9s %-8s %-11s | %-9s %s" % (
    "h (s)", "n", "var(d_mid)", "var(dmid-dV)", "ratio r", "corr",
    "1-corr^2", "r (overlap)", "n (overlap)"))
ratios = {}
for h in ED.HORIZONS:
    dm, dv = incs(h, False)
    res = [a - b for a, b in zip(dm, dv)]
    vm, vr = statistics.variance(dm), statistics.variance(res)
    r = vr / vm
    rho = statistics.correlation(dm, dv)
    dmo, dvo = incs(h, True)
    reso = [a - b for a, b in zip(dmo, dvo)]
    ro = statistics.variance(reso) / statistics.variance(dmo)
    ratios[h] = r
    print("%-7d %-7d %-12.4f %-14.4f %-9.4f %-8.4f %-11.4f | %-11.4f %d" % (
        h, len(dm), vm, vr, r, rho, 1 - rho * rho, ro, len(dmo)))

print("")
print("=" * 110)
print("Projection; 95%% half-width of TMB at 8 seeds under the control "
      "variate, maker size 0.12")
print("measured cost %.4f +/- %.4f bp -> band [%.4f, %.4f]" %
      (COST, COST_SE, COST_LO, COST_HI))
print("=" * 110)
print("%-7s %-11s %-11s %-11s %-11s %-26s %s" % (
    "h (s)", "hw now", "sqrt(r)", "hw cv", "|m| now",
    "hw_CV < cost_lo?", "|m|+-hw_CV vs cost"))
for h in ED.HORIZONS:
    hw_now = Z * TMB_SE[h]
    s = math.sqrt(ratios[h])
    hw = hw_now * s
    m = abs(TMB_M[h])
    can_below = hw < COST_LO
    lo, hi = max(0.0, m - hw), m + hw
    if hi < COST_LO:
        where = "BELOW if |m| holds"
    elif lo > COST_HI:
        where = "ABOVE if |m| holds"
    else:
        where = "STRADDLE if |m| holds"
    print("%-7d %-11.4f %-11.4f %-11.4f %-11.4f %-26s [%.3f, %.3f] %s" % (
        h, hw_now, s, hw, m, "yes" if can_below else "no", lo, hi, where))
