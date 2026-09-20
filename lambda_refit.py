# lambda_refit.py: refit of the recovered days on the lambda axis

import json
import math
import re
import sys
from datetime import date

CACHE_PATH = "regime_scan_rows.json"
RECOVER_PATH = "regime_recover_results.txt"
RESULTS_PATH = "lambda_refit_results.txt"

EXPECT_SIZE_INTERCEPT = 3.846236e-02
EXPECT_SIZE_SLOPE = 2.205875e-09
EXPECT_SIZE_RSD_PP = 3.8255
EXPECT_SIZE_R = 0.9413

out_lines = []


def emit(s=""):
    print(s)
    out_lines.append(s)


def ols(xs, ys):
    """Plain least squares plus the residual SD on n-2 df, the same estimator"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
    rsd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy)
    return intercept, slope, rsd, r, resid


def binom_tail_ge(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)"""
    tot = 0.0
    for i in range(k, n + 1):
        tot += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return tot


with open(CACHE_PATH, "r", encoding="utf-8") as f:
    cached = json.load(f)

clean = []
cache_by_day = {}
for c in cached:
    y, m, d = (int(x) for x in c["date"].split("-"))
    dt = date(y, m, d)
    cache_by_day[dt] = c
    if c.get("usable") and c["vol_ann"] is not None:
        clean.append({"d": dt, "vol": c["vol_ann"], "lam": c["lambda"],
                      "size": float(c["size"])})
clean.sort(key=lambda r: r["d"])

rec_line = re.compile(
    r"^\s+(\d{4}-\d{2}-\d{2}) \(\w{3}\) RECOVERED vol=\s*([\d.]+)%\s+"
    r"spread=\s*([\d.]+)bp\s+lambda=([\d.]+)\s+size=\s*(\d+)")
recovered = []
with open(RECOVER_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = rec_line.match(line)
        if m:
            y, mo, d = (int(x) for x in m.group(1).split("-"))
            recovered.append({"d": date(y, mo, d), "vol": float(m.group(2)) / 100.0,
                              "spread": float(m.group(3)), "lam": float(m.group(4)),
                              "size": float(m.group(5))})
recovered.sort(key=lambda r: r["d"])

emit("=" * 78)
emit("lambda_refit.py; the decisive test on the recovery")
emit("=" * 78)
emit("Source: %s (40 clean days) + %s (6 recovered days). No re-parse; this"
     % (CACHE_PATH, RECOVER_PATH))
emit("        script reads no .jsonl file.")
emit("")
emit("  clean days loaded    : %d" % len(clean))
emit("  recovered days loaded: %d" % len(recovered))
if len(clean) != 40 or len(recovered) != 6:
    emit("")
    emit("  expected 40 clean and 6 recovered. Cannot proceed. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)

emit("")
emit("=" * 78)
emit("Is lambda actually immune to the repair?; proven, not assumed")
emit("=" * 78)
emit("  The cache's lambda was written by regime_scan.py (control arm, before any")
emit("  repair existed). The results file's lambda was written by the REPAIRED")
emit("  arm. If the repair cannot touch the trade stream these must be identical.")
emit("")
emit("  The results file prints lambda to 5 decimal places, so the repaired value")
emit("  is compared at that precision; a tighter tolerance would only be testing")
emit("  my own parsing, not the data. Full cache precision is shown alongside.")
emit("")
emit("  %-12s %-18s %-14s %-10s"
     % ("day", "control (full)", "repaired (5dp)", "identical"))
lam_ok = True
for r in recovered:
    c = cache_by_day[r["d"]]
    same = abs(round(c["lambda"], 5) - r["lam"]) < 1e-12
    emit("  %-12s %18.12f %14.5f %-10s"
         % (r["d"].isoformat(), c["lambda"], r["lam"], "yes" if same else "*** no ***"))
    if not same:
        lam_ok = False
if not lam_ok:
    emit("")
    emit("  lambda differs between arms. It is NOT an independent axis and this")
    emit("  whole test is void. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)
emit("")
emit("  all six identical at the printed precision. Lambda is untouched by the")
emit("  repair, so it is a genuinely independent axis on which to test the")
emit("  recovered volatilities.")
emit("")
emit("  Having proven they agree, the full-precision cache lambda is used for the")
emit("  regression below, so no rounding from the results file enters the fit.")
for r in recovered:
    r["lam"] = cache_by_day[r["d"]]["lambda"]
    r["size"] = float(cache_by_day[r["d"]]["size"])
emit("  Note: the recovered volatilities are read from the results file at 2 dp")
emit("  (e.g. 25.87%), so each carries up to 0.005 pp of print rounding. Against")
emit("  a residual SD near 4 pp that is ~0.1% of one SD and changes nothing.")

emit("")
emit("=" * 78)
emit("Self-check: reproduce regime_recover.py's committed SIZE fit")
emit("=" * 78)
sx = [c["size"] for c in clean]
sy = [c["vol"] for c in clean]
s_int, s_slope, s_rsd, s_r, s_resid = ols(sx, sy)
checks = [
    ("intercept", EXPECT_SIZE_INTERCEPT, s_int, 1e-8),
    ("slope", EXPECT_SIZE_SLOPE, s_slope, 1e-15),
    ("residual SD (pp)", EXPECT_SIZE_RSD_PP, s_rsd * 100, 1e-3),
    ("r", EXPECT_SIZE_R, s_r, 1e-4),
]
ok = True
for name, want, got, tol in checks:
    good = abs(want - got) <= tol
    emit("  %-20s committed=%-16.8g got=%-16.8g %s"
         % (name, want, got, "OK" if good else "*** mismatch ***"))
    if not good:
        ok = False
if not ok:
    emit("")
    emit("  Self-check FAILED. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)
emit("  The size fit is reproduced exactly from the cache.")

lx = [c["lam"] for c in clean]
ly = [c["vol"] for c in clean]
l_int, l_slope, l_rsd, l_r, l_resid = ols(lx, ly)

emit("")
emit("=" * 78)
emit("(1) The lambda fit; 40 previously-clean days only, six held out")
emit("=" * 78)
emit("  vol = %.6f + %.6f * lambda" % (l_int, l_slope))
emit("  r = %+.4f    r^2 = %.4f    residual SD = %.4f pp    n = %d"
     % (l_r, l_r * l_r, l_rsd * 100, len(clean)))
emit("  in-sample lambda range: %.5f .. %.5f" % (min(lx), max(lx)))
emit("")
emit("  for comparison, the SIZE fit on the same 40 days:")
emit("  r = %+.4f    r^2 = %.4f    residual SD = %.4f pp"
     % (s_r, s_r * s_r, s_rsd * 100))
emit("  => lambda is the %s predictor of the two (residual SD %.4f vs %.4f pp)"
     % ("weaker" if l_rsd > s_rsd else "stronger", l_rsd * 100, s_rsd * 100))

lam_lo, lam_hi = min(lx), max(lx)
size_lo, size_hi = min(sx), max(sx)

emit("")
emit("=" * 78)
emit("(2) The six on lambda   (4) with extrapolation flagged on both axes")
emit("=" * 78)
emit("  in-sample lambda: %.5f .. %.5f      in-sample size: %.0f .. %.0f"
     % (lam_lo, lam_hi, size_lo, size_hi))
emit("")
emit("  %-12s %-9s %-10s %-10s %-11s %-8s %s"
     % ("day", "lambda", "predicted", "measured", "residual", "in SDs", "extrapolates?"))
rows = []
for r in recovered:
    pred = l_int + l_slope * r["lam"]
    res = r["vol"] - pred
    sds = res / l_rsd
    ext_l = r["lam"] > lam_hi or r["lam"] < lam_lo
    ext_s = r["size"] > size_hi or r["size"] < size_lo
    tags = []
    if ext_l:
        tags.append("LAMBDA")
    if ext_s:
        tags.append("size")
    rows.append({"d": r["d"], "lam": r["lam"], "pred": pred, "meas": r["vol"],
                 "res": res, "sds": sds, "ext_l": ext_l, "ext_s": ext_s})
    emit("  %-12s %9.5f %9.2f%% %9.2f%% %+10.2fpp %+7.2f  %s"
         % (r["d"].isoformat(), r["lam"], pred * 100, r["vol"] * 100,
            res * 100, sds, (", ".join(tags)) if tags else "no"))

emit("")
emit("  Same six under the committed SIZE fit, for direct comparison:")
emit("  %-12s %-10s %-10s %-11s %-8s" % ("day", "predicted", "measured", "residual", "in SDs"))
size_rows = []
for r in recovered:
    pred = s_int + s_slope * r["size"]
    res = r["vol"] - pred
    size_rows.append({"d": r["d"], "res": res, "sds": res / s_rsd})
    emit("  %-12s %9.2f%% %9.2f%% %+10.2fpp %+7.2f"
         % (r["d"].isoformat(), pred * 100, r["vol"] * 100, res * 100, res / s_rsd))

big_l = [x for x in rows if abs(x["sds"]) > 3.0]
big_s = [x for x in size_rows if abs(x["sds"]) > 3.0]
emit("")
emit("  days beyond 3 residual SDs on LAMBDA: %s"
     % (", ".join(x["d"].isoformat() for x in big_l) if big_l else "none"))
emit("  days beyond 3 residual SDs on SIZE  : %s"
     % (", ".join(x["d"].isoformat() for x in big_s) if big_s else "none"))
emit("  max |SDs| on lambda: %.2f      max |SDs| on size: %.2f"
     % (max(abs(x["sds"]) for x in rows), max(abs(x["sds"]) for x in size_rows)))

emit("")
emit("=" * 78)
emit("(5) CONTROL: the 40 clean days' own residual sign balance")
emit("=" * 78)
emit("  An OLS fit forces residuals to sum to zero, but not to split evenly. If")
emit("  the residual distribution is skewed, most points sit on one side of the")
emit("  line, and that; not 0.5; is the null the six must be judged against.")
emit("")
for name, resid in (("LAMBDA fit", l_resid), ("SIZE fit", s_resid)):
    pos = sum(1 for r in resid if r > 0)
    n = len(resid)
    med = sorted(resid)[n // 2]
    emit("  %-11s clean days above the line: %2d/%d = %.1f%%   median residual %+.3f pp"
         % (name, pos, n, 100.0 * pos / n, med * 100))

l_pos_rate = sum(1 for r in l_resid if r > 0) / len(l_resid)
s_pos_rate = sum(1 for r in s_resid if r > 0) / len(s_resid)

emit("")
emit("=" * 78)
emit("(3) Sign test on the six; both fits, side by side")
emit("=" * 78)
n_pos_l = sum(1 for x in rows if x["res"] > 0)
n_pos_s = sum(1 for x in size_rows if x["res"] > 0)
emit("  %-12s %-14s %-16s %-16s" % ("fit", "six positive", "p vs 0.5", "p vs clean rate"))
for name, npos, rate in (("LAMBDA", n_pos_l, l_pos_rate), ("SIZE", n_pos_s, s_pos_rate)):
    emit("  %-12s %d of 6         %-16.4f %-16.4f"
         % (name, npos, binom_tail_ge(npos, 6, 0.5), binom_tail_ge(npos, 6, rate)))
emit("")
emit("  caveat on every p ABOVE: the six are consecutive days and daily")
emit("  volatility has lag-1 autocorrelation +0.4370 (regime_scan_results.txt),")
emit("  so they are nowhere near six independent draws. A crude AR(1) adjustment")
emit("  n_eff = n(1-rho)/(1+rho) gives n_eff = %.2f, i.e. roughly two independent"
     % (6 * (1 - 0.4370) / (1 + 0.4370)))
emit("  observations. Treat these p-values as an upper bound on the evidence,")
emit("  not as a test that clears any threshold.")

emit("")
emit("  Residual sign, day by day, both fits:")
emit("  %-12s %-12s %-12s" % ("day", "on LAMBDA", "on SIZE"))
for a, b in zip(rows, size_rows):
    emit("  %-12s %+11.2fpp %+11.2fpp" % (a["d"].isoformat(), a["res"] * 100, b["res"] * 100))

emit("")
emit("=" * 78)
emit("Nearest clean days by lambda; the non-parametric read")
emit("=" * 78)
emit("  For each recovered day, the three clean days closest in lambda, with")
emit("  their volatilities. This makes no linearity assumption at all.")
for r in recovered:
    near = sorted(clean, key=lambda c: abs(c["lam"] - r["lam"]))[:3]
    emit("")
    emit("  %s  lambda=%.5f  vol=%.2f%%"
         % (r["d"].isoformat(), r["lam"], r["vol"] * 100))
    for c in near:
        emit("      nearest clean %s  lambda=%.5f  vol=%.2f%%  (diff %+.2f pp)"
             % (c["d"].isoformat(), c["lam"], c["vol"] * 100,
                (r["vol"] - c["vol"]) * 100))

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print("")
print("Results written to " + RESULTS_PATH)
