"""Plotted values, read from the committed results files by file, line number and pattern (fails if a file changed)."""
import os
import re
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "RESULTS_DIR", os.path.dirname(os.path.abspath(__file__)))

NUM = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
PM = NUM + r"\s*\+/-\s*" + NUM
_log = []


def line(fname, n, pattern, label):
    """Return the regex groups of `pattern` matched against line n of fname"""
    path = os.path.join(RESULTS, fname)
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    text = lines[n - 1]
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"{fname}:{n} does not match {pattern!r}\n  line: {text!r}")
    vals = tuple(float(g) for g in m.groups())
    _log.append((label, vals, f"{fname}:{n}"))
    return vals


def channels():
    """Figure 2: out-of-sample R^2 by attacker, configuration A, k = 0.17763"""
    raw_m, raw_se, norm_m, norm_se = line(
        "sniffer_skew_results.txt", 25,
        r"0\.17763\s+" + PM + r"\s+" + PM, "raw skew; normalized skew")[:4]
    full_m, full_se = line("sniffer_skew_results.txt", 39,
                           r"k=0\.17763\s+R2 = " + PM, "full-book mid")
    oracle = line("sniffer_fill_only_results.txt", 24,
                  r"0\.17763\s+" + NUM, "oracle (counterparty identity)")[0]
    tape = line("sniffer_fill_only_results.txt", 32,
                r"0\.17763\s+" + PM + r"\s+" + PM + r"\s+" + PM,
                "tape: univariate; multivariate; windows-only")
    return {
        "oracle": (oracle, 0.0),
        "norm": (norm_m, norm_se),
        "raw": (raw_m, raw_se),
        "full": (full_m, full_se),
        "tape": (tape[4], tape[5]),
    }


def smoothing():
    """Figure 3(a): R^2 vs averaging window M, 8 seeds. Figure 3(b): excess climb M=1 -> 300 vs the undefended arm, 24 seeds."""
    M, base, jit, qua = [], ([], []), ([], []), ([], [])
    for n in range(37, 43):
        v = line("phase7_nobs_results.txt", n,
                 r"^\s+(\d+)\s+" + PM + r"\s+" + PM + r"\s+" + PM,
                 f"curve row {n - 36}")
        M.append(int(v[0]))
        for (mu, se), (a, b) in zip((base, jit, qua), ((v[1], v[2]), (v[3], v[4]), (v[5], v[6]))):
            mu.append(a)
            se.append(b)
    levels = ["LOW", "COMMITTED", "HIGH"]
    jc, qc = ([], []), ([], [])
    for i, lv in enumerate(levels):
        a = line("phase8_toxicity_results.txt", 87 + i,
                 lv + r"\s+BASELINE ref " + PM, f"jitter excess climb {lv}")
        b = line("phase8_toxicity_results.txt", 91 + i,
                 lv + r"\s+BASELINE ref " + PM, f"quantized excess climb {lv}")
        jc[0].append(a[0]); jc[1].append(a[1])
        qc[0].append(b[0]); qc[1].append(b[1])
    lv_vals = [line("phase8_toxicity_results.txt", 78 + i, lv + r"\s+" + NUM, f"informed rate / lambda, {lv}")[0]
               for i, lv in enumerate(levels)]
    return {"M": M, "base": base, "jit": jit, "qua": qua,
            "levels": lv_vals, "jit_climb": jc, "qua_climb": qc}


def width():
    """Figure 4: the width sweep, 24 seeds"""
    fname = "phase7_width_results.txt"
    ws = [0.75, 1.00, 1.25, 1.50]
    caps, r2, r2se = [], [], []
    for i, w in enumerate(ws):
        caps.append(line(fname, 9 + i, rf"w/sd = {w:.2f}.*analytic cap {NUM}", f"cap {w}")[0])
        m, se = line(fname, 91 + i, r"R2\(mul\)=\s*" + PM, f"R2 mean {w}")
        r2.append(m); r2se.append(se)
    flat_m, flat_se = line(fname, 96, r"FLAT\s+measured.*R2\(mul\)=\s*" + PM, "zero skew R2")
    # per-seed R^2 for every arm (section A)
    seeds = {}
    with open(os.path.join(RESULTS, fname), encoding="utf-8-sig") as f:
        for n, text in enumerate(f.read().splitlines(), start=1):
            m = re.match(r"\s+(BASELINE|Q-\d\.\d\d|FLAT)\s+(\d+)\s+" + NUM + r"\s", text)
            if m and 100 < n < 300:
                seeds.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
    for k in ("Q-0.75", "Q-1.00", "Q-1.25", "Q-1.50", "FLAT"):
        assert len(seeds[k]) == 24, (k, len(seeds[k]))
    _log.append(("per-seed R2, 5 arms x 24 seeds", (len(seeds),), f"{fname}:128-276"))
    # leg 2: R_direct gain over zero skew
    gain, gse = [], []
    with open(os.path.join(RESULTS, fname), encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    gain_ws = ws + [1.75]
    for w in gain_ws:
        hdr = next(i for i, t in enumerate(lines) if re.match(rf"\s+Q-{w:.2f}\s+w/sd {w:.2f}", t))
        n = next(i for i in range(hdr, hdr + 8) if "LEG 2 CHEAPER/FLAT" in lines[i]) + 1
        m, se = line(fname, n, r"d =\s*" + PM, f"R_direct gain {w}")
        gain.append(m); gse.append(se)
    return {"w": ws, "cap": caps, "r2": r2, "r2se": r2se, "flat": (flat_m, flat_se),
            "seeds": seeds, "gain_w": gain_ws, "gain": gain, "gain_se": gse}


def calibration():
    """Figure 1: calibration targets"""
    rows = ["median width bp", "levels/side", "aggTrades/s", "mean/median shape"]
    absent, bands = [], []
    for i, r in enumerate(rows):
        v = line("clipped_confirm_results.txt", 9 + i,
                 re.escape(r) + r"\s+" + PM + r".*band \[\s*" + NUM + r",\s*" + NUM + r"\]",
                 f"maker absent: {r}")
        absent.append((v[0], v[1])); bands.append((v[2], v[3]))
    cb_rows = ["median width bp", "levels/side", "aggTrades/s", "mean/median shape"]
    cfg_b, cfg_c = [], []
    for i, r in enumerate(cb_rows):
        v = line("cluster_b_calibration_results.txt", 44 + i,
                 re.escape(r) + r"\s+" + PM + r".*?" + PM + r"\s+" + PM + r"\s*$",
                 f"maker present B and C: {r}")
        cfg_b.append((v[0], v[1])); cfg_c.append((v[4], v[5]))
    real_rows = ["width bp", "levels/side", "aggTrades/s", "mean/median shape"]
    real_lines = [118, 119, 121, 122]
    real = [line("phase8_toxicity_results.txt", n, re.escape(r) + r"\s+" + NUM, f"real data: {r}")[0]
            for r, n in zip(real_rows, real_lines)]
    return {"titles": ["Touch width (bp)", "Price levels per side", "Trades per second",
                       "Spread mean / median"],
            "absent": absent, "cfg_c": cfg_c, "cfg_b": cfg_b, "bands": bands, "real": real}


def all_data():
    return {"channels": channels(), "smoothing": smoothing(),
            "width": width(), "calibration": calibration()}


if __name__ == "__main__":
    d = all_data()
    for label, vals, src in _log:
        print(f"{src:42s} {label:48s} {vals}")
    print(f"\n{len(_log)} source lines read; every pattern matched.")
