# sd_ytest.py: sd_n(y_test) (population SD of maker inventory on each seed's test slice) for the
# Phase 7 configurations cited in the paper; puts R^2 on a BTC scale: rmse = sd_n * sqrt(1 - R^2)
# - ARM A: phase7_blind BASELINE, QUANT-LOW, QUANT-HIGH, FLAT; 8 seeds x 86400 s, cluster C
# - ARM B: phase7_nobs BASELINE, JITTER, QUANT; 8 seeds (y_test independent of window M)
# - WIDTH: phase7_width BASELINE, Q-0.75 .. Q-1.75, FLAT; 24 seeds
# - test slice captured by phase7_width's committed fit_score wrapper (installed on
#   phase7_nobs.fit_score for ARM B, then restored)
# halts unless: committed R^2 mean/SE (and R_direct) reproduce to half the last printed digit;
# rmse / sd_n = sqrt(1 - R^2) to relative 1e-6 where 1 - R^2 > 1e-6; wrapper is transparent
# runtime ~31 min

import contextlib
import io
import math
import os
import re
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import phase7_width as PW
import phase7_blind as BLIND
import phase7_nobs as NOBS

IDENTITY_MIN_GAP = 1e-6
IDENTITY_REL_TOL = 1e-6
FILES = {"ARM A": "phase7_blind_results.txt",
         "ARM B": "phase7_nobs_results.txt",
         "WIDTH": "phase7_width_results.txt"}
CONFIGS = (("ARM A", BLIND.ARMS, BLIND.SEEDS),
           ("ARM B", NOBS.ARMS, NOBS.SEEDS),
           ("WIDTH", PW.ARMS, PW.SEEDS))


def _finite(x, what):
    if x is None or not math.isfinite(x):
        raise ValueError("non-finite %s: %r; halting" % (what, x))
    return x


# --------------------------------------------------------------------------- #
# the runs
# --------------------------------------------------------------------------- #
def fit_stats(r2, yte, pred, what):
    """sd_n, n_te, RMSE of one captured fit; checks the identity."""
    n = len(yte)
    if n < 2 or len(pred) != n:
        raise ValueError("%s: test slice of %d values, %d predictions"
                         % (what, n, len(pred)))
    sd_n = statistics.pstdev(yte)
    if not _finite(sd_n, "sd_n " + what) > 0:
        raise ValueError("%s: sd_n(y_test) is zero" % what)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(yte, pred)) / n)
    _finite(rmse, "rmse " + what)
    _finite(r2, "R2 " + what)
    gap = 1.0 - r2
    if gap > IDENTITY_MIN_GAP:
        want = math.sqrt(gap)
        got = rmse / sd_n
        if abs(got - want) > IDENTITY_REL_TOL * want:
            raise ValueError("%s: rmse/sd_n = %.12g but sqrt(1-R2) = %.12g"
                             % (what, got, want))
    return {"r2": r2, "sd_n": sd_n, "n_te": n, "rmse": rmse}


def _captured(what):
    cap = PW._CAPTURE
    if "yte" not in cap or "pred" not in cap:
        raise ValueError("%s: nothing was captured" % what)
    return list(cap["yte"]), list(cap["pred"])


def run_blind(seed, arm, config):
    """ARM A and WIDTH, as phase7_width.main calls run_diag."""
    if config == "WIDTH" and arm not in (PW.BASE, PW.FLAT):
        w = PW.WIDTHS[PW.WIDTH_ARMS.index(arm)]
        out = PW.run_diag(seed, arm, mm=PW.make_width_mm(w))
    else:
        out = PW.run_diag(seed, arm)
    what = "%s %s seed %d" % (config, arm, seed)
    yte, pred = _captured(what)
    row = fit_stats(out["r2_mul"], yte, pred, what)
    row["r_direct"] = _finite(out["r_direct"], "R_direct " + what)
    return row


def run_nobs(seed, arm, horizon=None):
    """ARM B run with the capturing wrapper installed."""
    if NOBS.fit_score is not PW.fit_score:
        raise ValueError("phase7_nobs.fit_score is not the committed object")
    PW._CAPTURE.clear()
    NOBS.fit_score = PW._capturing_fit_score
    try:
        out = (NOBS.run(seed, arm) if horizon is None
               else NOBS.run(seed, arm, horizon=horizon))
    finally:
        NOBS.fit_score = PW.fit_score
    what = "ARM B %s seed %d" % (arm, seed)
    yte, pred = _captured(what)
    row = fit_stats(out["r2_unsmoothed"], yte, pred, what)
    row["n_kept_total"] = out["n_kept_total"]
    return row, out


def run_config(config, arms, seeds):
    res = {}
    for arm in arms:
        t0 = time.time()
        rows = []
        for s in seeds:
            if config == "ARM B":
                row, _o = run_nobs(s, arm)
            else:
                row = run_blind(s, arm, config)
            row["seed"] = s
            rows.append(row)
        res[arm] = rows
        print("  %-6s %-10s %d seeds in %4.0fs" % (config, arm, len(seeds),
                                                  time.time() - t0), flush=True)
    return res


# --------------------------------------------------------------------------- #
# anchor and report
# --------------------------------------------------------------------------- #
def _tol(s):
    return 0.5 * 10.0 ** (-len(s.split(".")[1])) if "." in s else 0.5


def parse_anchor(config, lines):
    """{arm: {field: (value, tol)}} from committed timing lines."""
    out = {}
    if config == "ARM B":
        pat = re.compile(r"^\s+(\S+)\s+measured in\s+\d+s\s+unsmoothed R2 = "
                         r"(-?[0-9.]+) \+/- ([0-9.]+)\s+kept = (\d+)\s*$")
    else:
        pat = re.compile(r"^\s+(\S+)\s+measured in\s+\d+s\s+R2\(mul\)=\s*"
                         r"(-?[0-9.]+)\+/-([0-9.]+)\s+R_direct=([0-9.]+)"
                         r"\+/-([0-9.]+)")
    for ln in lines:
        m = pat.match(ln)
        if not m:
            continue
        g = m.groups()
        if g[0] in out:
            raise ValueError("%s: arm %s appears twice" % (config, g[0]))
        d = {"r2": (float(g[1]), _tol(g[1])), "r2_se": (float(g[2]), _tol(g[2]))}
        if config == "ARM B":
            d["kept"] = (float(g[3]), 0.5)
        else:
            d["rd"] = (float(g[3]), _tol(g[3]))
            d["rd_se"] = (float(g[4]), _tol(g[4]))
        out[g[0]] = d
    return out


def anchor_check(config, res, committed):
    fails = []
    for arm, rows in res.items():
        if arm not in committed:
            raise ValueError("%s: arm %s has no committed line" % (config, arm))
        got = {}
        got["r2"], got["r2_se"] = BLIND.mean_se([r["r2"] for r in rows])
        if config == "ARM B":
            got["kept"] = rows[0]["n_kept_total"]
        else:
            got["rd"], got["rd_se"] = BLIND.mean_se([r["r_direct"] for r in rows])
        for k, (want, tol) in committed[arm].items():
            ok = abs(got[k] - want) <= tol
            if not ok:
                fails.append("%s %s %s" % (config, arm, k))
            print("  %-6s %-10s %-6s got %+.9f  committed %+.6f  tol %.0e  %s"
                  % (config, arm, k, got[k], want, tol,
                     "OK" if ok else "*** MISS ***"))
    return fails


def report(all_res):
    summary = {}
    for config, res in all_res.items():
        print("")
        print("  %s per seed: R2 | sd_n(y_test) BTC | n_te | RMSE BTC" % config)
        for arm, rows in res.items():
            for r in rows:
                print("    %-10s seed %2d  R2 %16.6f   sd_n %.6f   n_te %6d   "
                      "rmse %.6f" % (arm, r["seed"], r["r2"], r["sd_n"],
                                     r["n_te"], r["rmse"]))
    print("")
    print("  summary: mean +/- SE across seeds (BLIND.mean_se)")
    print("  %-6s %-10s %6s %22s %20s %22s %8s"
          % ("config", "arm", "seeds", "sd_n(y_test) BTC", "sd_n min-max",
             "RMSE BTC", "n_te"))
    for config, res in all_res.items():
        for arm, rows in res.items():
            sm, sse = BLIND.mean_se([r["sd_n"] for r in rows])
            rm, rse = BLIND.mean_se([r["rmse"] for r in rows])
            lo = min(r["sd_n"] for r in rows)
            hi = max(r["sd_n"] for r in rows)
            nte = statistics.median([r["n_te"] for r in rows])
            summary[(config, arm)] = {"sd_n": (sm, sse), "rmse": (rm, rse),
                                      "range": (lo, hi)}
            print("  %-6s %-10s %6d   %.6f +/- %.6f   %.5f-%.5f   "
                  "%.6f +/- %.6f %8.0f"
                  % (config, arm, len(rows), sm, sse, lo, hi, rm, rse, nte))
    return summary


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def t_fit_stats():
    yte = [0.01 * i for i in range(10)]
    pred = [0.01 * i + (0.002 if i % 2 else -0.002) for i in range(10)]
    m = statistics.mean(yte)
    ss_tot = sum((a - m) ** 2 for a in yte)
    ss_res = sum((a - b) ** 2 for a, b in zip(yte, pred))
    r2 = 1.0 - ss_res / ss_tot
    s = fit_stats(r2, yte, pred, "planted")
    assert abs(s["sd_n"] - statistics.pstdev(yte)) < 1e-15
    assert abs(s["rmse"] - 0.002) < 1e-12
    for bad in ((r2 - 0.1, yte, pred), (float("nan"), yte, pred),
                (r2, [0.5] * 10, pred)):
        try:
            fit_stats(bad[0], bad[1], bad[2], "planted bad")
            raise AssertionError("fit_stats accepted a bad input")
        except ValueError:
            pass
    print("  T-FITSTATS  PASS")


def t_parse():
    counts = {}
    for config, arms, _seeds in CONFIGS:
        with open(os.path.join(_HERE, FILES[config]), encoding="utf-8-sig") as f:
            a = parse_anchor(config, f.read().splitlines())
        assert set(a) == set(arms), (config, sorted(a), arms)
        counts[config] = len(a)
    with open(os.path.join(_HERE, FILES["WIDTH"]), encoding="utf-8-sig") as f:
        w = parse_anchor("WIDTH", f.read().splitlines())
    assert w["Q-0.75"]["r2"] == (0.555334, 5e-7), w["Q-0.75"]["r2"]
    assert w["Q-1.75"]["r2"][0] == -28181.98175
    with open(os.path.join(_HERE, FILES["ARM B"]), encoding="utf-8-sig") as f:
        b = parse_anchor("ARM B", f.read().splitlines())
    assert b["JITTER"]["r2"][0] == -10.79098 and b["QUANT"]["kept"][0] == 84672
    print("  T-PARSE     PASS  arms parsed: Arm A "
          "%d, Arm B %d, width %d"
          % (counts["ARM A"], counts["ARM B"], counts["WIDTH"]))


def t_postsim():
    def rows(n, r2, sd):
        return [{"seed": i, "r2": r2 + 1e-3 * (2 * (i % 2) - 1),
                 "sd_n": sd + 1e-4 * (i % 4), "n_te": 25000,
                 "rmse": sd * 0.5 + 1e-4 * (i % 2), "r_direct": 0.2 + 1e-3 * i,
                 "n_kept_total": 84672.0} for i in range(n)]
    res = {"ARM B": {"BASELINE": rows(8, 1.0, 0.04), "JITTER": rows(8, -10.0, 0.04)}}
    committed = {"BASELINE": {"r2": (1.0, 5e-7), "r2_se": (0.0, 5e-7),
                              "kept": (84672.0, 0.5)},
                 "JITTER": {"r2": (-10.0, 5e-7), "r2_se": (0.0, 5e-7),
                            "kept": (84672.0, 0.5)}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fails = anchor_check("ARM B", res["ARM B"], committed)
        summ = report(res)
    assert fails == ["ARM B BASELINE r2_se", "ARM B JITTER r2_se"], fails
    assert abs(summ[("ARM B", "JITTER")]["sd_n"][0] - (0.04 + 1e-4 * 1.5)) < 1e-12
    print("  T-POSTSIM   PASS")


def t_nobs_transparent():
    """ARM B wrapper leaves phase7_nobs.run output unchanged; JITTER so the identity check runs.
    - wrapper keeps the first fit_score call (the r2_unsmoothed fit)"""
    t0 = time.time()
    arm = "JITTER"
    plain = NOBS.run(0, arm, horizon=BLIND.SELFTEST_SHORT)
    row, wrapped = run_nobs(0, arm, horizon=BLIND.SELFTEST_SHORT)
    assert NOBS.fit_score is PW.fit_score, "phase7_nobs.fit_score not restored"
    for k in plain:
        assert plain[k] == wrapped[k],             "wrapper changed %s: %r without, %r with" % (k, plain[k],
                                                            wrapped[k])
    assert row["r2"] == wrapped["r2_unsmoothed"]
    gap = 1.0 - row["r2"]
    assert gap > IDENTITY_MIN_GAP, "identity check would have been skipped"
    print("  T-NOBS-TRANSPARENT PASS  %d fields "
          "identical" % len(plain))
    print("                    seed 0, %s, %ds; "
          "identity:" % (arm, BLIND.SELFTEST_SHORT))
    print("                    R2 %.6f, rmse/sd_n %.9f vs sqrt(1-R2) %.9f. (%.0fs)"
          % (row["r2"], row["rmse"] / row["sd_n"], math.sqrt(gap),
             time.time() - t0))


def selftest():
    print("=" * 118)
    print("Self-tests")
    print("=" * 118)
    t_fit_stats()
    t_parse()
    t_postsim()
    PW.t_capture_is_transparent()
    t_nobs_transparent()
    print("")
    print("  All self-tests passed.")
    return 0


def main():
    print("=" * 118)
    print("sd_n(y_test): Phase 7 Arm A (8 seeds), Arm B (8 seeds), width sweep "
          "(24 seeds); cluster C, 86400s")
    print("=" * 118)
    print("")
    all_res = {}
    for config, arms, seeds in CONFIGS:
        with open(os.path.join(_HERE, FILES[config]), encoding="utf-8-sig") as f:
            committed = parse_anchor(config, f.read().splitlines())
        print("running %s ..." % config, flush=True)
        res = run_config(config, arms, seeds)
        print("ANCHOR %s vs %s" % (config, FILES[config]))
        fails = anchor_check(config, res, committed)
        if fails:
            raise AssertionError("ANCHOR FAILED on %s; halting"
                                 % ", ".join(fails))
        print("  ANCHOR HELD for %s." % config)
        print("")
        all_res[config] = res
    report(all_res)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
