# cluster_b_calibration.py: the four calibration targets in the damage configuration (cluster B), maker present
# - cluster B: gamma 1.3392e-06, 604800 s; k 0.17763, quote size 0.020, JOIN clipping, seeds 0-7
# - targets: median width, levels/side, aggTrades/s, mean/median shape; scored by clipped_window.verdict
# - cluster_b() patches quotesize_verdict's T and GAMMA_CAP for one call (restored in a finally)
# - gates, halt on failure: anchor (unpatched measure(0.020) reproduces its committed row),
#   mirror (patched run == damage_qs12.run_ext at seed 0, to 1e-9), then score
# - readings: MET / MISSED / UNRESOLVED per target; derived width 0.389 bp confirmed iff the
#   measured median rounds to 0.389 at 3 dp, else replaced by the measured value
# - limitation: 8 seeds (confirmation used 15)

import contextlib
import io
import math
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import quotesize_verdict as QV
import damage_qs12 as QS
import damage_ab as DA
from clipped_window import verdict, band, margin_se, TARGETS
from clipped_mm_check import ABSENT

QSIZE = 0.020
MIRROR_TOL = 1e-9
DERIVED_WIDTH = 0.389
COMMITTED_FILE = os.path.join(_HERE, "quotesize_verdict_results.txt")


def _finite(x, what):
    if x is None or not math.isfinite(x):
        raise ValueError("NON-FINITE %s: %r -- refusing to continue" % (what, x))
    return x


@contextlib.contextmanager
def cluster_b():
    """Context manager: quotesize_verdict run length and gamma set to cluster B's, restored after."""
    saved = (QV.T, QV.GAMMA_CAP)
    QV.T, QV.GAMMA_CAP = DA.T, DA.GAMMA
    try:
        yield
    finally:
        QV.T, QV.GAMMA_CAP = saved


def check_identity():
    assert (QV.LAM, QV.P_MARKET, QV.LIFE, QV.DISP) == (DA.LAM, DA.P_MARKET,
                                                      DA.LIFE, DA.DISP)
    assert QV.K == DA.K == 0.17763
    assert list(QV.SEEDS) == list(DA.SEEDS) == list(range(8))
    assert DA.T == 604800.0 and abs(DA.GAMMA - 1.3392e-06) < 1e-9
    assert QV.T == 86400.0 and QV.GAMMA_CAP == 9.4e-6
    assert [k for _n, k, _t, _o in TARGETS] == ["median_bp", "levels_per_side",
                                                "agg_per_s", "shape"]


# committed 0.020 row
def _tol(s):
    return 0.5 * 10.0 ** (-len(s.split(".")[1])) if "." in s else 0.5


def parse_committed(lines):
    """{field: (value, tol)} for quote size 0.020 from the committed results."""
    out = {}
    tline = [ln for ln in lines if re.match(r"^\s+quote_size=0\.020 BTC\s+measured",
                                            ln)]
    if len(tline) != 1:
        raise ValueError("expected one 0.020 timing line, found %d" % len(tline))
    m = re.search(r"fill share=\s*([0-9.]+)%\s+vol share=\s*([0-9.]+)%\s+"
                  r"sd\(q\)=([0-9.]+)", tline[0])
    out["fill_share"] = (float(m.group(1)), _tol(m.group(1)))
    out["vol_share"] = (float(m.group(2)), _tol(m.group(2)))
    out["sd_q"] = (float(m.group(3)), _tol(m.group(3)))
    start = [i for i, ln in enumerate(lines) if ln.strip() == "quote_size=0.020"]
    if not start:
        raise ValueError("no 'quote_size=0.020' target block")
    names = {n: k for n, k, _t, _o in TARGETS}
    for ln in lines[start[0] + 1:start[0] + 5]:
        mm = re.match(r"^\s+(median width bp|levels/side|aggTrades/s|"
                      r"mean/median shape)\s+([0-9.]+) \+/- ([0-9.]+)", ln)
        if not mm:
            raise ValueError("target line did not parse: %r" % ln)
        k = names[mm.group(1)]
        out[k] = (float(mm.group(2)), _tol(mm.group(2)))
        out[k + "_se"] = (float(mm.group(3)), _tol(mm.group(3)))
    if len(out) != 11:
        raise ValueError("parsed %d fields, expected 11" % len(out))
    return out


def anchor_check(r, committed):
    fails = []
    for k, (want, tol) in committed.items():
        got = _finite(r[k], k)
        ok = abs(got - want) <= tol
        if not ok:
            fails.append(k)
        print("  %-18s got %12.6f  committed %12.5f  tol %.0e  %s"
              % (k, got, want, tol, "OK" if ok else "*** miss ***"))
    return fails


# scoring
def score(r, cluster_c):
    """Cluster B verdicts beside the maker-absent and cluster-C rows. Returns ({key: verdict}, width call)."""
    print("  %-18s %-34s %-10s %-9s %-24s %-24s"
          % ("target", "Cluster B, maker present (8)", "verdict", "margin",
             "maker absent (15, a8f801d)", "cluster C, maker (8)"))
    out = {}
    for name, k, tgt, tol in TARGETS:
        pt, se = _finite(r[k], k), _finite(r[k + "_se"], k + " SE")
        if not se > 0:
            raise ValueError("%s: SE is %r -- no interval" % (k, se))
        v = verdict(pt, se, tgt, tol)
        lo, hi = band(tgt, tol)
        mg = margin_se(pt, se, tgt, tol)
        out[k] = v
        print("  %-18s %10.5f +/- %-9.5f [%.4f, %.4f] %-10s %6.2f SE  "
              "%10.5f +/- %-9.5f %10.5f +/- %-9.5f"
              % (name, pt, se, lo, hi, v, mg, ABSENT[k], ABSENT[k + "_se"],
                 cluster_c[k][0], cluster_c[k + "_se"][0]))
    met = sum(1 for v in out.values() if v == "MET")
    missed = sum(1 for v in out.values() if v == "MISSED")
    print("  -> %d/4 MET, %d MISSED, %d UNRESOLVED" % (met, missed,
                                                      4 - met - missed))
    w = r["median_bp"]
    confirmed = round(w, 3) == DERIVED_WIDTH
    print("  derived width 0.389 bp (2 x touch_geometry's 0.1947): measured "
          "%.5f bp -> %s" % (w, "confirmed" if confirmed else
                             "replaced by the measured value"))
    return out, confirmed


# self-tests
def t_patch():
    with cluster_b():
        assert QV.T == 604800.0 and abs(QV.GAMMA_CAP - 1.3392e-06) < 1e-9
    assert QV.T == 86400.0 and QV.GAMMA_CAP == 9.4e-6
    try:
        with cluster_b():
            raise RuntimeError("planted")
    except RuntimeError:
        pass
    assert QV.T == 86400.0 and QV.GAMMA_CAP == 9.4e-6
    print("  T-patch     PASS  cluster_b() sets T=604800 and gamma=1.3392e-06, "
          "and restores 86400 / 9.4e-06,")
    print("                    including after a planted exception.")


def t_parse():
    with open(COMMITTED_FILE, encoding="utf-8-sig") as f:
        c = parse_committed(f.read().splitlines())
    assert c["median_bp"] == (0.32138, 5e-6) and c["levels_per_side"][0] == 550.98455
    assert c["agg_per_s"][0] == 0.04083 and c["shape_se"][0] == 0.08618
    assert c["sd_q"][0] == 0.04604 and c["fill_share"] == (5.80, 5e-3)
    print("  T-parse     PASS  the committed 0.020 row parses: width 0.32138, "
          "levels 550.98455, rate 0.04083,")
    print("                    shape SE 0.08618, sd(q) 0.04604, fill 5.80%; "
          "only the first 0.020 block is read.")
    return c


def t_score(c):
    planted = {"median_bp": 0.389, "median_bp_se": 0.005,
               "levels_per_side": 550.0, "levels_per_side_se": 2.5,
               "agg_per_s": 0.0408, "agg_per_s_se": 0.0007,
               "shape": 2.2, "shape_se": 0.08}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v, conf = score(planted, c)
    assert v == {"median_bp": "MISSED", "levels_per_side": "MET",
                 "agg_per_s": "UNRESOLVED", "shape": "MET"}, v
    assert conf
    p2 = dict(planted, median_bp=0.3905)
    with contextlib.redirect_stdout(buf):
        _v, conf2 = score(p2, c)
    assert not conf2
    for bad in (dict(planted, shape=float("nan")), dict(planted, agg_per_s_se=0.0)):
        try:
            with contextlib.redirect_stdout(buf):
                score(bad, c)
            raise AssertionError("score accepted a bad input")
        except ValueError:
            pass
    print("  T-score     PASS  score(); the function main() uses; on planted "
          "values: 0.389 +/- 0.005 misses,")
    print("                    levels and shape MET, rate UNRESOLVED; 0.389 "
          "confirmed, 0.3905 not; NaN and")
    print("                    zero SE raise. Output captured, not logged.")


def selftest():
    print("=" * 118)
    print("Self-tests; cluster_b_calibration.py. No run yet.")
    print("=" * 118)
    check_identity()
    print("  T-IDENT     PASS  working point, k = 0.17763 and seeds 0-7 equal "
          "damage_ab's; clipped_window.TARGETS keys as expected.")
    t_patch()
    c = t_parse()
    t_score(c)
    print("")
    print("  All self-tests PASSED.")
    return c


def main(committed):
    print("=" * 118)
    print("Cluster B, maker present; the four calibration targets at k = 0.17763, "
          "8 seeds x 604800 s")
    print("=" * 118)
    print("gamma=%.4e (damage_ab.GAMMA), quote size %.3f, lam=%.3f p_market=%.4f "
          "life=%.0f disp=%.4f JOIN." % (DA.GAMMA, QSIZE, DA.LAM, DA.P_MARKET,
                                         DA.LIFE, DA.DISP))
    print("Scored by clipped_window.verdict. Readings fixed in the header.")
    print("")

    print("Gate 1; anchor: unpatched quotesize_verdict.measure(0.020) vs its "
          "committed row", flush=True)
    t0 = time.time()
    r0 = QV.measure(QSIZE)
    fails = anchor_check(r0, committed)
    if fails:
        raise AssertionError("ANCHOR FAILED on %s -- halting" % ", ".join(fails))
    print("  anchor HELD (%.0fs)." % (time.time() - t0))
    print("")

    print("Gate 2; mirror: patched quotesize_verdict.run vs damage_qs12.run_ext, "
          "seed 0, cluster B", flush=True)
    t0 = time.time()
    with cluster_b():
        a = QV.run(0, QSIZE)
    b = QS.run_ext(0, False, QS.THETA, QS.HOLD, QSIZE)
    for ka, kb in (("mm_spread", "mm_spread_med"), ("sd_q", "sd_q")):
        ok = abs(a[ka] - b[kb]) <= MIRROR_TOL * max(1.0, abs(b[kb]))
        print("  %-10s patched run %.12f   run_ext %.12f   %s"
              % (ka, a[ka], b[kb], "OK" if ok else "*** mismatch ***"))
        if not ok:
            raise AssertionError("MIRROR FAILED on %s -- halting" % ka)
    print("  mirror HELD (%.0fs): the patch produces the damage runs' world."
          % (time.time() - t0))
    print("")

    print("Score; patched quotesize_verdict.measure(0.020), seeds 0-7", flush=True)
    t0 = time.time()
    with cluster_b():
        r = QV.measure(QSIZE, seeds=DA.SEEDS)
    print("  8 seven-day runs in %.0fs" % (time.time() - t0))
    print("")
    score(r, committed)
    print("")
    print("  also: maker spread $%.4f +/- %.4f, sd(q) %.5f +/- %.5f, mean width "
          "%.4f bp" % (r["mm_spread"], r["mm_spread_se"], r["sd_q"], r["sd_q_se"],
                       r["mean_bp"]))


if __name__ == "__main__":
    c = selftest()
    if "--selftest" in sys.argv:
        sys.exit(0)
    print("")
    main(c)
