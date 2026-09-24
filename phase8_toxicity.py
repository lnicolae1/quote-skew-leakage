# phase8_toxicity.py: Phase 8 axis 3: does the defense result survive toxicity (informed:noise ratio)?
# - levels LOW 0.02, COMMITTED 0.10, HIGH 0.30; arms BASELINE / JITTER / QUANT; 24 seeds x 86400 s; M 1 -> 300
# - BASELINE-referenced climbs (phase8_reference); defenses recalibrated per level from its own measured sd(q)
#   (COMMITTED uses SD_Q_REF so the anchor is reachable)
# - pre-registered, Z = 2.0 (decisions between 2.0 and t(23) = 2.069 flagged):
#   R-JITTER: JITTER-minus-BASELINE climb > 0 at 2.0 SE (one-sided)
#   R-QUANT: |QUANT-minus-BASELINE| < 2.0 SE (two-sided; corrected from one-sided before the run)
#   AXIS 3 HELD iff R-JITTER and R-QUANT hold at all three levels
#   R2: no seed > 40% of the squared deviation in a scored cell; R-CONTRAST reported only
# - anchors: COMMITTED reproduces phase8_reference's HIGH row (96 per-seed climbs);
#   drift anchor: quotesize_extend's 0.020 row reproduces through the patch

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

import clipped_placement
import clipped_gamma_sweep as CGS
import phase7_blind as BLIND
import phase7_nobs as NOBS
import phase8_gamma as GAM
import phase8_reference as REF
import quotesize_extend as QX
import sim_run

from informed_traders import (InformedTraderFlow, DEFAULT_LAMBDA_NOISE,
                              DEFAULT_LAMBDA_INFORMED, TOXICITY_PRESETS)
from phase8_gamma import calibrate, make_defended
from phase7_nobs import JitteredMM
from phase7_blind import QuantizedMM, mean_se, require_finite
from clipped_window import TARGETS
from clipped_depth_sweep import REAL_NEAR_BTC
from damage_qs12 import _finite, _ratio

GAMMA = GAM.GAMMA_HIGH
SEEDS_24 = list(REF.SEEDS_24)
DRIFT_SEEDS = list(range(8))
QSIZE = NOBS.QUOTE_SIZE
Z_SE = NOBS.Z_SE
T23 = 2.069
DOM_BAR = REF.DOM_BAR

COMMITTED_RATIO = TOXICITY_PRESETS["medium"]
LEVELS = (("COMMITTED", TOXICITY_PRESETS["medium"]),
          ("LOW", TOXICITY_PRESETS["low"]),
          ("HIGH", TOXICITY_PRESETS["high"]))
REPORT_ORDER = ("LOW", "COMMITTED", "HIGH")

REF_RESULTS = "phase8_reference_results.txt"
QX_RESULTS = "quotesize_extend_results.txt"
RESULTS_PATH = "phase8_toxicity_results.txt"

PATCHED_MODULES = (NOBS, QX, CGS)
_REAL_MAKE_WORLD = clipped_placement.make_world


# toxicity injection
def _world_at(ratio):
    """make_world with the informed flow rebuilt at ratio."""
    def mw(*a, **kw):
        path, vf, eng, noise, _discarded = _REAL_MAKE_WORLD(*a, **kw)
        seed = a[0] if a else kw["seed"]
        informed = InformedTraderFlow(
            lam_informed=ratio * DEFAULT_LAMBDA_NOISE,
            seed=sim_run.SEED_INF_BASE + 1000 * seed)
        return path, vf, eng, noise, informed
    mw.ratio = ratio
    return mw


@contextlib.contextmanager
def informed_at(ratio):
    saved = [(m, m.make_world) for m in PATCHED_MODULES]
    for m in PATCHED_MODULES:
        if m.make_world is not _REAL_MAKE_WORLD:
            raise RuntimeError("%s.make_world is already rebound -- nested or "
                               "leaked patch" % m.__name__)
    wrapper = _world_at(ratio)
    try:
        for m in PATCHED_MODULES:
            m.make_world = wrapper
        yield wrapper
    finally:
        for m, orig in saved:
            m.make_world = orig


def bindings_restored():
    return all(m.make_world is _REAL_MAKE_WORLD for m in PATCHED_MODULES)


# calibration used by the arm
def strict_mean_se(values, what, n_expected):
    if len(values) != n_expected:
        raise ValueError("%s: %d values, expected %d" % (what, len(values),
                                                        n_expected))
    require_finite(what, *values)
    m, se = mean_se(values)
    _finite(m, "mean " + what)
    if not _finite(se, "SE " + what) > 0:
        raise ValueError("%s: SE %r -- no interval" % (what, se))
    return m, se


def desc_mean_se(values, what, n_expected):
    """Descriptive mean/SE for the drift table; zero SE allowed."""
    if len(values) != n_expected:
        raise ValueError("%s: %d values, expected %d" % (what, len(values),
                                                        n_expected))
    require_finite(what, *values)
    m, se = mean_se(values)
    return _finite(m, "mean " + what), _finite(se, "SE " + what)


def level_calibration(ratio, base_results):
    """(sigma_eps, w_q, sd_q_used, sd_q_measured, source) for one level."""
    measured, _se = strict_mean_se([r["sd_q"] for r in base_results],
                                   "measured sd(q)", len(base_results))
    if ratio == COMMITTED_RATIO:
        sd_used, source = NOBS.SD_Q_REF, "committed SD_Q_REF (anchor)"
    else:
        sd_used, source = measured, "measured, this level's BASELINE seeds"
    sigma_eps, w_q = calibrate(GAMMA, sd_used)
    return {"sigma_eps": sigma_eps, "w_q": w_q, "sd_q_used": sd_used,
            "sd_q_measured": measured, "sd_q_measured_se": _se,
            "source": source,
            "sd_skew_model": sigma_eps / NOBS.SIGMA_EPS_MULT}


def build_checked(arm, seed, cal):
    """make_defended plus a run-time check on the built maker."""
    mm = make_defended(arm, seed, GAMMA, cal["sigma_eps"], cal["w_q"])
    if mm.gamma != GAMMA:
        raise RuntimeError("%s seed %d built at gamma %r" % (arm, seed,
                                                            mm.gamma))
    if arm == "JITTER":
        if not isinstance(mm, JitteredMM) or mm.sigma_eps != cal["sigma_eps"]:
            raise RuntimeError("CALL-SITE GAP: JITTER seed %d runs sigma_eps "
                               "%r, level calibrated %r" % (
                                   seed, getattr(mm, "sigma_eps", None),
                                   cal["sigma_eps"]))
    elif arm == "QUANT":
        if not isinstance(mm, QuantizedMM) or mm.w_q != cal["w_q"]:
            raise RuntimeError("CALL-SITE GAP: QUANT seed %d runs w_q %r, "
                               "level calibrated %r" % (
                                   seed, getattr(mm, "w_q", None), cal["w_q"]))
    return mm


# parsing committed results (value, printed precision)
_NUM = re.compile(r"[+-]?\d+\.\d+")


def _tok(s):
    """(value, tolerance) where tolerance = 0.5 * 10^-decimals printed"""
    s = s.strip().rstrip("%x")
    dec = len(s.split(".")[1]) if "." in s else 0
    return float(s), 0.5 * 10.0 ** (-dec)


def parse_reference_high():
    with open(REF_RESULTS, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if "PER-SEED CLIMBS" in l)
    ih = next(i for i in range(i0, len(lines))
              if lines[i].strip().startswith("gamma = HIGH"))
    rows = {}
    for l in lines[ih + 2: ih + 2 + 24]:
        parts = [p.strip() for p in l.split("|")]
        seed = int(parts[0])
        b = parts[1].split()
        q = parts[2].split()
        j = parts[3].split()
        rows[seed] = {"base_climb": _tok(b[2]), "quant_climb": _tok(q[2]),
                      "jitter_climb": _tok(j[2]), "qb_climb": _tok(parts[4])}
    if sorted(rows) != list(range(24)):
        raise ValueError("parsed seeds %r, expected 0..23" % sorted(rows))

    pat = re.compile(r"(zero ref|BASELINE ref), seeds 0-23\s+([+-]\d+\.\d+) "
                     r"\+/- (\d+\.\d+)\s+[\d.]+ SE\s+worst seed\s+(\d+) "
                     r"carries\s+([\d.]+%)")
    summ = {}
    for block, arm in (("THE 2x2 -- QUANT'S CLIMB", "QUANT"),
                       ("THE SAME 2x2 FOR JITTER", "JITTER")):
        ib = next(i for i, l in enumerate(lines) if block in l)
        ig = next(i for i in range(ib, len(lines))
                  if lines[i].strip().startswith("gamma = HIGH"))
        for l in lines[ig + 1: ig + 6]:
            m = pat.search(l)
            if m:
                ref = "zero" if m.group(1) == "zero ref" else "base"
                summ[(arm, ref)] = {"mean": _tok(m.group(2)),
                                    "se": _tok(m.group(3)),
                                    "dom_seed": int(m.group(4)),
                                    "dom_share_pct": _tok(m.group(5))}
    if len(summ) != 4:
        raise ValueError("parsed %d 24-seed summaries, expected 4" % len(summ))

    cal = next(l for l in lines if l.strip().startswith("HIGH ") and "+/-" in l
               and "x" in l.split()[-1])
    nums = _NUM.findall(cal)
    calib = {"sigma_eps": _tok(nums[0]), "w_q": _tok(nums[1]),
             "sd_q_24": _tok(nums[4]), "sd_q_24_se": _tok(nums[5])}
    return rows, summ, calib


def parse_qx_committed():
    """quotesize_extend_results.txt section 1, quote_size=0.020 block, keyed via clipped_window.TARGETS."""
    with open(QX_RESULTS, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    # anchor on section 1's header first (an earlier progress line also has "[committed]")
    isec = next(i for i, l in enumerate(lines)
                if "1. THE FOUR COMMITTED TARGETS" in l)
    i0 = next(i for i in range(isec, len(lines))
              if lines[i].strip().startswith("quote_size=0.020")
              and "[committed]" in lines[i])
    out = {}
    for name, key, _tgt, _tol in TARGETS:
        l = next(x for x in lines[i0 + 1: i0 + 6] if x.strip().startswith(name))
        nums = _NUM.findall(l[len(l) - len(l.lstrip()) + len(name):])
        out[key] = {"mean": _tok(nums[0]), "se": _tok(nums[1])}
    return out


# one level
def run_level(name, ratio):
    t0 = time.time()
    res = {}
    with informed_at(ratio) as w:
        if w.ratio != ratio:
            raise RuntimeError("patch carries the wrong ratio")
        res["BASELINE"] = [
            NOBS.run(s, "BASELINE", gamma=GAMMA,
                     mm_override=make_defended("BASELINE", s, GAMMA, 0.0, 1.0))
            for s in SEEDS_24]
        cal = level_calibration(ratio, res["BASELINE"])
        for arm in ("JITTER", "QUANT"):
            res[arm] = [NOBS.run(s, arm, gamma=GAMMA,
                                 mm_override=build_checked(arm, s, cal))
                        for s in SEEDS_24]
        drift_qx = [QX.run_ext(s, QSIZE) for s in DRIFT_SEEDS]
        drift_cg = [CGS.run(s, GAMMA) for s in DRIFT_SEEDS]
    if not bindings_restored():
        raise RuntimeError("make_world binding NOT restored after level %s"
                           % name)
    for arm, rr in res.items():
        for r in rr:
            require_finite("%s %s" % (name, arm), r["sd_q"], r["n_mm_fills"])
    return {"name": name, "ratio": ratio, "res": res, "cal": cal,
            "drift_qx": drift_qx, "drift_cg": drift_cg,
            "secs": time.time() - t0}


def score_level(lv):
    base = lv["res"]["BASELINE"]
    out = {}
    for arm in ("JITTER", "QUANT"):
        cells = REF.four_cells(lv["res"][arm], base)
        out[arm] = {"zero": cells[("zero", 24)], "base": cells[("base", 24)]}
    out["base_climb"] = REF.climb_series(base)
    # R-CONTRAST: (J-B) - (Q-B) per seed = paired JITTER-minus-QUANT climb
    out["CONTRAST"] = REF.cell(
        "JITTER-minus-QUANT, seeds 0-23",
        REF.climb_series(REF.diff_curve(lv["res"]["JITTER"],
                                        lv["res"]["QUANT"])), SEEDS_24)
    return out


def call(c, rule):
    """Verdict for one cell. Returns (ok, |t|, between_2.0_and_t23, direction_if_failed)."""
    m, se = c["mean"], c["se"]
    t = _ratio(abs(m), se, "%s |mean|/SE" % rule)
    if rule in ("R-JITTER", "R-CONTRAST"):
        ok = (m > 0) and (t >= Z_SE)
        between = (Z_SE <= t < T23) and (m > 0)
        direction = None if ok else ("non-positive" if m <= 0
                                     else "positive but < 2.0 SE")
    elif rule == "R-QUANT":
        ok = t < Z_SE
        between = Z_SE <= t < T23
        direction = None if ok else (
            "RECOVERY -- the defense fails" if m > 0
            else "degrades beyond the smoother -- not flat, NOT a defense "
                 "failure")
    else:
        raise ValueError("unknown rule %r" % (rule,))
    return ok, t, between, direction


# self-tests
def t_ident():
    assert calibrate is GAM.calibrate and make_defended is GAM.make_defended
    assert REF.calibrate is GAM.calibrate
    assert GAMMA == NOBS.GAMMA_CAP == 9.4e-6
    assert COMMITTED_RATIO * DEFAULT_LAMBDA_NOISE == DEFAULT_LAMBDA_INFORMED, \
        "the committed ratio does not rebuild the default informed rate"
    assert [r for _n, r in LEVELS] == [0.10, 0.02, 0.30]
    assert all(0.001 <= r <= 0.30 for _n, r in LEVELS), "outside the sweep"
    assert QSIZE == 0.02 and Z_SE == 2.0 and DOM_BAR == 0.40
    assert len(SEEDS_24) == 24 and SEEDS_24 == list(range(24))
    assert DRIFT_SEEDS == list(range(8))
    assert bindings_restored()
    print("  T-IDENT       PASS  calibrate/make_defended are phase8_gamma's; "
          "gamma 9.4e-6; levels are")
    print("                      TOXICITY_PRESETS low/medium/high = 0.02/0.10/"
          "0.30, all inside the sweep;")
    print("                      0.10 * DEFAULT_LAMBDA_NOISE == "
          "DEFAULT_LAMBDA_INFORMED EXACTLY.")


def t_callsite():
    """Test the calibration call exactly as the arm makes it."""
    planted = [{"sd_q": v} for v in (0.050, 0.052, 0.054, 0.056)]
    pm = statistics.mean(r["sd_q"] for r in planted)
    c = level_calibration(COMMITTED_RATIO, planted)
    assert c["sigma_eps"] == NOBS.SIGMA_EPS and c["w_q"] == BLIND.W_Q_LOW, \
        "committed branch did not return the committed constants EXACTLY"
    assert c["sd_q_used"] == NOBS.SD_Q_REF and \
        abs(c["sd_q_measured"] - pm) < 1e-15, \
        "committed branch must IGNORE the measured sd(q) but report it"
    for nm, r in LEVELS:
        if nm == "COMMITTED":
            continue
        c2 = level_calibration(r, planted)
        assert abs(c2["sd_q_used"] - pm) < 1e-15, \
            "%s did not use its own measured sd(q)" % nm
        e_s, e_w = calibrate(GAMMA, c2["sd_q_used"])
        assert c2["sigma_eps"] == e_s and c2["w_q"] == e_w
        assert abs(c2["w_q"] / c2["sd_q_used"] - 0.34) < 1e-12
        assert abs(c2["sigma_eps"] / c2["sd_skew_model"] - 1.0) < 1e-12
        assert c2["sigma_eps"] != NOBS.SIGMA_EPS and \
            c2["w_q"] != BLIND.W_Q_LOW, \
            "%s silently fell back to the committed constants" % nm
    for arm in ("JITTER", "QUANT"):
        mm = build_checked(arm, 0, c)
        assert (mm.sigma_eps if arm == "JITTER" else mm.w_q) == \
            (c["sigma_eps"] if arm == "JITTER" else c["w_q"])
    bad = dict(c)
    bad["sigma_eps"] = c["sigma_eps"] * 1.0000001
    orig = GAM.make_defended
    try:
        GAM.make_defended = lambda *a, **k: orig(a[0], a[1], a[2],
                                                 c["sigma_eps"], a[4])
        globals()["make_defended"] = GAM.make_defended
        try:
            build_checked("JITTER", 0, bad)
            raise AssertionError("the run-time check missed a mismatch")
        except RuntimeError:
            pass
    finally:
        GAM.make_defended = orig
        globals()["make_defended"] = orig
    assert make_defended is GAM.make_defended
    print("  T-callsite    PASS  level_calibration; the function the arm "
          "calls; returns the committed")
    print("                      constants EXACTLY at 0.10 while ignoring a "
          "planted measured sd(q), and uses")
    print("                      the measured sd(q) at 0.02 and 0.30 with "
          "w/sd = 0.34 and sigma_eps/sd_skew = 1.")
    print("                      The run-time check on the built maker fires "
          "on a 1e-7 sigma_eps mismatch.")


def t_parse():
    def close(got, want):
        """Compare (value, tolerance) pairs with a float tolerance."""
        return abs(got[0] - want[0]) < 1e-12 and abs(got[1] - want[1]) < 1e-18
    rows, summ, calib = parse_reference_high()
    assert close(rows[0]["base_climb"], (-0.009617, 5e-7))
    assert close(rows[0]["jitter_climb"], (2.0080, 5e-5))
    assert close(rows[23]["qb_climb"], (0.019226, 5e-7))
    assert close(summ[("QUANT", "base")]["mean"], (0.0058, 5e-7))
    assert summ[("JITTER", "zero")]["dom_seed"] == 3
    assert abs(calib["sigma_eps"][0] - 0.315532) < 1e-12
    qx = parse_qx_committed()
    assert close(qx["median_bp"]["mean"], (0.32138, 5e-6))
    assert close(qx["agg_per_s"]["se"], (0.00066, 5e-6))
    base_mean = statistics.mean(rows[s]["base_climb"][0] for s in range(24))
    print("  T-parse       PASS  96 per-seed values, 4 summaries, the "
          "calibration row and the 0.020 drift")
    print("                      row parsed from the committed files with "
          "tolerance = half the last printed digit.")
    print("                      Committed BASELINE climb, mean of its 24 "
          "printed values: %+.6f." % base_mean)


def t_rules():
    def c(m, se):
        return {"mean": m, "se": se}
    # R-JITTER, one-sided
    assert call(c(1.0, 0.1), "R-JITTER")[0]
    assert not call(c(0.1, 0.1), "R-JITTER")[0]
    assert not call(c(-1.0, 0.1), "R-JITTER")[0]
    # R-QUANT, two-sided: large positive QUANT-minus-BASELINE is RECOVERY and must fail
    ok, _t, _b, dirn = call(c(1.0, 0.1), "R-QUANT")
    assert not ok and dirn.startswith("RECOVERY"), \
        "a 10-SE positive QUANT-minus-BASELINE must fail as RECOVERY"
    ok, _t, _b, dirn = call(c(-1.0, 0.1), "R-QUANT")
    assert not ok and dirn.startswith("degrades"), \
        "a 10-SE negative must fail as 'not flat', NOT as a defense failure"
    assert call(c(0.1, 0.1), "R-QUANT")[0] and call(c(-0.1, 0.1),
                                                    "R-QUANT")[0]
    assert call(c(0.0058, 0.004795), "R-QUANT")[0], \
        "the committed +0.005800 +/- 0.004795 (1.21 SE) is flat"
    # flag for decisions between 2.0 and 2.069
    assert call(c(0.205, 0.1), "R-JITTER")[2], "2.05 SE is between"
    assert not call(c(0.21, 0.1), "R-JITTER")[2]
    assert not call(c(-0.205, 0.1), "R-JITTER")[2], \
        "a negative JITTER mean fails at both thresholds -- not 'between'"
    assert call(c(0.205, 0.1), "R-QUANT")[2], "R-QUANT between, positive"
    assert call(c(-0.205, 0.1), "R-QUANT")[2], "R-QUANT between, negative"
    # R-CONTRAST, one-sided, reported
    assert call(c(1.0, 0.1), "R-CONTRAST")[0]
    assert not call(c(-1.0, 0.1), "R-CONTRAST")[0]
    # zero SE raises for every rule (one try block per rule)
    for rule in ("R-JITTER", "R-QUANT", "R-CONTRAST"):
        try:
            call(c(1.0, 0.0), rule)
        except ValueError:
            continue
        raise AssertionError("a zero SE produced a %s verdict" % rule)
    d = [0.0] * 23 + [10.0]
    assert REF.dominance(d)[1] > DOM_BAR
    print("  T-rules       PASS  R-JITTER one-sided; R-QUANT two-sided; a "
          "10-SE positive value now fails as")
    print("                      RECOVERY (it passed under the rule first "
          "specified), a 10-SE negative fails as")
    print("                      'not flat', and the committed +0.0058 at 1.21 "
          "SE is flat. The between flag fires")
    print("                      for R-QUANT on both signs, for R-JITTER only "
          "when positive. Zero SE raises; R2 fires.")


def t_patch():
    """Patch inert at 0.10, live at 0.02, restored after an exception."""
    t0 = time.time()
    s = 0
    plain = NOBS.run(s, "BASELINE", gamma=GAMMA,
                     mm_override=make_defended("BASELINE", s, GAMMA, 0.0, 1.0))
    with informed_at(COMMITTED_RATIO):
        same = NOBS.run(s, "BASELINE", gamma=GAMMA,
                        mm_override=make_defended("BASELINE", s, GAMMA, 0.0,
                                                  1.0))
    assert bindings_restored()
    for k in plain:
        if k == "ma_curve":
            assert plain[k] == same[k], "ma_curve differs at 0.10"
        else:
            assert plain[k] == same[k], "%s differs at 0.10: %r vs %r" % (
                k, plain[k], same[k])
    with informed_at(TOXICITY_PRESETS["low"]):
        low = NOBS.run(s, "BASELINE", gamma=GAMMA,
                       mm_override=make_defended("BASELINE", s, GAMMA, 0.0,
                                                 1.0))
    assert bindings_restored()
    assert low["n_prints"] != plain["n_prints"], \
        "the patch did not reach phase7_nobs.run's world"
    try:
        with informed_at(0.30):
            assert not bindings_restored()
            raise KeyError("planted")
    except KeyError:
        pass
    assert bindings_restored(), "binding leaked out of an exception"
    print("  T-patch       PASS  at 0.10 the patched run is bit-identical to the "
          "unpatched one on every key;")
    print("                      at 0.02 it differs (%d vs %d public prints), "
          "so the patch is live at the" % (low["n_prints"], plain["n_prints"]))
    print("                      call site; every binding is restored, "
          "including after an exception. (%.0fs)" % (time.time() - t0))


def _planted_levels():
    """Planted results for all three levels covering nearly every branch."""
    def lin(s, amp):
        return amp * (s - 11.5) / 11.5
    spec = {"COMMITTED": ("outlier", "flat"), "LOW": ("negative", "down"),
            "HIGH": ("normal", "up")}
    levels = {}
    for name, ratio in LEVELS:
        jmode, qmode = spec[name]
        base, jit, qua = [], [], []
        for s in SEEDS_24:
            b = -0.015 + lin(s, 0.005)
            jd = {"normal": 5.7 + lin(s, 0.5),
                  "outlier": 60.0 if s == 7 else 5.0 + lin(s, 0.01),
                  "negative": -0.1 + lin(s, 0.001)}[jmode]
            qd = {"flat": 0.002 if s % 2 else -0.002,
                  "up": 0.05 + lin(s, 0.001),
                  "down": -0.05 + lin(s, 0.001)}[qmode]
            base.append({"ma_curve": [(NOBS.M_CONTROL, 1.0),
                                      (NOBS.M_TOP, 1.0 + b)],
                         "sd_q": 0.045 + 0.0002 * s, "n_mm_fills": 400.0})
            qua.append({"ma_curve": [(NOBS.M_CONTROL, 0.9),
                                     (NOBS.M_TOP, 0.9 + b + qd)],
                        "sd_q": 0.045, "n_mm_fills": 400.0})
            jit.append({"ma_curve": [(NOBS.M_CONTROL, -5.0),
                                     (NOBS.M_TOP, -5.0 + b + jd)],
                        "sd_q": 0.045, "n_mm_fills": 400.0})
        levels[name] = {
            "name": name, "ratio": ratio, "secs": 0.0,
            "res": {"BASELINE": base, "JITTER": jit, "QUANT": qua},
            "cal": level_calibration(ratio, base),
            "drift_qx": [{"median_bp": 0.32 + 0.001 * s,
                          "levels_per_side": 550.0 + s,
                          "near_ns": 0.04 + 0.001 * s,
                          "agg_per_s": 0.040 + 0.0001 * s,
                          "shape": 2.1 + 0.01 * s} for s in DRIFT_SEEDS],
            "drift_cg": [{"mm_spread": 11.3 + 0.01 * s,
                          "resid_spread": 1.5 + 0.01 * s,
                          "skew_sd": 0.30 + 0.001 * s} for s in DRIFT_SEEDS]}
    return levels


def t_postsim():
    """Run report_anchor() and report() on planted data; output captured."""
    lv = _planted_levels()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fails, n, dfails = report_anchor(lv["COMMITTED"])
        summ = report(lv)
    text = buf.getvalue()
    assert fails and n > 96 and dfails, \
        "planted data must FAIL both anchors, and the anchor code must run"
    assert summ["axis3_held"] is False
    assert summ["recovery"] == ["HIGH"], summ["recovery"]
    assert summ["not_flat"] == ["LOW"], summ["not_flat"]
    assert any(x.startswith("R-JITTER at LOW") for x in summ["fails_named"])
    assert any(x.startswith("R-QUANT at HIGH -- RECOVERY")
               for x in summ["fails_named"])
    assert any(x.startswith("R-QUANT at LOW -- degrades")
               for x in summ["fails_named"])
    assert len(summ["contrast_fail"]) == 1 and \
        summ["contrast_fail"][0].startswith("LOW")
    assert len(summ["r2_bad"]) == 1 and \
        summ["r2_bad"][0].startswith("JITTER COMMITTED seed 7"), summ["r2_bad"]
    for marker in ("ANCHOR FAILED", "DRIFT ANCHOR FAILED", "CALIBRATION",
                   "AXIS 3 NOT HELD", "RECOVERY at HIGH",
                   "QUANT NOT FLAT at LOW", "R-CONTRAST (reported): fails at",
                   "R2 FAILED in", "MARKET DRIFT", "gate 2"):
        assert marker in text, "post-simulation output lacks %r" % marker
    assert set(summ["drift"]) == set(REPORT_ORDER)
    print("  T-postsim     PASS  report_anchor() and report(); the code the "
          "run uses after the simulation --")
    print("                      ran on planted results for all three levels "
          "(%d lines, captured, not logged)." % len(text.splitlines()))
    print("                      It returned RECOVERY at HIGH, NOT FLAT at LOW, "
          "R-JITTER failing at LOW, R-CONTRAST")
    print("                      failing at LOW and R2 failing on the planted "
          "outlier; both anchors reported failure.")


def selftest():
    print("=" * 128)
    print("Self-tests: phase8_toxicity.py. The arm has not run.")
    print("=" * 128)
    t_ident()
    t_parse()
    t_rules()
    t_callsite()
    t_postsim()
    t_patch()
    print("")
    print("  All 6 self-tests PASSED.")
    return 0


def check_anchor(lv, sc):
    rows, summ, calib = parse_reference_high()
    base = lv["res"]["BASELINE"]
    got = {"base_climb": REF.climb_series(base),
           "quant_climb": REF.climb_series(lv["res"]["QUANT"]),
           "jitter_climb": REF.climb_series(lv["res"]["JITTER"]),
           "qb_climb": REF.climb_series(REF.diff_curve(lv["res"]["QUANT"],
                                                       base))}
    fails, n = [], 0
    for key in got:
        for s in range(24):
            want, tol = rows[s][key]
            n += 1
            if abs(got[key][s] - want) > tol:
                fails.append("%s seed %d: %.9f vs %.6f" % (key, s,
                                                           got[key][s], want))
    lines = []
    for (arm, ref), w in sorted(summ.items()):
        cc = sc[arm][ref]
        for fld, val in (("mean", cc["mean"]), ("se", cc["se"])):
            want, tol = w[fld]
            n += 1
            ok = abs(val - want) <= tol
            lines.append((arm, ref, fld, val, want, tol, ok))
            if not ok:
                fails.append("%s %s %s" % (arm, ref, fld))
        wp, tp = w["dom_share_pct"]
        n += 2
        okp = abs(100.0 * cc["dom_share"] - wp) <= tp
        oks = cc["dom_seed"] == w["dom_seed"]
        lines.append((arm, ref, "worst-seed share %", 100.0 * cc["dom_share"],
                      wp, tp, okp))
        if not (okp and oks):
            fails.append("%s %s worst seed %d/%d" % (arm, ref, cc["dom_seed"],
                                                     w["dom_seed"]))
    cal = lv["cal"]
    for fld, val in (("sigma_eps", cal["sigma_eps"]), ("w_q", cal["w_q"]),
                     ("sd_q_24", cal["sd_q_measured"]),
                     ("sd_q_24_se", cal["sd_q_measured_se"])):
        want, tol = calib[fld]
        n += 1
        ok = abs(val - want) <= tol
        lines.append(("calib", "", fld, val, want, tol, ok))
        if not ok:
            fails.append("calib " + fld)
    return fails, n, lines


def check_drift_anchor(lv):
    qx = parse_qx_committed()
    fails, lines = [], []
    for name, key, _t, _tol in TARGETS:
        vals = [r[key] for r in lv["drift_qx"]]
        m, se = desc_mean_se(vals, key, len(DRIFT_SEEDS))
        for fld, val in (("mean", m), ("se", se)):
            want, tol = qx[key][fld]
            ok = abs(val - want) <= tol
            lines.append((name, fld, val, want, tol, ok))
            if not ok:
                fails.append("%s %s" % (name, fld))
    return fails, lines


def drift_row(lv):
    out = {}
    for key in ("median_bp", "levels_per_side", "near_ns", "agg_per_s",
                "shape"):
        out[key] = desc_mean_se([r[key] for r in lv["drift_qx"]], key,
                                len(DRIFT_SEEDS))
    mm = desc_mean_se([r["mm_spread"] for r in lv["drift_cg"]], "mm_spread",
                      len(DRIFT_SEEDS))
    rs = desc_mean_se([r["resid_spread"] for r in lv["drift_cg"]],
                      "resid_spread", len(DRIFT_SEEDS))
    # gate 2: guarded division (zero = failed measurement)
    out["gate2"] = (_ratio(mm[0], rs[0], "gate 2 %s" % lv["name"]), mm[0],
                    rs[0])
    out["skew_sd_measured"] = desc_mean_se(
        [r["skew_sd"] for r in lv["drift_cg"]], "skew_sd", len(DRIFT_SEEDS))
    return out


def report_anchor(lv):
    """Anchor and drift-anchor checks for COMMITTED; returns both fail lists."""
    sc = score_level(lv)
    fails, n, lines = check_anchor(lv, sc)
    print("")
    print("=" * 128)
    print("Anchor: the committed level must reproduce "
          "phase8_reference_results.txt's HIGH row, 24 seeds")
    print("=" * 128)
    print("  96 per-seed climbs (BASELINE, QUANT, JITTER, Q-minus-B x 24 "
          "seeds) checked individually.")
    for arm, ref, fld, val, want, tol, ok in lines:
        print("  %-7s %-5s %-19s got %+.9f  committed %+.6f  tol %.0e  %s"
              % (arm, ref, fld, val, want, tol,
                 "OK" if ok else "*** mismatch ***"))
    if fails:
        print("")
        for x in fails[:20]:
            print("  FAIL: %s" % x)
        print("  ANCHOR FAILED on %d of %d checks." % (len(fails), n))
    else:
        print("  Anchor HELD on all %d checks." % n)
    dfails, dlines = check_drift_anchor(lv)
    for name2, fld, val, want, tol, ok in dlines:
        print("  drift  %-18s %-4s got %.6f  committed %.5f  tol %.0e  %s"
              % (name2, fld, val, want, tol,
                 "OK" if ok else "*** mismatch ***"))
    if dfails:
        print("  DRIFT ANCHOR FAILED: %s" % ", ".join(dfails))
    else:
        print("  Drift anchor HELD: quotesize_extend's 0.020 row reproduces "
              "through the patch.")
    print("")
    return fails, n, dfails


def main():
    print("=" * 128)
    print("Phase 8 axis 3: does the defense result survive toxicity?")
    print("=" * 128)
    print("gamma=%.2e, T=%.0fs, k=%.5f, quote_size=%.3f, 24 seeds, arms "
          "BASELINE / JITTER / QUANT, M %d -> %d."
          % (GAMMA, NOBS.T, NOBS.K, QSIZE, NOBS.M_CONTROL, NOBS.M_TOP))
    print("Toxicity = informed:noise arrival ratio. Levels: %s."
          % ", ".join("%s %.2f" % (n, r) for n, r in LEVELS))
    print("Pre-registered: R-JITTER (one-sided), R-QUANT (two-sided, corrected "
          "before the run), AXIS 3 HELD, R2;")
    print("R-CONTRAST reported. Z=2.0 with t(23)=2.069 flagged. Estimated "
          "~42 min.")
    print("")

    levels = {}
    for name, ratio in LEVELS:
        print("running toxicity %-9s ratio=%.2f ..." % (name, ratio),
              flush=True)
        lv = run_level(name, ratio)
        levels[name] = lv
        print("  done in %.0fs" % lv["secs"], flush=True)
        if name == "COMMITTED":
            fails, n, dfails = report_anchor(lv)
            if fails:
                raise AssertionError(
                    "ANCHOR FAILED on %d of %d checks -- the committed level "
                    "is not the committed configuration. Nothing else is "
                    "comparable. HALTING." % (len(fails), n))
            if dfails:
                raise AssertionError("DRIFT ANCHOR FAILED: %s. HALTING."
                                     % ", ".join(dfails))

    return report(levels)


def report(levels):
    """Scoring, calibration table, rules, R-CONTRAST, verdict and drift table."""
    scores = {n: score_level(levels[n]) for n in REPORT_ORDER}

    print("=" * 128)
    print("CALIBRATION: ratios held, absolute strengths re-derived per level")
    print("=" * 128)
    print("  %-10s %-6s %-22s %-11s %-12s %-22s %-12s %-12s %s" % (
        "level", "ratio", "sd(q) measured, 24 sd", "sd(q) used",
        "sd(skew) as", "sd(skew) measured, 8 sd", "sigma_eps", "w_q",
        "source"))
    for n in REPORT_ORDER:
        lv = levels[n]
        c = lv["cal"]
        sk = drift_row(lv)["skew_sd_measured"]
        print("  %-10s %-6.2f %.6f +/- %.6f  %-11.6f %-12.6f %.6f +/- %.6f"
              "  %-12.6f %-12.9f %s" % (
                  n, lv["ratio"], c["sd_q_measured"], c["sd_q_measured_se"],
                  c["sd_q_used"], c["sd_skew_model"], sk[0], sk[1],
                  c["sigma_eps"], c["w_q"], c["source"]))
    print("  w_q / sd(q)_used = 0.34 and sigma_eps / sd(skew)_AS = 1.0 at "
          "every level (asserted in T-callsite).")
    print("")

    print("=" * 128)
    print("Rules (BASELINE-referenced, 24 seeds; zero reference beside it)")
    print("=" * 128)
    verdict_ok, fails_named, between, r2_bad = True, [], [], []
    recovery, not_flat = [], []
    for arm, rule in (("JITTER", "R-JITTER"), ("QUANT", "R-QUANT")):
        print("  %s%s" % (rule, "  (two-sided: |QUANT-minus-BASELINE| < 2.0 "
                                "SE)" if rule == "R-QUANT" else
                          "  (one-sided: JITTER-minus-BASELINE > 0 at 2.0 "
                          "SE)"))
        for n in REPORT_ORDER:
            s = scores[n]
            cb, cz = s[arm]["base"], s[arm]["zero"]
            ok, t, btw, dirn = call(cb, rule)
            if not ok:
                verdict_ok = False
                if rule == "R-QUANT":
                    fails_named.append("R-QUANT at %s -- %s" % (n, dirn))
                    (recovery if cb["mean"] > 0 else not_flat).append(n)
                else:
                    fails_named.append("R-JITTER at %s -- %s" % (n, dirn))
            if btw:
                between.append("%s at %s (%.3f SE)" % (rule, n, t))
            r2 = cb["dom_share"] > DOM_BAR
            if r2:
                r2_bad.append("%s %s seed %d %.1f%%" % (
                    arm, n, cb["dom_seed"], 100.0 * cb["dom_share"]))
            print("    %-10s BASELINE ref %+.6f +/- %.6f  %5.2f SE  %-5s | "
                  "zero ref %+.6f +/- %.6f | worst seed %2d %5.1f%%%s" % (
                      n, cb["mean"], cb["se"], t, "ok" if ok else "FAIL",
                      cz["mean"], cz["se"], cb["dom_seed"],
                      100.0 * cb["dom_share"], "  R2 FAIL" if r2 else ""))
    print("")
    print("  R-CONTRAST  (reported, not part of the verdict: JITTER-minus-QUANT "
          "climb > 0 at 2.0 SE)")
    contrast_fail = []
    for n in REPORT_ORDER:
        cc = scores[n]["CONTRAST"]
        ok, t, btw, dirn = call(cc, "R-CONTRAST")
        if not ok:
            contrast_fail.append("%s (%s)" % (n, dirn))
        if btw:
            between.append("R-CONTRAST at %s (%.3f SE)" % (n, t))
        print("    %-10s %+.6f +/- %.6f  %5.2f SE  %-5s | worst seed %2d "
              "%5.1f%%" % (n, cc["mean"], cc["se"], t, "ok" if ok else "FAIL",
                           cc["dom_seed"], 100.0 * cc["dom_share"]))
    print("")
    for n in REPORT_ORDER:
        bm, bse = mean_se(scores[n]["base_climb"])
        print("  BASELINE climb at %-10s %+.6f +/- %.6f" % (n, bm, bse))
    print("")
    print("  R2 (no seed > 40%% of the squared deviation in a scored cell): %s"
          % ("HELD" if not r2_bad else "FAILED; " + "; ".join(r2_bad)))
    print("  Calls decided between 2.0 SE and t(23)=2.069: %s"
          % ("none" if not between else "; ".join(between)))
    print("")
    print("=" * 128)
    print("The verdict")
    print("=" * 128)
    if verdict_ok:
        print("  AXIS 3 HELD. R-JITTER and the two-sided R-QUANT hold at LOW, "
              "COMMITTED and HIGH toxicity.")
    else:
        print("  AXIS 3 NOT HELD. Failed: %s." % "; ".join(fails_named))
        if recovery:
            print("  >>> RECOVERY at %s: averaging recovers the quantized "
                  "signal beyond the smoother; the defense fails there."
                  % ", ".join(recovery))
        if not_flat:
            print("  QUANT NOT FLAT at %s, negative: degrades beyond the "
                  "smoother. Reported as" % ", ".join(
                      not_flat))
            print("  'not flat'; not a defense failure, but fails the two-sided rule.")
    print("  R-CONTRAST (reported): %s" % (
        "holds at every level" if not contrast_fail
        else "fails at " + "; ".join(contrast_fail)))
    if r2_bad:
        print("  R2 FAILED in: %s; those cells are one-seed-dominated." % "; ".join(r2_bad))
    print("")

    print("=" * 128)
    print("MARKET DRIFT: book movement at each toxicity level vs the "
          "calibrated one (8 seeds, BASELINE)")
    print("=" * 128)
    ref_vals = {"median_bp": [t for n2, k, t, _ in TARGETS
                              if k == "median_bp"][0],
                "levels_per_side": [t for n2, k, t, _ in TARGETS
                                    if k == "levels_per_side"][0],
                "agg_per_s": [t for n2, k, t, _ in TARGETS
                              if k == "agg_per_s"][0],
                "shape": [t for n2, k, t, _ in TARGETS if k == "shape"][0],
                "near_ns": REAL_NEAR_BTC}
    labels = (("median_bp", "width bp"), ("levels_per_side", "levels/side"),
              ("near_ns", "near-touch BTC/side"), ("agg_per_s", "aggTrades/s"),
              ("shape", "mean/median shape"))
    rows = {n: drift_row(levels[n]) for n in REPORT_ORDER}
    print("  %-20s %-12s" % ("target", "real data") + "".join(
        "%-26s" % n for n in REPORT_ORDER))
    for key, lab in labels:
        print("  %-20s %-12.5f" % (lab, ref_vals[key]) + "".join(
            "%11.5f +/- %-10.5f " % rows[n][key] for n in REPORT_ORDER))
    print("  %-20s %-12s" % ("gate 2 (<= 10x)", "") + "".join(
        "%8.3fx (%.3f/%.3f)      " % rows[n]["gate2"] for n in REPORT_ORDER))
    print("")
    print("  Width, levels, agg rate, shape: clipped_window's four "
          "scored targets; near-touch")
    print("  depth: a working-point solve target, not scored. Working point")
    print("  not re-solved at 0.02 or 0.30. Gate 2 via "
          "clipped_gamma_sweep.run, unchanged.")
    return {"axis3_held": verdict_ok, "fails_named": fails_named,
            "recovery": recovery, "not_flat": not_flat,
            "contrast_fail": contrast_fail, "r2_bad": r2_bad,
            "between": between, "drift": rows}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
