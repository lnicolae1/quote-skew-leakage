# phase8_gamma.py: Phase 8 axis 2: does Q2 (QUANT flat in M, JITTER climbs) hold across risk aversion?
# - gammas: LOW 1.0e-6 (DEFAULT_GAMMA), MEDIUM 3.0e-6, HIGH 9.4e-6 (incumbent, gate-2 cap); same 8 seeds x 86400 s
# - defenses recalibrated per gamma from that gamma's own sd(q): sigma_eps = one sd of skew, W_Q at fixed w/sd
#   (a frozen SIGMA_EPS would scale the noise-to-signal ratio with gamma and inflate JITTER's climb at low gamma)
# - pre-registered Q2-gamma: QUANT's M=1 -> M=300 climb flat within 2.0 SE at all three gammas,
#   and JITTER's climb positive at 2.0 SE at all three
# - anchor: HIGH must reproduce the committed arm B climbs

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import phase7_blind as BLIND
import phase7_nobs as NOBS

from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from phase7_blind import DegenerateComparison, QuantizedMM, mean_se, se_mult
from phase7_nobs import JitteredMM, paired_between, score_q2

# is-identity with phase7_nobs
assert JitteredMM is NOBS.JitteredMM, \
    "JitteredMM must be phase7_nobs's own class"
assert QuantizedMM is NOBS.QuantizedMM, (
    "QuantizedMM must be the SAME class Arm A and Arm B both used, or the "
    "defense under test is not the defense that was measured")
assert score_q2 is NOBS.score_q2, \
    "Q2 must be scored by phase7_nobs's own rule, not a re-implementation"
assert paired_between is NOBS.paired_between, \
    "the paired climb must be computed by phase7_nobs's own function"
assert issubclass(JitteredMM, MarketMaker), "JitteredMM must subclass"
assert issubclass(QuantizedMM, MarketMaker), "QuantizedMM must subclass"

# constants
SEEDS = list(NOBS.SEEDS)
T = NOBS.T
K = NOBS.K
QUOTE_SIZE = NOBS.QUOTE_SIZE
Z_SE = NOBS.Z_SE

GAMMA_LOW = 1.0e-6
GAMMA_MED = 3.0e-6
GAMMA_HIGH = 9.4e-6

GAMMAS = [("LOW", GAMMA_LOW), ("MEDIUM", GAMMA_MED), ("HIGH", GAMMA_HIGH)]

# committed arm B climbs (printed beside HIGH; no rule uses them)
ANCHOR_QUANT = (-0.006231, 0.007353)
ANCHOR_JITTER = (11.382120, 3.529188)

# sd(q) at LOW and MEDIUM from the 3-seed sweep (for the uncalibrated-ratio figure only)
SWEEP_SD_Q = {GAMMA_LOW: 0.04407, GAMMA_MED: 0.04368}

M_CONTROL = NOBS.M_CONTROL
M_TOP = NOBS.M_TOP

assert GAMMA_HIGH == NOBS.GAMMA_CAP, (
    "the HIGH arm must be the incumbent gamma EXACTLY, or the no-op "
    "self-test anchors nothing")
assert GAMMA_LOW == DEFAULT_GAMMA, \
    "the LOW arm is DEFAULT_GAMMA by construction, not by coincidence"
assert [g for _n, g in GAMMAS] == sorted(g for _n, g in GAMMAS), \
    "GAMMAS must be ascending so the report reads low to high"


# calibration layer
def calibrate(gamma, sd_q):
    """Defense strengths at one gamma from its own sd(q). Returns (sigma_eps, w_q)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive, got %r" % (gamma,))
    if sd_q <= 0:
        raise ValueError("sd_q must be positive, got %r" % (sd_q,))
    sd_skew = sd_q * gamma * NOBS.SIG_ABS * NOBS.SIG_ABS * NOBS.MEAN_TAU
    sigma_eps = NOBS.SIGMA_EPS_MULT * sd_skew
    w_q = BLIND.W_OVER_SD_LOW * sd_q
    return sigma_eps, w_q


def uncalibrated_ratio(gamma, sd_q):
    """Noise-to-signal ratio an uncalibrated run would carry at gamma (design value 1.0)."""
    sigma_eps_here, _w = calibrate(gamma, sd_q)
    return NOBS.SIGMA_EPS / sigma_eps_here


def make_defended(arm, seed, gamma, sigma_eps, w_q):
    """Calibrated defended maker for one arm at one gamma, passed to phase7_nobs.run via mm_override."""
    if arm == "BASELINE":
        mm = MarketMaker(horizon=T, k=K, gamma=gamma, quote_size=QUOTE_SIZE)
    elif arm == "JITTER":
        mm = JitteredMM(sigma_eps, NOBS.JITTER_SEED_BASE + seed, horizon=T,
                        k=K, gamma=gamma, quote_size=QUOTE_SIZE)
    elif arm == "QUANT":
        mm = QuantizedMM(w_q, horizon=T, k=K, gamma=gamma,
                         quote_size=QUOTE_SIZE)
    else:
        raise ValueError("unknown arm %r" % (arm,))
    if mm.gamma != gamma:
        raise DegenerateComparison(
            "built a %s maker at gamma %r but it carries %r -- the override "
            "path would silently ignore the gamma argument"
            % (arm, gamma, mm.gamma))
    return mm


# the arm
def run_one_gamma(gamma, seeds=None):
    """BASELINE (calibrates), then the two defended arms at that gamma."""
    seeds = SEEDS if seeds is None else seeds
    res = {}
    res["BASELINE"] = [
        NOBS.run(s, "BASELINE", gamma=gamma,
                 mm_override=make_defended("BASELINE", s, gamma, 0.0, 1.0))
        for s in seeds]

    sd_q_mean, sd_q_se = mean_se([r["sd_q"] for r in res["BASELINE"]])
    sigma_eps, w_q = calibrate(gamma, sd_q_mean)

    for arm in ("JITTER", "QUANT"):
        res[arm] = [
            NOBS.run(s, arm, gamma=gamma,
                     mm_override=make_defended(arm, s, gamma, sigma_eps,
                                               w_q))
            for s in seeds]
    return {"gamma": gamma, "res": res, "sd_q": sd_q_mean,
            "sd_q_se": sd_q_se, "sigma_eps": sigma_eps, "w_q": w_q}


def score_q2_gamma(cells):
    """Q2-gamma: QUANT flat and JITTER climbing at Z_SE at every gamma. Returns (verdict, rows)."""
    rows, quant_ok, jitter_ok = [], True, True
    for name, gamma in GAMMAS:
        cell = cells[gamma]
        qv, qm, qse, qmult = score_q2(cell["res"]["QUANT"])
        jm, jse, jmult = paired_between(cell["res"]["JITTER"],
                                        M_CONTROL, M_TOP)
        q_flat = (qmult < Z_SE)
        j_rise = (jm > 0.0) and (jmult >= Z_SE)
        quant_ok = quant_ok and q_flat
        jitter_ok = jitter_ok and j_rise
        rows.append({"name": name, "gamma": gamma, "q_verdict": qv,
                     "q": (qm, qse, qmult), "q_flat": q_flat,
                     "j": (jm, jse, jmult), "j_rise": j_rise})
    if quant_ok and jitter_ok:
        v = "HELD"
    elif not quant_ok and not jitter_ok:
        v = "REFUTED (QUANT not flat, and JITTER does not rise)"
    elif not quant_ok:
        v = "REFUTED (QUANT is NOT flat at every gamma)"
    else:
        v = "REFUTED (JITTER does not rise at every gamma)"
    return v, rows


# self-tests
def t_is_identity():
    """Scoring objects are phase7_nobs's own."""
    assert NOBS.run.__module__ == "phase7_nobs", "run() must be Arm B's own"
    assert mean_se is BLIND.mean_se, "mean_se must be phase7_blind's own"
    assert se_mult is BLIND.se_mult, "se_mult must be phase7_blind's own"
    assert NOBS.MA_WINDOWS == [1, 3, 10, 30, 100, 300], \
        "the M sweep moved; Q2-GAMMA's endpoints are defined on it"
    assert M_CONTROL == 1 and M_TOP == 300, \
        "Q2-GAMMA is scored M=1 -> M=300, as Q2 was"
    print("  t_is_identity                  PASS")


def t_no_default_moved():
    """Import does not perturb committed defaults."""
    assert DEFAULT_GAMMA == 1e-6, "DEFAULT_GAMMA moved"
    assert DEFAULT_QUOTE_SIZE == 0.02, "DEFAULT_QUOTE_SIZE moved"
    assert NOBS.GAMMA_CAP == 9.4e-6, "phase7_nobs.GAMMA_CAP moved"
    assert NOBS.SD_Q_REF == 0.04604, "phase7_nobs.SD_Q_REF moved"
    assert NOBS.SIGMA_EPS_MULT == 1.0, (
        "SIGMA_EPS_MULT is the statement 'one sd of skew' and must stay 1.0 "
        "for the calibration to mean what this file says it means")
    assert BLIND.W_OVER_SD_LOW == 0.34, "phase7_blind.W_OVER_SD_LOW moved"
    assert list(NOBS.SEEDS) == list(range(8)), \
        "phase7_nobs.SEEDS was mutated by this import"
    assert NOBS.QUOTE_SIZE == 0.020, "phase7_nobs.QUOTE_SIZE moved"
    print("  t_no_default_moved             PASS")


def t_calibration_is_noop_at_cap():
    """At the incumbent gamma the calibration returns the committed constants exactly."""
    sigma_eps, w_q = calibrate(NOBS.GAMMA_CAP, NOBS.SD_Q_REF)
    assert sigma_eps == NOBS.SIGMA_EPS, (
        "CALIBRATION IS NOT A NO-OP AT THE CAP: re-derived sigma_eps "
        "%.17g against the committed %.17g. The three-gamma comparison "
        "would be unanchored." % (sigma_eps, NOBS.SIGMA_EPS))
    assert w_q == BLIND.W_Q_LOW, (
        "CALIBRATION IS NOT A NO-OP AT THE CAP: re-derived w_q %.17g "
        "against the committed %.17g" % (w_q, BLIND.W_Q_LOW))
    assert uncalibrated_ratio(NOBS.GAMMA_CAP, NOBS.SD_Q_REF) == 1.0, (
        "the uncalibrated ratio at the incumbent point must be exactly 1.0 "
        "by construction, but it is %.17g"
        % uncalibrated_ratio(NOBS.GAMMA_CAP, NOBS.SD_Q_REF))
    print("  t_calibration_is_noop_at_cap   PASS  (sigma_eps = $%.9f, "
          "w_q = %.9f BTC, both equal to committed)" % (sigma_eps, w_q))


def t_calibration_holds_the_ratio():
    """w/sd fixed and sigma_eps = one sd of skew at every gamma."""
    for sd_q in (0.030, 0.04604, 0.055):
        for gamma in (GAMMA_LOW, GAMMA_MED, GAMMA_HIGH):
            sigma_eps, w_q = calibrate(gamma, sd_q)
            assert abs(w_q / sd_q - BLIND.W_OVER_SD_LOW) < 1e-15, (
                "w/sd is %.17g at sd_q %r, not W_OVER_SD_LOW"
                % (w_q / sd_q, sd_q))
            want = sd_q * gamma * NOBS.SIG_ABS * NOBS.SIG_ABS * NOBS.MEAN_TAU
            assert sigma_eps == want, (
                "sigma_eps %.17g is not one sd of skew %.17g at gamma %r"
                % (sigma_eps, want, gamma))
    print("  t_calibration_holds_the_ratio  PASS")


def t_calibration_scales_linearly():
    """sigma_eps linear in gamma at fixed sd(q); w_q independent of gamma."""
    s1, w1 = calibrate(1.0e-6, 0.04604)
    s2, w2 = calibrate(2.0e-6, 0.04604)
    assert abs(s2 / s1 - 2.0) < 1e-12, \
        "sigma_eps is not linear in gamma: ratio %.17g" % (s2 / s1,)
    assert w1 == w2, \
        "w_q must not depend on gamma, but %.17g != %.17g" % (w1, w2)
    for bad in (0.0, -1e-6):
        try:
            calibrate(bad, 0.04604)
        except ValueError:
            pass
        else:
            raise AssertionError("calibrate accepted gamma %r" % (bad,))
        try:
            calibrate(1e-6, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("calibrate accepted sd_q %r" % (bad,))
    print("  t_calibration_scales_linearly  PASS")


def t_uncalibrated_ratio_is_the_trap():
    """Derive the ~9.8 uncalibrated ratio from committed values."""
    lo = uncalibrated_ratio(GAMMA_LOW, SWEEP_SD_Q[GAMMA_LOW])
    med = uncalibrated_ratio(GAMMA_MED, SWEEP_SD_Q[GAMMA_MED])
    assert lo > 5.0, (
        "the uncalibrated LOW ratio is %.4f, under 5 -- if the trap were "
        "this mild the calibration layer would need re-justifying" % lo)
    assert med > 2.0, "the uncalibrated MEDIUM ratio is only %.4f" % med
    assert lo > med > 1.0, \
        "the uncalibrated ratio must fall monotonically toward the cap"
    print("  t_uncalibrated_ratio_is_trap   PASS  (a naive run would have "
          "carried %.2f sd of noise at LOW, %.2f at medium, against a "
          "design value of 1.00)" % (lo, med))


def t_gammas_defensible():
    """Every gamma committed and at or below the gate-2 cap."""
    for name, gamma in GAMMAS:
        assert 0.0 < gamma <= NOBS.GAMMA_CAP, (
            "%s gamma %r is above the gate-2 spread-realism cap %r, where "
            "the maker quotes more than 10x the market touch"
            % (name, gamma, NOBS.GAMMA_CAP))
    assert GAMMA_MED in SWEEP_SD_Q and GAMMA_LOW in SWEEP_SD_Q, \
        "LOW and MEDIUM must both be points the committed sweep measured"
    print("  t_gammas_defensible            PASS  (%s, all <= the cap)"
          % ", ".join("%s=%.2e" % (n, g) for n, g in GAMMAS))


def t_q2_rule_fires():
    """The reused rule returns each of its three verdicts on constructed curves."""
    def curves(delta):
        return [{"ma_curve": [(m, 0.5 + (delta if m == M_TOP else 0.0))
                              for m in NOBS.MA_WINDOWS]}
                for _ in range(8)]

    flat = [{"ma_curve": [(m, 0.5 + 0.0001 * (i % 3 - 1))
                          for m in NOBS.MA_WINDOWS]} for i in range(8)]
    v, _m, _se, _mu = score_q2(flat)
    assert v == "HELD", "a flat QUANT curve must be HELD, got %r" % (v,)
    v, _m, _se, _mu = score_q2(curves(0.9))
    assert v == "REFUTED (RISES)", "a rising curve must refute, got %r" % (v,)
    v, _m, _se, _mu = score_q2(curves(-0.9))
    assert v == "REFUTED (FALLS)", "a falling curve must refute, got %r" % (v,)
    print("  t_q2_rule_fires                PASS")


def t_q2_gamma_conjunction():
    """Q2-gamma refutes on a single bad gamma."""
    def cell(q_delta, j_delta):
        mk = lambda d, i: {"ma_curve": [
            (m, 0.5 + (d + 0.001 * (i % 3 - 1) if m == M_TOP else 0.0))
            for m in NOBS.MA_WINDOWS]}
        return {"res": {"QUANT": [mk(q_delta, i) for i in range(8)],
                        "JITTER": [mk(j_delta, i) for i in range(8)]}}

    good = dict((g, cell(0.0, 5.0)) for _n, g in GAMMAS)
    v, _rows = score_q2_gamma(good)
    assert v == "HELD", "three good gammas must be HELD, got %r" % (v,)

    one_bad = dict(good)
    one_bad[GAMMA_MED] = cell(5.0, 5.0)
    v, _rows = score_q2_gamma(one_bad)
    assert v == "REFUTED (QUANT is NOT flat at every gamma)", (
        "ONE non-flat gamma must refute the conjunction, got %r" % (v,))

    no_rise = dict((g, cell(0.0, 0.0)) for _n, g in GAMMAS)
    v, _rows = score_q2_gamma(no_rise)
    assert v == "REFUTED (JITTER does not rise at every gamma)", (
        "a JITTER that never climbs must refute, got %r" % (v,))
    print("  t_q2_gamma_conjunction         PASS")


def t_nan_halts():
    """NaN reaching the rule halts."""
    bad = [{"ma_curve": [(m, float("nan")) for m in NOBS.MA_WINDOWS]}
           for _ in range(8)]
    try:
        score_q2(bad)
    except DegenerateComparison:
        print("  t_nan_halts                    PASS")
        return
    raise AssertionError("a NaN QUANT curve did not halt score_q2")


def t_override_reaches_the_maker():
    """mm_override is the maker run() uses."""
    probe = make_defended("QUANT", 0, GAMMA_LOW, 0.0, 0.0303)
    assert probe.w_q == 0.0303, "make_defended did not set w_q"
    assert probe.gamma == GAMMA_LOW, "make_defended did not set gamma"
    jit = make_defended("JITTER", 3, GAMMA_MED, 0.5, 1.0)
    assert jit.sigma_eps == 0.5, "make_defended did not set sigma_eps"
    assert jit.gamma == GAMMA_MED, "make_defended did not set gamma"
    assert jit.rng is not None, "JitteredMM must carry its own RNG"
    try:
        make_defended("NOSUCHARM", 0, GAMMA_LOW, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("make_defended accepted an unknown arm")
    print("  t_override_reaches_the_maker   PASS")


def t_prepass_measures_sd_q():
    """Pre-pass returns a positive sd(q)."""
    r = NOBS.run(0, "BASELINE", gamma=GAMMA_LOW,
                 mm_override=make_defended("BASELINE", 0, GAMMA_LOW, 0.0,
                                           1.0))
    assert r["n_mm_fills"] > 0, (
        "the maker took ZERO fills at gamma %r, so sd(q) is not a "
        "measurement of anything" % (GAMMA_LOW,))
    assert r["sd_q"] > 0.0, "sd(q) is %r at LOW gamma" % (r["sd_q"],)
    assert math.isfinite(r["sd_q"]), "sd(q) is not finite"
    print("  t_prepass_measures_sd_q        PASS  (seed 0 at LOW: %.0f "
          "fills, sd(q) = %.6f BTC)" % (r["n_mm_fills"], r["sd_q"]))


def t_zero_jitter_is_baseline():
    """Control: sigma_eps = 0 reproduces BASELINE at a non-incumbent gamma."""
    base = NOBS.run(0, "BASELINE", gamma=GAMMA_LOW,
                    mm_override=make_defended("BASELINE", 0, GAMMA_LOW, 0.0,
                                              1.0))
    zero = NOBS.run(0, "JITTER", gamma=GAMMA_LOW,
                    mm_override=make_defended("JITTER", 0, GAMMA_LOW, 0.0,
                                              1.0))
    assert zero["r2_unsmoothed"] == base["r2_unsmoothed"], (
        "zero jitter gave R2 %.17g against BASELINE's %.17g at gamma %r"
        % (zero["r2_unsmoothed"], base["r2_unsmoothed"], GAMMA_LOW))
    assert zero["sd_q"] == base["sd_q"], \
        "zero jitter moved sd(q): %.17g vs %.17g" % (zero["sd_q"],
                                                     base["sd_q"])
    assert zero["n_draws"] > 0, \
        "the zero-jitter maker never drew, so the control is vacuous"
    print("  t_zero_jitter_is_baseline      PASS  (%.0f draws, R2 identical)"
          % zero["n_draws"])


def t_calibrated_jitter_degrades():
    """Calibrated jitter still degrades the fit at LOW."""
    base = NOBS.run(0, "BASELINE", gamma=GAMMA_LOW,
                    mm_override=make_defended("BASELINE", 0, GAMMA_LOW, 0.0,
                                              1.0))
    sigma_eps, w_q = calibrate(GAMMA_LOW, base["sd_q"])
    jit = NOBS.run(0, "JITTER", gamma=GAMMA_LOW,
                   mm_override=make_defended("JITTER", 0, GAMMA_LOW,
                                             sigma_eps, w_q))
    assert jit["r2_unsmoothed"] < base["r2_unsmoothed"] - 0.10, (
        "calibrated jitter moved unsmoothed R2 only from %.6f to %.6f at "
        "LOW gamma. One sd of skew must degrade the fit materially or the "
        "low arm has no range for smoothing to climb through."
        % (base["r2_unsmoothed"], jit["r2_unsmoothed"]))
    print("  t_calibrated_jitter_degrades   PASS  (sigma_eps $%.6f took "
          "R2 %.6f -> %.6f at LOW)"
          % (sigma_eps, base["r2_unsmoothed"], jit["r2_unsmoothed"]))


def t_determinism():
    """Same seed and gamma twice give identical curves."""
    a = NOBS.run(1, "QUANT", gamma=GAMMA_MED,
                 mm_override=make_defended("QUANT", 1, GAMMA_MED, 0.0,
                                           0.0157))
    b = NOBS.run(1, "QUANT", gamma=GAMMA_MED,
                 mm_override=make_defended("QUANT", 1, GAMMA_MED, 0.0,
                                           0.0157))
    assert a["ma_curve"] == b["ma_curve"], \
        "the M-curve is not reproducible at MEDIUM gamma"
    assert a["sd_q"] == b["sd_q"], "sd(q) is not reproducible"
    print("  t_determinism                  PASS  (%.0f fills, %.0f views; "
          "identical twice)" % (a["n_mm_fills"], a["n_views"]))


def selftest():
    print("=" * 74)
    print("phase8_gamma.py self-test")
    print("=" * 74)
    print("  gammas   %s"
          % ", ".join("%s %.2e" % (n, g) for n, g in GAMMAS))
    print("  incumbent SIGMA_EPS = $%.9f, W_Q_LOW = %.9f BTC"
          % (NOBS.SIGMA_EPS, BLIND.W_Q_LOW))
    print("  Q2-gamma is scored M=%d -> M=%d, paired, at %.1f SE"
          % (M_CONTROL, M_TOP, Z_SE))
    print("")
    t0 = time.time()
    t_is_identity()
    t_no_default_moved()
    t_calibration_is_noop_at_cap()
    t_calibration_holds_the_ratio()
    t_calibration_scales_linearly()
    t_uncalibrated_ratio_is_the_trap()
    t_gammas_defensible()
    t_q2_rule_fires()
    t_q2_gamma_conjunction()
    t_nan_halts()
    t_override_reaches_the_maker()
    t_prepass_measures_sd_q()
    t_zero_jitter_is_baseline()
    t_calibrated_jitter_degrades()
    t_determinism()
    print("")
    print("  all self-tests PASS in %.0fs" % (time.time() - t0))
    print("=" * 74)


# main
def main():
    bar = "=" * 118
    print(bar)
    print("Phase 8 axis 2: does Q2 survive gamma?")
    print(bar)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (NOBS.LAM, NOBS.P_MARKET, NOBS.LIFE, NOBS.DISP))
    print("k=%.5f. quote_size=%.3f BTC = DEFAULT_QUOTE_SIZE, untouched. "
          "%d seeds x %.0fs per arm, 3 arms x 3 gammas."
          % (K, QUOTE_SIZE, len(SEEDS), T))
    print("")
    print("Calibration:")
    print("  phase7_nobs freezes SIGMA_EPS at GAMMA_CAP and")
    print("  passes it to JitteredMM at any gamma.")
    print("  Skew scales linearly in gamma; frozen")
    print("  noise does not. A naive re-run at LOW would have")
    print("  carried %.2f sd of noise against a design value of 1.00:"
          % uncalibrated_ratio(GAMMA_LOW, SWEEP_SD_Q[GAMMA_LOW]))
    print("  more noise to average away,")
    print("  so JITTER would climb harder while QUANT (w/sd barely")
    print("  moves) stayed flat, confirming Q2")
    print("  at low gamma as an artifact of defense scaling.")
    print("")
    print("Pre-registered (committed before the run):")
    print("  Q2-gamma  QUANT's M=%d -> M=%d climb flat within %.1f SE "
          "at all three" % (M_CONTROL, M_TOP, Z_SE))
    print("            gammas, and JITTER's positive at %.1f SE at all three." % Z_SE)
    print("  Anchor    HIGH arm must reproduce the committed arm B climbs:")
    print("            QUANT  %+.6f +/- %.6f" % ANCHOR_QUANT)
    print("            JITTER %+.6f +/- %.6f" % ANCHOR_JITTER)
    print("")

    cells = {}
    for name, gamma in GAMMAS:
        t0 = time.time()
        cells[gamma] = run_one_gamma(gamma)
        cells[gamma]["secs"] = time.time() - t0
        print("  %-6s gamma=%.2e measured in %4.0fs"
              % (name, gamma, cells[gamma]["secs"]))
    print("")

    print(bar)
    print("Calibration (measured in this arm)")
    print(bar)
    print("  %-7s %-10s %-20s %-13s %-12s %-8s %s"
          % ("arm", "gamma", "sd(q) BTC +/- SE", "sigma_eps $", "w_q BTC",
             "w/sd", "naive N/S"))
    for name, gamma in GAMMAS:
        c = cells[gamma]
        print("  %-7s %-10.2e %.6f +/- %.6f   %-13.6f %-12.6f %-8.4f %.2f"
              % (name, gamma, c["sd_q"], c["sd_q_se"], c["sigma_eps"],
                 c["w_q"], c["w_q"] / c["sd_q"],
                 uncalibrated_ratio(gamma, c["sd_q"])))
    print("")
    print("  w/sd held at %.2f at every gamma; last"
          % BLIND.W_OVER_SD_LOW)
    print("  column = noise-to-signal ratio of an uncalibrated run")
    print("  (design value 1.00; exactly 1.00 at")
    print("  HIGH, the no-op the self-test asserts).")
    print("")

    print(bar)
    print("M-curves per gamma")
    print(bar)
    for name, gamma in GAMMAS:
        c = cells[gamma]
        print("  %s  gamma=%.2e" % (name, gamma))
        print("    %-9s %s" % ("arm", "  ".join("M=%-11d" % m
                                                for m in NOBS.MA_WINDOWS)))
        for arm in NOBS.ARMS:
            vals = []
            for m in NOBS.MA_WINDOWS:
                mv, sev = mean_se(NOBS.curve_values(c["res"][arm], m))
                vals.append("%+9.6f+-%-4.3f" % (mv, sev))
            print("    %-9s %s" % (arm, "  ".join(vals)))
        print("")

    verdict, rows = score_q2_gamma(cells)
    print(bar)
    print("Q2-gamma (pre-registered rule)")
    print(bar)
    print("  %-7s %-10s %-30s %-8s %s"
          % ("arm", "gamma", "QUANT M=1 -> M=300", "flat?", "verdict"))
    for r in rows:
        m, se, mult = r["q"]
        print("  %-7s %-10.2e %+.6f +/- %.6f %5.2f SE  %-8s %s"
              % (r["name"], r["gamma"], m, se, mult,
                 "yes" if r["q_flat"] else "no", r["q_verdict"]))
    print("")
    print("  %-7s %-10s %-30s %s"
          % ("arm", "gamma", "JITTER M=1 -> M=300", "rises?"))
    for r in rows:
        m, se, mult = r["j"]
        print("  %-7s %-10.2e %+.6f +/- %.6f %5.2f SE  %s"
              % (r["name"], r["gamma"], m, se, mult,
                 "yes" if r["j_rise"] else "no"))
    print("")
    print("  Q2-gamma -> %s" % verdict)
    print("")

    print(bar)
    print("Anchor: does HIGH reproduce the committed arm B?")
    print(bar)
    hi = cells[GAMMA_HIGH]
    qm, qse, _qmu = paired_between(hi["res"]["QUANT"], M_CONTROL, M_TOP)
    jm, jse, _jmu = paired_between(hi["res"]["JITTER"], M_CONTROL, M_TOP)
    print("  %-9s %-26s %-26s %s"
          % ("arm", "this run at HIGH", "committed 3625d32", "gap in SE"))
    for label, got, want in (("QUANT", (qm, qse), ANCHOR_QUANT),
                             ("JITTER", (jm, jse), ANCHOR_JITTER)):
        pooled = math.sqrt(got[1] * got[1] + want[1] * want[1])
        gap = abs(got[0] - want[0]) / pooled if pooled > 0 else float("inf")
        print("  %-9s %+.6f +/- %-9.6f %+.6f +/- %-9.6f %.2f"
              % (label, got[0], got[1], want[0], want[1], gap))
    print("")
    print("  Calibration is a no-op at HIGH (asserted")
    print("  by t_calibration_is_noop_at_cap), so")
    print("  these should agree to seed-level reproducibility;")
    print("  a gap would be a defect in this file.")
    print("")

    print(bar)
    print("Limitations")
    print(bar)
    print("  G1  At low gamma the hidden quantity is smaller:")
    print("      sd(skew) scales with gamma;")
    print("      at gamma 1e-6 sd(skew) = $0.02724,")
    print("      2.7 ticks ($0.01 tick).")
    print("      Below a tick the quote centre does not move")
    print("      and a null would be uninterpretable,")
    print("      so no gamma below 1e-6.")
    print("  G2  Three points, not a sweep; no shape against")
    print("      gamma is claimed.")
    print("  G3  Non-adaptive smoothing attacker, as in arm B: every")
    print("      R2 is an upper bound on defense performance.")
    print("  G4  Gate 2 caps gamma at %.2e for spread realism; nothing"
          % NOBS.GAMMA_CAP)
    print("      above the cap is run.")
    print("  G5  Defenses not calibrated to each other")
    print("      (JITTER one sd of skew, QUANT %.2f sd of inventory)."
          % BLIND.W_OVER_SD_LOW)
    print("      Only shapes in M are compared and scored;")
    print("      each defense is calibrated to itself")
    print("      across gammas.")
    print("  G6  No committed module or default modified; both defenses")
    print("      are the phase7 classes, passed")
    print("      in through run()'s mm_override.")
    print("")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
