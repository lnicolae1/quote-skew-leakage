# phase8_reference.py: the 2x2: reference (zero vs BASELINE at the same M) x seed count (8 vs 24)
# - Q2 scores QUANT's M=1 -> M=300 climb against zero; BASELINE at the same M corrects for the smoother's own blurring
# - the committed 8-seed Q2 cell is dominated by seed 7 (83.6% of the squared deviation)
# - three arms at three gammas (LOW 1.0e-6, MEDIUM 3.0e-6, HIGH 9.4e-6), 24 seeds x 86400 s; seeds 0-7 are a subset
# - calibration input fixed at SD_Q_REF at all gammas, so HIGH is the committed configuration and the anchor is reachable
# - pre-registered:
#   R1: under the BASELINE reference at 24 seeds, QUANT's climb not below BASELINE's at 2.0 SE at any gamma
#   R2: under the zero reference at 24 seeds, no seed carries more than 40% of the squared deviation
# - anchor: seeds 0-7 at HIGH reproduce the committed arm B values to printed precision

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
import phase8_gamma as GAM

from market_maker import MarketMaker
from phase7_blind import DegenerateComparison, QuantizedMM, mean_se, se_mult
from phase7_blind import require_finite
from phase7_nobs import JitteredMM, curve_at, paired_arms, paired_between
from phase8_gamma import calibrate, make_defended

# is-identity
assert paired_between is NOBS.paired_between, \
    "the paired climb must be phase7_nobs's own function"
assert paired_arms is NOBS.paired_arms, (
    "the BASELINE-relative difference must be phase7_nobs's own paired_arms "
    "-- the function phase7_nobs_results.txt:64 already applies to JITTER")
assert calibrate is GAM.calibrate, \
    "the calibration must be phase8_gamma's own, not a re-implementation"
assert make_defended is GAM.make_defended, \
    "the makers must be built by phase8_gamma's own constructor"
assert QuantizedMM is NOBS.QuantizedMM is BLIND.QuantizedMM, \
    "one QuantizedMM class across all three files, or the arms differ"
assert JitteredMM is NOBS.JitteredMM, "one JitteredMM class"
assert issubclass(QuantizedMM, MarketMaker) and issubclass(JitteredMM,
                                                           MarketMaker)

# constants
SEEDS_8 = list(NOBS.SEEDS)
SEEDS_24 = list(range(24))
T = NOBS.T
K = NOBS.K
QUOTE_SIZE = NOBS.QUOTE_SIZE
Z_SE = NOBS.Z_SE
M_CONTROL = NOBS.M_CONTROL
M_TOP = NOBS.M_TOP
ARMS = ("BASELINE", "JITTER", "QUANT")

GAMMAS = list(GAM.GAMMAS)
GAMMA_HIGH = GAM.GAMMA_HIGH

# calibration input, fixed at all three gammas
CALIB_SD_Q = NOBS.SD_Q_REF

# R2 dominance bar: 40% vs an equal share of 1/24 = 4.2%
DOM_BAR = 0.40

# anchor (committed arm B values); no rule uses them
ANCHOR = {
    "QUANT_M1": (0.846980, 0.091807),
    "QUANT_CLIMB": GAM.ANCHOR_QUANT,
    "JITTER_CLIMB": GAM.ANCHOR_JITTER,
    "BASELINE_M1": 1.000000,
}
ANCHOR_TOL = 5e-7

assert SEEDS_8 == SEEDS_24[:len(SEEDS_8)], (
    "seeds 0-7 must be the LEADING slice of the 24, or the 8-seed column is "
    "not a subset of the 24-seed one and the 2x2 is not a 2x2")
assert set(SEEDS_8) == set(range(8)), "the committed cell is seeds 0-7"
assert len(SEEDS_24) == 24 and len(set(SEEDS_24)) == 24


# 2x2 machinery
def climb_series(per_seed):
    """Per-seed M_CONTROL -> M_TOP climb."""
    return [curve_at(p, M_TOP) - curve_at(p, M_CONTROL) for p in per_seed]


def diff_curve(a, b):
    """Per-seed object whose curve is the paired arm difference a - b (BASELINE-relative climb)."""
    if len(a) != len(b):
        raise DegenerateComparison(
            "diff_curve got %d and %d seeds" % (len(a), len(b)))
    out = []
    for x, y in zip(a, b):
        wx = [w for w, _ in x["ma_curve"]]
        wy = [w for w, _ in y["ma_curve"]]
        if wx != wy:
            raise DegenerateComparison(
                "diff_curve: windows %r against %r" % (wx, wy))
        out.append({"ma_curve": [(w, rx - ry)
                                 for (w, rx), (_w, ry)
                                 in zip(x["ma_curve"], y["ma_curve"])]})
    return out


def dominance(d):
    """(most deviant seed, its share of the squared deviation)."""
    require_finite("dominance sample", *d)
    if len(d) < 2:
        raise DegenerateComparison(
            "dominance needs at least 2 values, got %d" % (len(d),))
    m = statistics.mean(d)
    sq = [(x - m) ** 2 for x in d]
    tot = sum(sq)
    if tot <= 0.0:
        return 0, 0.0
    i = max(range(len(sq)), key=lambda j: sq[j])
    return i, sq[i] / tot


def cell(label, d, seeds):
    """One 2x2 cell: mean, SE, SE multiple, worst seed's share."""
    require_finite(label, *d)
    if len(d) != len(seeds):
        raise DegenerateComparison(
            "%s: %d values for %d seeds" % (label, len(d), len(seeds)))
    m, se = mean_se(d)
    i, share = dominance(d)
    return {"label": label, "n": len(d), "mean": m, "se": se,
            "mult": se_mult(m, se), "dom_seed": seeds[i],
            "dom_share": share, "d": d}


def four_cells(quant24, base24):
    """The 2x2 at one gamma; the 8-seed column is seeds 0-7 of the 24."""
    n8 = len(SEEDS_8)
    q8, b8 = quant24[:n8], base24[:n8]
    dif24 = diff_curve(quant24, base24)
    dif8 = dif24[:n8]
    return {
        ("zero", 8): cell("zero ref, seeds 0-7", climb_series(q8), SEEDS_8),
        ("zero", 24): cell("zero ref, seeds 0-23", climb_series(quant24),
                           SEEDS_24),
        ("base", 8): cell("BASELINE ref, seeds 0-7", climb_series(dif8),
                          SEEDS_8),
        ("base", 24): cell("BASELINE ref, seeds 0-23", climb_series(dif24),
                           SEEDS_24),
    }


# the arm
def run_cell(gamma, seeds):
    """Three arms at one gamma, calibrated from CALIB_SD_Q; measured sd(q) returned, not fed back."""
    sigma_eps, w_q = calibrate(gamma, CALIB_SD_Q)
    res = {}
    for arm in ARMS:
        res[arm] = [
            NOBS.run(s, arm, gamma=gamma,
                     mm_override=make_defended(arm, s, gamma, sigma_eps,
                                               w_q))
            for s in seeds]
    sd8_m, sd8_se = mean_se([r["sd_q"] for r in res["BASELINE"][:8]])
    sd24_m, sd24_se = mean_se([r["sd_q"] for r in res["BASELINE"]])
    return {"gamma": gamma, "res": res, "sigma_eps": sigma_eps, "w_q": w_q,
            "sd_q_8": (sd8_m, sd8_se), "sd_q_24": (sd24_m, sd24_se)}


def check_anchor(cellres):
    """HIGH seeds 0-7 must reproduce the committed values to printed precision."""
    n8 = len(SEEDS_8)
    q8 = cellres["res"]["QUANT"][:n8]
    j8 = cellres["res"]["JITTER"][:n8]
    b8 = cellres["res"]["BASELINE"][:n8]

    qm1, qse1 = mean_se(NOBS.curve_values(q8, M_CONTROL))
    qcm, qcse, _ = paired_between(q8, M_CONTROL, M_TOP)
    jcm, jcse, _ = paired_between(j8, M_CONTROL, M_TOP)
    bm1, _bse1 = mean_se(NOBS.curve_values(b8, M_CONTROL))

    rows = [
        ("QUANT M=1 mean", qm1, ANCHOR["QUANT_M1"][0]),
        ("QUANT M=1 SE", qse1, ANCHOR["QUANT_M1"][1]),
        ("QUANT climb mean", qcm, ANCHOR["QUANT_CLIMB"][0]),
        ("QUANT climb SE", qcse, ANCHOR["QUANT_CLIMB"][1]),
        ("JITTER climb mean", jcm, ANCHOR["JITTER_CLIMB"][0]),
        ("JITTER climb SE", jcse, ANCHOR["JITTER_CLIMB"][1]),
        ("BASELINE M=1 mean", bm1, ANCHOR["BASELINE_M1"]),
    ]
    out = [(nm, got, want, got - want) for nm, got, want in rows]
    ok = all(abs(dl) < ANCHOR_TOL for _n, _g, _w, dl in out)
    return ok, out


# scoring
def score_r1(cells_by_gamma):
    """R1: QUANT's climb not below BASELINE's at Z_SE (BASELINE ref, 24 seeds) at any gamma."""
    rows, held = [], True
    for name, gamma in GAMMAS:
        c = cells_by_gamma[gamma][("base", 24)]
        require_finite("R1 %s" % name, c["mean"], c["se"])
        bad = (c["mean"] < 0.0) and (c["mult"] >= Z_SE)
        held = held and not bad
        rows.append({"name": name, "gamma": gamma, "cell": c, "bad": bad})
    return ("HELD" if held else
            "REFUTED (QUANT falls further than BASELINE at some gamma)"), rows


def score_r2(cells_by_gamma):
    """R2: no seed carries more than DOM_BAR of the squared deviation (zero ref, 24 seeds) at any gamma."""
    rows, held = [], True
    for name, gamma in GAMMAS:
        c = cells_by_gamma[gamma][("zero", 24)]
        require_finite("R2 %s" % name, c["dom_share"])
        bad = c["dom_share"] > DOM_BAR
        held = held and not bad
        rows.append({"name": name, "gamma": gamma, "cell": c, "bad": bad})
    return ("HELD" if held else
            "REFUTED (a single seed dominates at 24 seeds -- the seed "
            "population is heavy-tailed, not the cell small)"), rows


# self-tests
def t_calibration_no_op():
    """At GAMMA_HIGH the calibration returns the committed constants exactly."""
    se_, wq = calibrate(GAMMA_HIGH, CALIB_SD_Q)
    assert se_ == NOBS.SIGMA_EPS, (
        "sigma_eps at the cap is %.17g but phase7_nobs.SIGMA_EPS is %.17g -- "
        "the HIGH arm would NOT be the committed configuration"
        % (se_, NOBS.SIGMA_EPS))
    assert wq == BLIND.W_Q_LOW, (
        "w_q at the cap is %.17g but phase7_blind.W_Q_LOW is %.17g -- a "
        "0.011%% width change already flipped a verdict at 99fb708"
        % (wq, BLIND.W_Q_LOW))
    assert GAM.uncalibrated_ratio(GAMMA_HIGH, CALIB_SD_Q) == 1.0, \
        "the uncalibrated ratio at the cap must be exactly 1.0"
    return "calibration is an EXACT no-op at HIGH: sigma_eps=%.9f w_q=%.9f" \
        % (se_, wq)


def t_sigma_eps_scales():
    """sigma_eps scales linearly in gamma at fixed sd_q."""
    hi, _ = calibrate(GAMMA_HIGH, CALIB_SD_Q)
    obs = []
    for name, g in GAMMAS:
        se_, _wq = calibrate(g, CALIB_SD_Q)
        want = hi * (g / GAMMA_HIGH)
        assert abs(se_ - want) <= 1e-15 * max(1.0, abs(want)), (
            "sigma_eps at %s is %.17g but linear scaling wants %.17g"
            % (name, se_, want))
        obs.append((name, se_, NOBS.SIGMA_EPS / se_))
    lo_ratio = obs[0][2]
    assert lo_ratio > 5.0, (
        "the frozen-SIGMA_EPS trap should carry >5 sd of noise at LOW; the "
        "derived ratio is only %.2f, so the trap is not what it was" % lo_ratio)
    return ("sigma_eps LOW/MED/HIGH = %.9f / %.9f / %.9f; a frozen constant "
            "would have carried %.2f / %.2f / %.2f sd of noise"
            % (obs[0][1], obs[1][1], obs[2][1],
               obs[0][2], obs[1][2], obs[2][2]))


def t_w_q_constant_across_gamma():
    """w_q identical at all three gammas at fixed sd_q."""
    ws = [calibrate(g, CALIB_SD_Q)[1] for _n, g in GAMMAS]
    assert ws[0] == ws[1] == ws[2] == BLIND.W_Q_LOW, (
        "w_q differs across gammas: %r -- QUANT would be a different defense "
        "at each one" % (ws,))
    return "w_q = %.9f BTC at all three gammas (= W_Q_LOW)" % ws[0]


def t_diff_curve_matches_paired_arms():
    """diff_curve agrees with phase7_nobs.paired_arms at every window."""
    a = [{"ma_curve": [(1, 0.9), (3, 0.8), (300, 0.5)]},
         {"ma_curve": [(1, 0.4), (3, 0.3), (300, 0.7)]},
         {"ma_curve": [(1, 1.0), (3, 0.1), (300, -0.2)]}]
    b = [{"ma_curve": [(1, 1.0), (3, 0.95), (300, 0.6)]},
         {"ma_curve": [(1, 1.0), (3, 0.90), (300, 0.4)]},
         {"ma_curve": [(1, 1.0), (3, 0.85), (300, 0.3)]}]
    d = diff_curve(a, b)
    for m in (1, 3, 300):
        pm, pse, _pmult = paired_arms(a, b, m)
        vals = [curve_at(p, m) for p in d]
        dm, dse = mean_se(vals)
        assert dm == pm and dse == pse, (
            "at M=%d diff_curve gives %.17g +/- %.17g but paired_arms gives "
            "%.17g +/- %.17g" % (m, dm, dse, pm, pse))
    return "diff_curve reproduces paired_arms at M=1, 3, 300 exactly"


def t_baseline_climb_is_not_zero_climb():
    """The two references give different statistics on the same data."""
    q = [{"ma_curve": [(1, 0.90), (300, 0.86)]},
         {"ma_curve": [(1, 0.80), (300, 0.77)]}]
    b = [{"ma_curve": [(1, 1.00), (300, 0.90)]},
         {"ma_curve": [(1, 1.00), (300, 0.91)]}]
    zero = climb_series(q)
    base = climb_series(diff_curve(q, b))
    assert all(x < 0 for x in zero), zero
    assert all(x > 0 for x in base), base
    zm, _ = mean_se(zero)
    bm, _ = mean_se(base)
    assert zm < 0.0 < bm, (zm, bm)
    return ("zero ref %+.6f (a FALL) vs BASELINE ref %+.6f (a RISE) on the "
            "same curves" % (zm, bm))


# committed per-seed M=1 QUANT levels at HIGH (quant_repro_results.txt arm A), transcribed
COMMITTED_M1_HIGH = [0.970847909, 0.974923769, 0.966281516, 0.913032548,
                     0.818646811, 0.923988765, 0.989371044, 0.218746354]


def t_dominance_arithmetic():
    """dominance reproduces the committed seed-7 share from the committed per-seed values."""
    m, se = mean_se(COMMITTED_M1_HIGH)
    assert abs(m - ANCHOR["QUANT_M1"][0]) < ANCHOR_TOL, (
        "the per-seed values give mean %.6f, not the committed %.6f -- they "
        "are not the committed values" % (m, ANCHOR["QUANT_M1"][0]))
    assert abs(se - ANCHOR["QUANT_M1"][1]) < ANCHOR_TOL, (
        "the per-seed values give SE %.6f, not the committed %.6f"
        % (se, ANCHOR["QUANT_M1"][1]))
    i, share = dominance(COMMITTED_M1_HIGH)
    assert i == 7, "the most deviant seed should be index 7, got %d" % i
    assert abs(share - 0.836) < 0.002, (
        "seed 7's share is %.4f; the committed reading is 0.836" % share)
    flat = dominance([1.0, 1.0, 1.0, 1.0])
    assert flat == (0, 0.0), flat
    return ("the committed values give %.6f +/- %.6f; seed 7 carries %.1f%% "
            "of the squared deviation" % (m, se, 100.0 * share))


def t_lone_outlier_does_not_dilute():
    """Padding one outlier with well-behaved seeds does not dilute its share much."""
    _i8, s8 = dominance(COMMITTED_M1_HIGH)

    tight = [x for x in COMMITTED_M1_HIGH if x > 0.5]
    lone = COMMITTED_M1_HIGH + [tight[j % len(tight)] for j in range(16)]
    _il, s_lone = dominance(lone)
    assert s_lone >= s8, (
        "padding with well-behaved seeds LOWERED the share, %.4f -> %.4f. "
        "If that were true R2 would be a test of sample size, and the "
        "pre-registration says it is not." % (s8, s_lone))
    assert s_lone > DOM_BAR, (
        "a lone outlier among 24 gives share %.4f, which is below the %.2f "
        "bar -- R2 would then hold even in the branch it exists to detect"
        % (s_lone, DOM_BAR))

    company = (COMMITTED_M1_HIGH
               + [tight[j % len(tight)] for j in range(13)]
               + [0.30, 0.25, 0.35])
    _ic, s_comp = dominance(company)
    assert s_comp < DOM_BAR, (
        "with three MORE seeds like seed 7 the share is %.4f, still above "
        "the %.2f bar -- then R2 could not hold in either branch and the "
        "bar is mis-set" % (s_comp, DOM_BAR))
    return ("seed 7 alone: %.1f%% of 8 -> %.1f%% of 24 (NOT diluted); with "
            "three seeds like it: %.1f%%. Bar %.0f%% separates the two."
            % (100.0 * s8, 100.0 * s_lone, 100.0 * s_comp, 100.0 * DOM_BAR))


def t_r1_can_fail():
    """R1 fires and holds on constructed cells."""
    def mk(mean, se):
        return {("base", 24): {"label": "x", "n": 24, "mean": mean, "se": se,
                               "mult": se_mult(mean, se), "dom_seed": 0,
                               "dom_share": 0.0, "d": [mean] * 24}}
    bad = {g: mk(-0.05, 0.01) for _n, g in GAMMAS}
    v, rows = score_r1(bad)
    assert v.startswith("REFUTED"), v
    assert all(r["bad"] for r in rows), rows

    good = {g: mk(+0.05, 0.01) for _n, g in GAMMAS}
    v2, rows2 = score_r1(good)
    assert v2 == "HELD", v2
    assert not any(r["bad"] for r in rows2), rows2

    # negative but not significant must hold
    weak = {g: mk(-0.05, 0.10) for _n, g in GAMMAS}
    assert score_r1(weak)[0] == "HELD", "R1 fired on a 0.5 SE decline"

    # one bad gamma out of three refutes the conjunction
    mixed = {GAMMAS[0][1]: mk(+0.05, 0.01),
             GAMMAS[1][1]: mk(+0.05, 0.01),
             GAMMAS[2][1]: mk(-0.05, 0.01)}
    assert score_r1(mixed)[0].startswith("REFUTED"), \
        "one failing gamma must refute the conjunction"
    return "R1 refutes on a 5.0 SE decline, holds on a 0.5 SE one and on a rise"


def t_r2_can_fail():
    """R2 fires at and around the bar."""
    def mk(share):
        return {("zero", 24): {"label": "x", "n": 24, "mean": -0.01,
                               "se": 0.005, "mult": 2.0, "dom_seed": 7,
                               "dom_share": share, "d": [-0.01] * 24}}
    assert score_r2({g: mk(0.85) for _n, g in GAMMAS})[0].startswith("REFUTED")
    assert score_r2({g: mk(0.10) for _n, g in GAMMAS})[0] == "HELD"
    assert score_r2({g: mk(DOM_BAR) for _n, g in GAMMAS})[0] == "HELD", \
        "the bar is exclusive: exactly DOM_BAR must not refute"
    mixed = {GAMMAS[0][1]: mk(0.10),
             GAMMAS[1][1]: mk(0.10),
             GAMMAS[2][1]: mk(0.85)}
    assert score_r2(mixed)[0].startswith("REFUTED"), \
        "one dominated gamma must refute"
    return "R2 refutes at 85%%, holds at 10%% and exactly at the %.0f%% bar" \
        % (100.0 * DOM_BAR)


def t_anchor_check_fires():
    """check_anchor fails on a curve perturbed by 1e-5."""
    def mkcell(q_m1, q_top, j_m1, j_top):
        n = len(SEEDS_8)
        return {"res": {
            "QUANT": [{"ma_curve": [(1, q_m1[i]), (300, q_top[i])]}
                      for i in range(n)],
            "JITTER": [{"ma_curve": [(1, j_m1[i]), (300, j_top[i])]}
                       for i in range(n)],
            "BASELINE": [{"ma_curve": [(1, 1.0), (300, 0.984353)]}
                         for _ in range(n)]}}

    # values chosen so means and SEs land on the anchor
    qm1 = [0.970847909, 0.936193096, 0.958270142, 0.989370682,
           0.887861712, 0.860291255, 0.954260173, 0.218746354]
    ok0, rows0 = check_anchor(mkcell(qm1, [x - 0.006231 for x in qm1],
                                     [0.0] * 8, [0.0] * 8))
    # QUANT M=1 mean must match; climb SE will not (constant climb)
    m1row = [r for r in rows0 if r[0] == "QUANT M=1 mean"][0]
    assert abs(m1row[3]) < ANCHOR_TOL, (
        "the QUANT M=1 mean should reproduce on the committed per-seed "
        "values, got delta %.3e" % m1row[3])

    bumped = list(qm1)
    bumped[7] += 1e-5
    ok1, rows1 = check_anchor(mkcell(bumped,
                                     [x - 0.006231 for x in bumped],
                                     [0.0] * 8, [0.0] * 8))
    m1row2 = [r for r in rows1 if r[0] == "QUANT M=1 mean"][0]
    assert abs(m1row2[3]) >= ANCHOR_TOL, (
        "a 1e-5 move on one seed slipped past the anchor tolerance -- the "
        "check would not have caught the 0.167 move at 99fb708 either")
    assert not ok1, "check_anchor reported OK on a perturbed curve"
    return ("anchor passes the committed M=1 mean and FAILS a 1e-5 move on "
            "one seed")


def t_eight_is_a_slice_of_24():
    """8-seed column is a literal subset of the 24-seed run."""
    fake = [{"ma_curve": [(1, 0.9 + 0.001 * s), (300, 0.8 + 0.001 * s)]}
            for s in SEEDS_24]
    cells = four_cells(fake, [{"ma_curve": [(1, 1.0), (300, 0.98)]}
                              for _ in SEEDS_24])
    d8 = cells[("zero", 8)]["d"]
    d24 = cells[("zero", 24)]["d"]
    assert d8 == d24[:8], (
        "the 8-seed cell is not the leading slice of the 24-seed one")
    assert cells[("zero", 8)]["n"] == 8
    assert cells[("zero", 24)]["n"] == 24
    return "seeds 0-7 cell is the literal leading slice of the 24-seed cell"


def t_no_default_moved():
    """No committed default shifted."""
    assert NOBS.GAMMA_CAP == 9.4e-6, "GAMMA_CAP moved"
    assert NOBS.SD_Q_REF == 0.04604, "SD_Q_REF moved"
    assert BLIND.W_OVER_SD_LOW == 0.34, "W_OVER_SD_LOW moved"
    assert BLIND.W_Q_LOW == BLIND.W_OVER_SD_LOW * NOBS.SD_Q_REF, \
        "W_Q_LOW is no longer 0.34 * SD_Q_REF"
    assert NOBS.SIGMA_EPS_MULT == 1.0, "SIGMA_EPS_MULT moved"
    assert NOBS.Z_SE == 2.0, "Z_SE moved"
    assert NOBS.M_CONTROL == 1 and NOBS.M_TOP == 300, "the windows moved"
    assert NOBS.MA_WINDOWS == [1, 3, 10, 30, 100, 300], "MA_WINDOWS moved"
    assert NOBS.QUOTE_SIZE == 0.020, "quote_size moved"
    assert NOBS.K == 0.17763, "k moved"
    assert [g for _n, g in GAMMAS] == [1.0e-6, 3.0e-6, 9.4e-6], "gammas moved"
    return ("GAMMA_CAP, SD_Q_REF, W_Q_LOW, SIGMA_EPS_MULT, Z_SE, windows, "
            "quote_size, k and the three gammas all unmoved")


def t_zero_jitter_is_baseline():
    """sigma_eps = 0 reproduces BASELINE at LOW gamma."""
    g = GAMMAS[0][1]
    b = NOBS.run(0, "BASELINE", gamma=g,
                 mm_override=MarketMaker(horizon=T, k=K, gamma=g,
                                         quote_size=QUOTE_SIZE))
    j = NOBS.run(0, "JITTER", gamma=g,
                 mm_override=JitteredMM(0.0, NOBS.JITTER_SEED_BASE, horizon=T,
                                        k=K, gamma=g, quote_size=QUOTE_SIZE))
    assert b["ma_curve"] == j["ma_curve"], (
        "zero jitter did not reproduce BASELINE:\n  %r\n  %r"
        % (b["ma_curve"], j["ma_curve"]))
    assert j["n_draws"] > 0, "JitteredMM never drew -- the path is not live"
    return "zero-jitter == BASELINE bit for bit over %d draws at LOW gamma" \
        % int(j["n_draws"])


def t_determinism():
    """Same seed and arm give the same curve twice."""
    g = GAMMAS[2][1]
    se_, wq = calibrate(g, CALIB_SD_Q)
    a = NOBS.run(3, "QUANT", gamma=g, horizon=3600.0,
                 mm_override=make_defended("QUANT", 3, g, se_, wq))
    b = NOBS.run(3, "QUANT", gamma=g, horizon=3600.0,
                 mm_override=make_defended("QUANT", 3, g, se_, wq))
    assert a["ma_curve"] == b["ma_curve"], "QUANT is not deterministic"
    return "QUANT at HIGH reproduces its own curve exactly on re-run"


def t_new_seeds_are_live():
    """Seeds 8-23 produce distinct worlds."""
    g = GAMMAS[2][1]
    se_, wq = calibrate(g, CALIB_SD_Q)
    out = {}
    for s in (0, 8, 23):
        r = NOBS.run(s, "BASELINE", gamma=g,
                     mm_override=make_defended("BASELINE", s, g, se_, wq))
        assert r["n_views"] > 0, "seed %d produced no views" % s
        assert r["sd_q"] > 0, "seed %d had a constant inventory" % s
        out[s] = r
    assert out[0]["ma_curve"] != out[8]["ma_curve"], \
        "seed 8 reproduced seed 0's curve -- the seed is not reaching the world"
    assert out[8]["ma_curve"] != out[23]["ma_curve"], \
        "seeds 8 and 23 gave identical curves"
    return ("seeds 0, 8, 23 give distinct curves; sd(q) = %.6f / %.6f / %.6f"
            % (out[0]["sd_q"], out[8]["sd_q"], out[23]["sd_q"]))


def t_degenerate_inputs_halt():
    """Empty and mismatched inputs raise."""
    for bad in ([], [float("nan"), 1.0], [1.0, float("inf")]):
        try:
            dominance(bad)
        except DegenerateComparison:
            pass
        except Exception as e:
            raise AssertionError("dominance(%r) raised %r" % (bad, e))
        else:
            raise AssertionError("dominance(%r) did not raise" % (bad,))
    try:
        diff_curve([{"ma_curve": [(1, 0.5)]}], [])
    except DegenerateComparison:
        pass
    else:
        raise AssertionError("diff_curve accepted mismatched seed counts")
    try:
        diff_curve([{"ma_curve": [(1, 0.5)]}], [{"ma_curve": [(3, 0.5)]}])
    except DegenerateComparison:
        pass
    else:
        raise AssertionError("diff_curve accepted mismatched windows")
    try:
        cell("x", [1.0, 2.0], [0])
    except DegenerateComparison:
        pass
    else:
        raise AssertionError("cell accepted 2 values for 1 seed")
    return "empty, NaN, inf, mismatched seeds and mismatched windows all halt"


def selftest():
    tests = [t_calibration_no_op, t_sigma_eps_scales,
             t_w_q_constant_across_gamma, t_diff_curve_matches_paired_arms,
             t_baseline_climb_is_not_zero_climb, t_dominance_arithmetic,
             t_lone_outlier_does_not_dilute, t_r1_can_fail, t_r2_can_fail,
             t_anchor_check_fires, t_eight_is_a_slice_of_24,
             t_no_default_moved, t_zero_jitter_is_baseline, t_determinism,
             t_new_seeds_are_live, t_degenerate_inputs_halt]
    print("=" * 100)
    print("Self-tests; phase8_reference.py")
    print("=" * 100)
    bad = 0
    for f in tests:
        t0 = time.time()
        try:
            msg = f()
            print("  PASS  %-34s %6.1fs  %s"
                  % (f.__name__, time.time() - t0, msg))
        except Exception as e:
            bad += 1
            print("  FAIL  %-34s %6.1fs  %s: %s"
                  % (f.__name__, time.time() - t0, type(e).__name__, e))
    print("")
    print("  %d of %d PASS" % (len(tests) - bad, len(tests)))
    return bad == 0


# report
def _fmt_cell(c):
    return ("%+.6f +/- %.6f  %5.2f SE   worst seed %2d carries %5.1f%%"
            % (c["mean"], c["se"], c["mult"], c["dom_seed"],
               100.0 * c["dom_share"]))


def main():
    bar = "=" * 100
    t_start = time.time()
    print(bar)
    print("Phase 8: 2x2 of reference and seed count")
    print(bar)
    print("Working point: k=%.5f, quote_size=%.3f BTC, %d seeds x %.0fs per "
          "arm, 3 arms x 3 gammas." % (K, QUOTE_SIZE, len(SEEDS_24), T))
    print("Calibration input: sd_q = SD_Q_REF = %.5f at all three gammas, "
          "NOT a measured per-cell sd(q)." % CALIB_SD_Q)
    print("  (deviation from phase8_gamma's fix (3), deliberate:")
    print("  calibrate(GAMMA_CAP, SD_Q_REF) is an exact no-op, so the HIGH "
          "arm is the committed configuration")
    print("  and the anchor is reachable; a measured sd(q) at 24 seeds "
          "would make it unreachable")
    print("  (a 0.011% width change already flipped a verdict).")
    print("  sigma_eps still scales with gamma (fix 1) and w/sd is still "
          "held at %.2f (fix 2)." % BLIND.W_OVER_SD_LOW)
    print("")
    print("Pre-registered (committed before this ran):")
    print("  R1  BASELINE ref, 24 seeds: QUANT's climb not below "
          "BASELINE's at %.0f SE, at any gamma." % Z_SE)
    print("  R2  zero ref, 24 seeds: no single seed carries more than "
          "%.0f%% of the squared deviation." % (100.0 * DOM_BAR))
    print("      R2 failing at 24 seeds = heavy-tailed seed "
          "population, not a small cell.")
    print("")

    # HIGH first: an anchor failure costs a third of the run
    order = [("HIGH", GAMMA_HIGH)] + [(n, g) for n, g in GAMMAS
                                      if g != GAMMA_HIGH]
    cells_run, timings = {}, {}
    for name, g in order:
        t0 = time.time()
        print("running %-6s gamma=%.2e ..." % (name, g))
        sys.stdout.flush()
        cells_run[g] = run_cell(g, SEEDS_24)
        timings[g] = time.time() - t0
        print("  done in %.0fs" % timings[g])
        sys.stdout.flush()
        if g == GAMMA_HIGH:
            ok, rows = check_anchor(cells_run[g])
            print("")
            print(bar)
            print("Anchor: seeds 0-7 at HIGH must reproduce the committed values")
            print(bar)
            for nm, got, want, dl in rows:
                print("  %-20s got %+.9f   committed %+.9f   delta %+.3e  %s"
                      % (nm, got, want, dl,
                         "OK" if abs(dl) < ANCHOR_TOL else "mismatch"))
            print("")
            if not ok:
                print("  ANCHOR FAILED: seeds 0-7 are the same")
                print("  worlds as the committed 8-seed run and")
                print("  the mm_override path is inert at the")
                print("  committed width, so the")
                print("  harness has drifted; cells below")
                print("  would be uninterpretable and are not")
                print("  reported.")
                raise DegenerateComparison(
                    "the HIGH / seeds 0-7 cell does not reproduce 3625d32")
            print("  Anchor held: HIGH is the committed configuration; the "
                  "four cells below are readable")
            print("  against the committed values and each other.")
            print("")

    # calibration and sd(q)
    print(bar)
    print("Calibration, and the measured sd(q) not fed back")
    print(bar)
    print("  gamma      sigma_eps($)   w_q(BTC)      measured sd(q) 8 seeds"
          "      24 seeds          24 vs SD_Q_REF")
    for name, g in GAMMAS:
        c = cells_run[g]
        print("  %-8s   %-12.6f   %-11.9f   %.6f +/- %.6f      "
              "%.6f +/- %.6f   %.4fx"
              % (name, c["sigma_eps"], c["w_q"],
                 c["sd_q_8"][0], c["sd_q_8"][1],
                 c["sd_q_24"][0], c["sd_q_24"][1],
                 c["sd_q_24"][0] / CALIB_SD_Q))
    print("")
    print("  Last column: size of the fix (3) deviation")
    print("  (far from 1.0 = the deviation")
    print("  cost something).")
    print("")

    # per-seed climbs
    print(bar)
    print("PER-SEED CLIMBS, M=1 -> M=300, all three arms at all three gammas")
    print(bar)
    print("")
    for name, g in GAMMAS:
        c = cells_run[g]
        print("  gamma = %s (%.2e)" % (name, g))
        print("    seed |   BASE M=1   BASE M=300  BASE climb |"
              "  QUANT M=1  QUANT M=300 QUANT climb |"
              " JITTER M=1 JITTER M=300 JIT climb |  Q-B climb")
        b, q, j = c["res"]["BASELINE"], c["res"]["QUANT"], c["res"]["JITTER"]
        for i, s in enumerate(SEEDS_24):
            b1, b3 = curve_at(b[i], M_CONTROL), curve_at(b[i], M_TOP)
            q1, q3 = curve_at(q[i], M_CONTROL), curve_at(q[i], M_TOP)
            j1, j3 = curve_at(j[i], M_CONTROL), curve_at(j[i], M_TOP)
            print("    %4d | %+10.6f %+11.6f %+11.6f |"
                  " %+10.6f %+11.6f %+11.6f |"
                  " %+10.4f %+11.4f %+10.4f | %+10.6f"
                  % (s, b1, b3, b3 - b1, q1, q3, q3 - q1, j1, j3, j3 - j1,
                     (q3 - q1) - (b3 - b1)))
        print("")

    # 2x2
    cells_by_gamma = {g: four_cells(cells_run[g]["res"]["QUANT"],
                                    cells_run[g]["res"]["BASELINE"])
                      for _n, g in GAMMAS}
    print(bar)
    print("THE 2x2 -- QUANT'S CLIMB")
    print(bar)
    print("  Down a column: effect of seed count.")
    print("  Across a row: effect of reference.")
    print("")
    for name, g in GAMMAS:
        fc = cells_by_gamma[g]
        print("  gamma = %s (%.2e)" % (name, g))
        for key in (("zero", 8), ("zero", 24), ("base", 8), ("base", 24)):
            c = fc[key]
            print("    %-26s %s" % (c["label"], _fmt_cell(c)))
        print("")
    print("  The zero-ref / seeds 0-7 / HIGH row is the committed Q2 cell:")
    print("  %+.6f +/- %.6f against the committed %+.6f +/- %.6f."
          % (cells_by_gamma[GAMMA_HIGH][("zero", 8)]["mean"],
             cells_by_gamma[GAMMA_HIGH][("zero", 8)]["se"],
             ANCHOR["QUANT_CLIMB"][0], ANCHOR["QUANT_CLIMB"][1]))
    print("")

    # JITTER 2x2 (descriptive)
    print(bar)
    print("THE SAME 2x2 FOR JITTER (descriptive, not scored)")
    print(bar)
    print("  (JITTER's rise was the other half of Q2-gamma;")
    print("  no per-seed value for it has ever existed either. No rule fires")
    print("  on these numbers.")
    print("")
    for name, g in GAMMAS:
        c = cells_run[g]
        jc = four_cells(c["res"]["JITTER"], c["res"]["BASELINE"])
        print("  gamma = %s (%.2e)" % (name, g))
        for key in (("zero", 8), ("zero", 24), ("base", 8), ("base", 24)):
            cc = jc[key]
            print("    %-26s %s" % (cc["label"], _fmt_cell(cc)))
        print("")

    # rules
    print(bar)
    print("Rules")
    print(bar)
    v1, rows1 = score_r1(cells_by_gamma)
    print("  R1  QUANT's climb not below BASELINE's at %.0f SE, at any gamma" % Z_SE)
    for r in rows1:
        c = r["cell"]
        print("      %-8s %+.6f +/- %.6f  %5.2f SE   %s"
              % (r["name"], c["mean"], c["se"], c["mult"],
                 "violation" if r["bad"] else "ok"))
    print("      -> R1 %s" % v1)
    print("")
    v2, rows2 = score_r2(cells_by_gamma)
    print("  R2  no single seed carries more than %.0f%% of the squared "
          "deviation at 24 seeds" % (100.0 * DOM_BAR))
    for r in rows2:
        c = r["cell"]
        print("      %-8s worst seed %2d carries %5.1f%%   %s"
              % (r["name"], c["dom_seed"], 100.0 * c["dom_share"],
                 "violation" if r["bad"] else "ok"))
    print("      -> R2 %s" % v2)
    print("")
    if v2.startswith("REFUTED"):
        print("  R2 FAILED: the diagnosis")
        print("  (seed 7 ordinary, cell too small)")
        print("  is contradicted; heavy-tailed seed population;")
        print("  more seeds will not converge; the")
        print("  estimator needs fixing. Every 8-seed cell in")
        print("  phases 6-8 is then under-powered rather than")
        print("  unlucky.")
        print("")

    print(bar)
    print("Scope")
    print(bar)
    print("  Q2's committed HELD and Q2-gamma's committed REFUTED")
    print("  stand as what those rules returned; nothing")
    print("  above re-scores either. Adds the missing cells")
    print("  and the per-seed values.")
    print("  BASELINE reference corrects for")
    print("  the smoother's blurring only.")
    print("  Non-adaptive attacker: every R2 is")
    print("  an upper bound on what this sniffer extracts.")
    print("")
    print("  total wall clock %.0fs (HIGH %.0fs, medium %.0fs, LOW %.0fs)"
          % (time.time() - t_start, timings[GAMMA_HIGH],
             timings[GAMMAS[1][1]], timings[GAMMAS[0][1]]))
    print("")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
