"""Phase 7; the width sweep"""

import contextlib
import io
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import participation_skew as PSMOD
import phase7_blind as BLIND
import sniffer_fill_only as FILLMOD
import sniffer_skew as SKMOD

from market_maker import MarketMaker, DEFAULT_GAMMA
from participation_skew import FlatQuoteMM, series_stats
from sniffer_fill_only import ols, TRAIN_FRAC
from sniffer_skew import build_features, fit_score, TAU_FLOOR_FRAC

from phase7_blind import (DegenerateComparison, GAMMA_CAP, K, MAX_ABS_Q_REF,
                          QUOTE_SIZE, QuantizedMM, SD_Q_REF, SELFTEST_SHORT,
                          T, mean_se, paired, require_finite, se_mult)

# is-identity
assert QuantizedMM is BLIND.QuantizedMM, \
    "QuantizedMM must be phase7_blind's own class, not a reimplementation"
assert FlatQuoteMM is PSMOD.FlatQuoteMM, \
    "FlatQuoteMM must be participation_skew's own class, not a copy"
assert series_stats is PSMOD.series_stats, \
    "series_stats (R_direct) must be participation_skew's own object"
assert build_features is SKMOD.build_features, \
    "build_features must be sniffer_skew's own object"
assert fit_score is SKMOD.fit_score, \
    "fit_score must be sniffer_skew's own object"
assert fit_score is BLIND.fit_score, \
    "phase7_blind must be holding the same fit_score object this file holds"
assert ols is FILLMOD.ols, \
    ("ols must be sniffer_fill_only's own object: the diagnostic refit uses "
     "it to reproduce the committed fit and would otherwise be checking a "
     "different solver against fit_score")
assert TAU_FLOOR_FRAC is SKMOD.TAU_FLOOR_FRAC, \
    "TAU_FLOOR_FRAC must be sniffer_skew's own value"
assert mean_se is BLIND.mean_se and se_mult is BLIND.se_mult, \
    "the statistics layer must be phase7_blind's own, not a copy"
assert paired is BLIND.paired and require_finite is BLIND.require_finite, \
    "the statistics layer must be phase7_blind's own, not a copy"
assert DegenerateComparison is BLIND.DegenerateComparison, \
    "the halt exception must be phase7_blind's own class"
assert issubclass(QuantizedMM, MarketMaker), \
    "QuantizedMM must subclass the committed MarketMaker"


# constants
SEEDS = list(range(24))

# band from inverting the cap: w/sd = sqrt(12 * (1 - cap))
W_OVER_SD = (0.75, 1.00, 1.25, 1.50, 1.75)
WIDTHS = tuple(r * SD_Q_REF for r in W_OVER_SD)

# Bonferroni over five widths: 0.05 / 5 = 0.01 two-sided -> z = 2.576
Z_NOMINAL = 2.0
Z_FAMILY = 2.58
N_COMPARISONS = len(W_OVER_SD)

MARGIN_R2 = 0.15
SE_R2_COMMITTED_8 = 0.091807

# P1: BASELINE R2 is an algebraic identity
P1_TARGET = 1.0
P1_TOL = 1e-6

# P5 plausibility thresholds (change labels only)
FLAG_RATIO = 5.0
FLAG_MAGN = 3.0

# adversary model: build_features columns except the full-mid sensitivity (10)
N_MUL_COLS = 10

BASE = "BASELINE"
FLAT = "FLAT"


def width_label(ratio):
    """Arm name for a width (inert when mm_override is given)."""
    return "Q-%.2f" % ratio


WIDTH_ARMS = tuple(width_label(r) for r in W_OVER_SD)
ARMS = (BASE,) + WIDTH_ARMS + (FLAT,)


def analytic_cap(w_q, sd_q=SD_Q_REF):
    """R2 ceiling under quantization: 1 - (w^2/12) / Var(q)."""
    return 1.0 - (w_q * w_q / 12.0) / (sd_q * sd_q)


# per-fit diagnostics (fit_score swapped for a capturing wrapper, restored in a finally)
_CAPTURE = {}


def _standardized_pivot_ratio(X):
    """Pivot max/min on the unit-sd Gram matrix (condition proxy)."""
    p = len(X[0])
    sds = []
    for c in range(p):
        col = [row[c] for row in X]
        sd = statistics.pstdev(col) if len(col) > 1 else 0.0
        sds.append(sd if sd > 0 else 1.0)
    A = [[0.0] * p for _ in range(p)]
    for xi in X:
        z = [xi[a] / sds[a] for a in range(p)]
        for a in range(p):
            za = z[a]
            Aa = A[a]
            for b in range(a, p):
                Aa[b] += za * z[b]
    for a in range(p):
        for b in range(a):
            A[a][b] = A[b][a]
    piv = []
    for c in range(p):
        pr = max(range(c, p), key=lambda r: abs(A[r][c]))
        A[c], A[pr] = A[pr], A[c]
        piv.append(abs(A[c][c]))
        if A[c][c] == 0:
            continue
        for r in range(p):
            if r == c:
                continue
            f = A[r][c] / A[c][c]
            for cc in range(c, p):
                A[r][cc] -= f * A[c][cc]
    nz = [v for v in piv if v > 0]
    if not nz:
        return float("inf")
    return max(piv) / min(nz)


def _capturing_fit_score(rows, y, keep, cols):
    """Committed fit_score, returned verbatim; first call also records diagnostics."""
    out = fit_score(rows, y, keep, cols)
    if "pred" in _CAPTURE:
        return out
    if len(cols) != N_MUL_COLS:
        raise DegenerateComparison(
            "the first fit_score call used %d columns, expected %d. The "
            "diagnostic is attached to the multi-feature fit and the call "
            "order has changed." % (len(cols), N_MUL_COLS))
    idx = [i for i in range(len(y)) if keep[i]]
    cut = int(TRAIN_FRAC * len(idx))
    tr, te = idx[:cut], idx[cut:]
    Xtr = [[rows[i][c] for c in cols] for i in tr]
    Xte = [[rows[i][c] for c in cols] for i in te]
    ytr = [y[i] for i in tr]
    yte = [y[i] for i in te]
    beta = ols(Xtr, ytr)
    pred = [sum(b * x for b, x in zip(beta, row)) for row in Xte]
    m = statistics.mean(yte)
    ss_res = sum((a - b) ** 2 for a, b in zip(yte, pred))
    ss_tot = sum((a - m) ** 2 for a in yte)
    r2_check = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if not (math.isnan(r2_check) and math.isnan(out[0])):
        if abs(r2_check - out[0]) > 1e-9:
            raise DegenerateComparison(
                "the diagnostic refit does not reproduce the committed "
                "fit_score: %.12f vs %.12f. Every diagnostic below would be "
                "describing a different fit." % (r2_check, out[0]))
    _CAPTURE["pred"] = pred
    _CAPTURE["yte"] = yte
    _CAPTURE["ytr"] = ytr
    _CAPTURE["pivot_std"] = _standardized_pivot_ratio(Xtr)
    return out


def run_diag(seed, arm, mm=None):
    """phase7_blind.run() with per-fit diagnostics; fit_score restored in a finally."""
    _CAPTURE.clear()
    BLIND.fit_score = _capturing_fit_score
    try:
        out = BLIND.run(seed, arm, mm_override=mm)
    finally:
        BLIND.fit_score = fit_score
    if "pred" not in _CAPTURE:
        raise DegenerateComparison(
            "run(%r, %r) completed without the diagnostic ever capturing a "
            "fit. Nothing was scored." % (seed, arm))
    pred, yte, ytr = _CAPTURE["pred"], _CAPTURE["yte"], _CAPTURE["ytr"]
    sd_y = statistics.pstdev(yte) if len(yte) > 1 else 0.0
    sd_p = statistics.pstdev(pred) if len(pred) > 1 else 0.0
    lo, hi = min(ytr), max(ytr)
    mq = out["max_abs_q"]
    out["diag_ratio"] = (sd_p / sd_y) if sd_y > 0 else float("inf")
    out["diag_magn"] = (max(abs(v) for v in pred) / mq) if mq > 0 \
        else float("inf")
    out["diag_outfrac"] = (sum(1 for v in yte if v < lo or v > hi)
                           / float(len(yte))) if yte else 0.0
    out["diag_pivot_std"] = _CAPTURE["pivot_std"]
    out["flagged"] = (out["diag_ratio"] >= FLAG_RATIO
                      or out["diag_magn"] > FLAG_MAGN)
    return out


def make_width_mm(w_q):
    """QuantizedMM at one width."""
    return QuantizedMM(w_q, horizon=T, k=K, gamma=GAMMA_CAP,
                       quote_size=QUOTE_SIZE)


# bounded companion statistic
def sign_split(diffs):
    """(n_positive, n_negative, n_zero) of paired differences."""
    require_finite("sign_split", *diffs)
    pos = sum(1 for d in diffs if d > 0.0)
    neg = sum(1 for d in diffs if d < 0.0)
    return pos, neg, len(diffs) - pos - neg


def sign_p(n, c):
    """Exact two-sided binomial p at 0.5; ties count into n."""
    if n <= 0:
        raise DegenerateComparison("sign_p got n=%d; an empty sign test" % n)
    if c < 0 or c > n:
        raise DegenerateComparison("sign_p got c=%d out of n=%d" % (c, n))
    k = max(c, n - c)
    tail = sum(math.comb(n, j) for j in range(k, n + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def leg(diffs, direction):
    """One leg's headline and companion"""
    m, se = mean_se(diffs)
    mult = se_mult(m, se)
    pos, neg, _zero = sign_split(diffs)
    n_pass = pos if direction > 0 else neg
    p = sign_p(len(diffs), n_pass)
    passed = (m * direction > 0.0) and (mult >= Z_FAMILY)
    return m, se, mult, n_pass, p, passed


def leg3_noninferiority(diffs):
    """LEG 3"""
    m, se = mean_se(diffs)
    ub = m + Z_FAMILY * se
    n_below = sum(1 for d in diffs if d < MARGIN_R2)
    p = sign_p(len(diffs), n_below)
    return m, se, ub, n_below, p, (ub < MARGIN_R2)


# preconditions
def p1_identity(res_base):
    """BASELINE R2 must reproduce 1.000000."""
    m, se = mean_se([r["r2_mul"] for r in res_base])
    ok = abs(m - P1_TARGET) <= P1_TOL
    return ok, m, se


def p2_endpoint(res_base, res_flat):
    """FLAT must blind and cost."""
    _d, bm, bse = paired(res_base, res_flat, "r2_mul")
    _c, cm, cse = paired(res_base, res_flat, "r_direct")
    b_mult, c_mult = se_mult(bm, bse), se_mult(cm, cse)
    blinds = (bm > 0.0) and (b_mult >= Z_FAMILY)
    costs = (cm > 0.0) and (c_mult >= Z_FAMILY)
    return (blinds and costs), bm, bse, b_mult, cm, cse, c_mult


def p3_monotone(res_by_arm):
    """Blinding must not fall as w widens. Returns (records, ok)."""
    out, ok = [], True
    for i in range(len(WIDTH_ARMS) - 1):
        a, b = WIDTH_ARMS[i], WIDTH_ARMS[i + 1]
        d = [x["r2_mul"] - y["r2_mul"]
             for x, y in zip(res_by_arm[a], res_by_arm[b])]
        m, se = mean_se(d)
        mult = se_mult(m, se)
        # m < 0: wider width blinds less (violation)
        violated = (m < 0.0) and (mult >= Z_FAMILY)
        pos, _neg, _z = sign_split(d)
        out.append((a, b, m, se, mult, pos, violated))
        if violated:
            ok = False
    return ok, out


def p4_var_q(res_arm, res_base):
    """sd(q) stability at one width."""
    _d, m, se = paired(res_arm, res_base, "sd_q_all")
    mult = se_mult(m, se)
    return (mult < Z_FAMILY), m, se, mult


# sweep-level verdict
def sweep_verdict(p1_ok, p2_ok, p3_ok, width_records):
    """The only place a label short-circuits"""
    if not p1_ok:
        return "INCONCLUSIVE"
    if not p2_ok:
        return "NO ROOM"
    if not p3_ok:
        return "NOT A CURVE"
    clean = [w for w in width_records
             if w["passes_all"] and not w["provisional"]]
    if clean:
        return "INTERIOR POINT EXISTS"
    prov = [w for w in width_records if w["passes_all"]]
    if prov:
        return "INTERIOR POINT PROVISIONAL"
    return "NO INTERIOR"


# self-tests
def t_identity():
    """Imported names are the owning modules' objects."""
    assert QuantizedMM is BLIND.QuantizedMM
    assert FlatQuoteMM is PSMOD.FlatQuoteMM
    assert series_stats is PSMOD.series_stats
    assert build_features is SKMOD.build_features
    assert fit_score is SKMOD.fit_score
    assert ols is FILLMOD.ols
    assert mean_se is BLIND.mean_se
    assert se_mult is BLIND.se_mult
    assert paired is BLIND.paired
    assert require_finite is BLIND.require_finite
    assert DegenerateComparison is BLIND.DegenerateComparison
    assert TRAIN_FRAC == 0.70, \
        "TRAIN_FRAC moved off 0.70; every committed R2 assumed 0.70"
    assert BLIND.fit_score is fit_score, \
        ("phase7_blind is not holding the committed fit_score. A previous "
         "run_diag left the wrapper installed and every number after it is "
         "suspect.")
    assert SD_Q_REF == 0.04604 and MAX_ABS_Q_REF == 0.15129, \
        "the committed sd(q) / max|q| anchors moved"
    print("  t_identity              PASS  (11 objects is-identity, "
          "TRAIN_FRAC 0.70, fit_score unpatched)")


def t_widths_and_cap():
    """Widths and analytic cap re-derived from committed constants."""
    want_w = (0.034530, 0.046040, 0.057550, 0.069060, 0.080570)
    want_cap = (0.953125, 0.916667, 0.869792, 0.812500, 0.744792)
    for i, ratio in enumerate(W_OVER_SD):
        w = WIDTHS[i]
        assert abs(w - want_w[i]) < 1e-9, \
            "width %d is %.9f, expected %.6f" % (i, w, want_w[i])
        cap = analytic_cap(w)
        assert abs(cap - want_cap[i]) < 1e-6, \
            "cap at w/sd=%.2f is %.6f, expected %.6f" % (ratio, cap,
                                                         want_cap[i])
        # band inversion round-trips
        back = math.sqrt(12.0 * (1.0 - cap))
        assert abs(back - ratio) < 1e-9, \
            "cap inversion does not round-trip at w/sd=%.2f: %.9f" % (ratio,
                                                                      back)
    # committed widths, cross-check
    assert abs(analytic_cap(0.34 * SD_Q_REF) - 0.990367) < 1e-6
    assert abs(analytic_cap(2.50 * SD_Q_REF) - 0.479167) < 1e-6
    print("  t_widths_and_cap        PASS  (5 widths + 2 committed, cap "
          "inverts to w/sd = sqrt(12(1-cap)))")


def t_margin_arithmetic():
    """Leg 3 margin exceeds Z_FAMILY * projected SE."""
    projected = SE_R2_COMMITTED_8 * math.sqrt(8.0 / len(SEEDS))
    floor = Z_FAMILY * projected
    assert abs(projected - 0.05301) < 1e-4, \
        "projected SE moved: %.6f" % projected
    assert abs(floor - 0.13677) < 1e-4, "margin floor moved: %.6f" % floor
    assert MARGIN_R2 > floor, \
        ("MARGIN_R2 = %.4f is at or below the floor %.4f. Leg 3 could not be "
         "passed by an arm exactly as blinding as FLAT." % (MARGIN_R2, floor))
    # an equivalent arm at the projected SE passes
    m, se, ub, _n, _p, passed = leg3_noninferiority([0.0] * len(SEEDS))
    assert passed and abs(m) < 1e-12 and se == 0.0 and ub < MARGIN_R2
    # an arm at the margin fails
    _m2, _se2, _ub2, _n2, _p2, p2ok = leg3_noninferiority(
        [MARGIN_R2] * len(SEEDS))
    assert not p2ok, "an arm sitting exactly at the margin passed Leg 3"
    assert Z_FAMILY > Z_NOMINAL, "the Bonferroni Z must be the stricter one"
    print("  t_margin_arithmetic     PASS  (margin %.2f > floor %.5f; "
          "equivalent passes, at-margin fails)" % (MARGIN_R2, floor))


def t_nan_halts():
    """NaN reaching any scorer raises."""
    nan, inf = float("nan"), float("inf")
    n_raised = 0
    for bad in ([nan, 0.1], [0.1, inf], [nan], []):
        try:
            leg(bad, -1)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError("leg(%r) did not halt" % (bad,))
    for bad in ([nan, 0.1], [inf, 0.1], []):
        try:
            leg3_noninferiority(bad)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError("leg3(%r) did not halt" % (bad,))
    for bad in ([nan], [0.1, inf]):
        try:
            sign_split(bad)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError("sign_split(%r) did not halt" % (bad,))
    for args in ((0, 0), (5, 6), (5, -1)):
        try:
            sign_p(*args)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError("sign_p%r did not halt" % (args,))
    try:
        p1_identity([{"r2_mul": nan}])
    except DegenerateComparison:
        n_raised += 1
    else:
        raise AssertionError("p1_identity accepted a NaN")
    # scorers must not fire on legitimate values
    m, _se, mult, n_pass, p, passed = leg([-0.4] * 24, -1)
    assert passed and n_pass == 24 and mult == float("inf") and m < 0
    assert abs(p - 2.0 / (2 ** 24)) < 1e-15, p
    print("  t_nan_halts             PASS  (%d degenerate inputs all raised; "
          "sound inputs still score)" % n_raised)


def t_sign_statistics():
    """Sign statistics: 24/24 p ~ 1.2e-7, coin split p = 1.0, ties count."""
    assert abs(sign_p(24, 24) - 2.0 / (2 ** 24)) < 1e-15
    assert abs(sign_p(24, 24) - 1.1921e-07) < 1e-10, sign_p(24, 24)
    assert sign_p(24, 12) == 1.0
    assert sign_p(8, 8) == 2.0 / 256.0
    assert abs(sign_p(8, 8) - 0.0078125) < 1e-12
    pos, neg, zero = sign_split([1.0, -1.0, 0.0, 2.0])
    assert (pos, neg, zero) == (2, 1, 1)
    # ties count into n, not as successes
    _m, _se, _mult, n_pass, p, _ok = leg([-1.0] * 23 + [0.0], -1)
    assert n_pass == 23 and p > sign_p(24, 24), \
        "a tie must weaken the sign test, not be silently dropped"
    print("  t_sign_statistics       PASS  (24/24 p=%.4e, 12/24 p=1.0, ties "
          "counted into n)" % sign_p(24, 24))


def t_flag_fires():
    """P5 fires on the known divergence and not on sound fits."""
    # diverged case: seed 7 of QUANT-HIGH (ratio 15.54, magn 15.8)
    div_ratio, div_magn = 15.54, 2.9207 / 0.1849
    assert div_ratio >= FLAG_RATIO and div_magn > FLAG_MAGN, \
        "the measured divergence does not trip the thresholds"
    # sound cases (ratio 1.24 / 0.92 / 0.47)
    for ratio, magn in ((1.24, 0.85), (0.92, 0.59), (0.47, 0.89)):
        assert not (ratio >= FLAG_RATIO or magn > FLAG_MAGN), \
            "a sound fit (ratio %.2f, magn %.2f) tripped a flag" % (ratio,
                                                                    magn)
    # separation not marginal
    assert FLAG_RATIO > 1.24 * 2.0, "FLAG_RATIO sits too close to sound fits"
    assert FLAG_MAGN > 1.15 * 2.0, "FLAG_MAGN sits too close to sound fits"
    assert div_ratio > FLAG_RATIO * 2.0 and div_magn > FLAG_MAGN * 2.0, \
        "the diverged fit sits too close to the thresholds"
    print("  t_flag_fires            PASS  (diverged ratio %.2f / magn %.2f "
          "trips; sound 0.47-1.24 does not)" % (div_ratio, div_magn))


def t_verdict_labels():
    """Every verdict label is reachable."""
    clean = {"passes_all": True, "provisional": False}
    flagged = {"passes_all": True, "provisional": True}
    fails = {"passes_all": False, "provisional": False}
    cases = [
        ((False, True, True, [clean]), "INCONCLUSIVE"),
        ((True, False, True, [clean]), "NO ROOM"),
        ((True, True, False, [clean]), "NOT A CURVE"),
        ((True, True, True, [clean, fails]), "INTERIOR POINT EXISTS"),
        ((True, True, True, [flagged, fails]), "INTERIOR POINT PROVISIONAL"),
        ((True, True, True, [fails, fails]), "NO INTERIOR"),
        ((True, True, True, []), "NO INTERIOR"),
    ]
    seen = set()
    for args, want in cases:
        got = sweep_verdict(*args)
        assert got == want, "sweep_verdict%r returned %r, wanted %r" % (
            args, got, want)
        seen.add(got)
    assert len(seen) == 6, "only %d distinct labels reachable" % len(seen)
    assert sweep_verdict(False, False, False, [clean]) == "INCONCLUSIVE"
    assert sweep_verdict(True, False, False, [clean]) == "NO ROOM"
    # clean pass outranks provisional
    assert sweep_verdict(True, True, True,
                         [flagged, clean]) == "INTERIOR POINT EXISTS"
    print("  t_verdict_labels        PASS  (%d labels reachable, precondition "
          "order enforced)" % len(seen))


def t_preconditions():
    """P1-P4 on constructed inputs."""
    def rec(r2, rd=0.2, sdq=0.046):
        return {"r2_mul": r2, "r_direct": rd, "sd_q_all": sdq}

    base = [rec(1.0) for _ in range(8)]
    ok, m, _se = p1_identity(base)
    assert ok and abs(m - 1.0) < 1e-12
    ok, _m, _se = p1_identity([rec(0.99) for _ in range(8)])
    assert not ok, "P1 accepted a BASELINE that is not the identity"

    # P2: FLAT blinds and costs at Z_FAMILY
    flat_good = [rec(0.0 + 0.001 * i, rd=0.16 + 0.001 * i) for i in range(8)]
    ok, _bm, _bse, _bmu, _cm, _cse, _cmu = p2_endpoint(base, flat_good)
    assert ok, "P2 rejected a FLAT that both blinds and costs"
    flat_free = [rec(0.0 + 0.001 * i, rd=0.2) for i in range(8)]
    ok, _bm, _bse, _bmu, _cm, _cse, _cmu = p2_endpoint(base, flat_free)
    assert not ok, "P2 accepted a FLAT that costs nothing"

    # P3: blinding must not fall as w widens
    rising = {}
    for i, arm in enumerate(WIDTH_ARMS):
        rising[arm] = [rec(0.9 - 0.1 * i + 0.001 * s) for s in range(8)]
    ok, rows = p3_monotone(rising)
    assert ok and len(rows) == len(WIDTH_ARMS) - 1, rows
    falling = dict(rising)
    falling[WIDTH_ARMS[-1]] = [rec(0.95 + 0.001 * s) for s in range(8)]
    ok, rows = p3_monotone(falling)
    assert not ok, "P3 accepted blinding that COLLAPSED at the widest width"
    assert rows[-1][6], "P3 blamed the wrong adjacent pair"

    # P4: a width that moves sd(q) voids its cap
    same = [rec(0.5, sdq=0.046 + 0.0001 * s) for s in range(8)]
    ok, _m, _se, _mu = p4_var_q(same, [rec(1.0, sdq=0.046 + 0.0001 * s)
                                       for s in range(8)])
    assert ok, "P4 voided a cap when sd(q) did not move at all"
    moved = [rec(0.5, sdq=0.056 + 0.0001 * s) for s in range(8)]
    ok, _m, _se, mult = p4_var_q(moved, [rec(1.0, sdq=0.046 + 0.0001 * s)
                                         for s in range(8)])
    assert not ok and mult >= Z_FAMILY, \
        "P4 did not void the cap on a 10x-SE shift in sd(q)"
    print("  t_preconditions         PASS  (P1 P2 P3 P4 each accept and each "
          "reject on constructed input)")


def t_capture_is_transparent():
    """Diagnostic wrapper returns bit-identical results."""
    mm = make_width_mm(WIDTHS[0])
    a = BLIND.run(0, width_label(W_OVER_SD[0]), mm_override=mm)
    mm2 = make_width_mm(WIDTHS[0])
    b = run_diag(0, width_label(W_OVER_SD[0]), mm=mm2)
    for f in ("r2_mul", "r2_norm", "r_direct", "r_ar1", "phi", "sd_q",
              "sd_q_all", "max_abs_q", "pnl", "n_mm_fills", "n_views",
              "n_kept", "n_prints"):
        assert a[f] == b[f], \
            ("THE DIAGNOSTIC WRAPPER CHANGED %s: %.17g without, %.17g with. "
             "Every number in this file would be the wrapper's." % (f, a[f],
                                                                    b[f]))
    assert BLIND.fit_score is fit_score, \
        "run_diag did not restore phase7_blind.fit_score"
    assert b["n_mm_fills"] > 0, "the transparency check ran on ZERO fills"
    assert 0.0 <= b["diag_outfrac"] <= 1.0
    assert b["diag_pivot_std"] > 0.0
    print("  t_capture_is_transparent PASS (13 fields bit-identical; pivot "
          "ratio %.4e, out-of-range %.3f)"
          % (b["diag_pivot_std"], b["diag_outfrac"]))


def t_determinism_and_nonempty():
    """Deterministic and non-empty."""
    w = WIDTHS[2]
    a = run_diag(1, width_label(W_OVER_SD[2]), mm=make_width_mm(w))
    b = run_diag(1, width_label(W_OVER_SD[2]), mm=make_width_mm(w))
    for f in ("r2_mul", "r2_norm", "r_direct", "sd_q_all", "max_abs_q", "pnl",
              "diag_ratio", "diag_magn", "diag_pivot_std"):
        assert a[f] == b[f], \
            "NOT DETERMINISTIC on %s: %.17g vs %.17g" % (f, a[f], b[f])
    assert a["n_mm_fills"] > 0, \
        "the maker got ZERO fills; every downstream check is vacuous"
    assert a["n_views"] > 0 and a["n_kept"] > 0 and a["n_prints"] > 0, \
        ("empty run: views=%.0f kept=%.0f prints=%.0f"
         % (a["n_views"], a["n_kept"], a["n_prints"]))
    assert SELFTEST_SHORT == int(T), \
        "SELFTEST_SHORT and T diverged; the maker's clock would disagree"
    print("  t_determinism_and_nonempty PASS (%.0f fills, %.0f views, %.0f "
          "kept, %.0f prints; identical twice)"
          % (a["n_mm_fills"], a["n_views"], a["n_kept"], a["n_prints"]))


def t_no_default_moved():
    """No committed default mutated by import."""
    assert DEFAULT_GAMMA == 1e-6, "DEFAULT_GAMMA moved"
    from market_maker import DEFAULT_QUOTE_SIZE, DEFAULT_TICK
    assert DEFAULT_QUOTE_SIZE == 0.02, "DEFAULT_QUOTE_SIZE moved"
    assert DEFAULT_TICK == 0.01, "DEFAULT_TICK moved"
    assert MarketMaker.total_spread is QuantizedMM.total_spread, \
        ("QuantizedMM must NOT override total_spread -- the arms have to quote "
         "the same width or they win different flow")
    assert MarketMaker.total_spread is FlatQuoteMM.total_spread, \
        "FlatQuoteMM must NOT override total_spread either"
    assert BLIND.SEEDS == list(range(8)), \
        "phase7_blind's own seed list was mutated by this file"
    assert len(SEEDS) == 24, "this sweep's seed count moved off 24"
    print("  t_no_default_moved      PASS  (3 defaults intact; neither "
          "defense overrides total_spread; Arm A's seeds untouched)")


def _fake_res(degenerate):
    """Constructed result set shaped like run_diag's output."""
    n = len(SEEDS)
    res = {BASE: [{"r2_mul": 1.0, "r_direct": 0.20 + 0.0005 * s,
                   "sd_q_all": 0.0460 + 0.00001 * s, "diag_ratio": 0.9,
                   "diag_magn": 0.8, "diag_outfrac": 0.10,
                   "diag_pivot_std": 1.41e2, "flagged": False}
                  for s in range(n)],
           FLAT: [{"r2_mul": 0.001 * s, "r_direct": 0.16 + 0.0005 * s,
                   "sd_q_all": 0.0496 + 0.00001 * s, "diag_ratio": 1.1,
                   "diag_magn": 0.9, "diag_outfrac": 0.20,
                   "diag_pivot_std": 2.05e2, "flagged": False}
                  for s in range(n)]}
    for i, arm in enumerate(WIDTH_ARMS):
        rows = []
        for s in range(n):
            r2 = 0.90 - 0.18 * i + 0.002 * s
            sdq = 0.0460 + 0.00001 * s
            if degenerate and i == len(WIDTH_ARMS) - 1:
                r2 = 0.95 + 0.002 * s
                sdq = 0.0560 + 0.00001 * s
            flag = bool(degenerate and i == 0 and s == 0)
            rows.append({"r2_mul": r2,
                         "r_direct": 0.18 + 0.0005 * s,
                         "sd_q_all": sdq,
                         "diag_ratio": 15.54 if flag else 1.0 + 0.01 * s,
                         "diag_magn": 15.80 if flag else 0.9,
                         "diag_outfrac": 0.95 if flag else 0.15,
                         "diag_pivot_std": 2.24e3 if flag else 1.06e2,
                         "flagged": flag})
        res[arm] = rows
    return res


def t_output_smoke():
    """Every output format string exercised on constructed data."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_header()
        print_preregistration()
        for degenerate in (False, True):
            res = _fake_res(degenerate)
            for arm in ARMS:
                print_arm_timing(arm, 12.3, res[arm])
            print_section_a(res)
            flags = print_section_b(res)
            recs = [print_width_block(i, res)
                    for i in range(len(WIDTH_ARMS))]
            print_verdict(sweep_verdict(*(flags + (recs,))), recs)
        # every verdict label's print branch
        passing = [{"arm": "Q-1.00", "passes_all": True, "provisional": False,
                    "cap_valid": False}]
        for label in ("INTERIOR POINT EXISTS", "INTERIOR POINT PROVISIONAL",
                      "NO INTERIOR", "NO ROOM", "NOT A CURVE",
                      "INCONCLUSIVE"):
            print_verdict(label, passing)
    out = buf.getvalue()
    for marker in ("THE WIDTH SWEEP", "LEG 3  NOT WORSE THAN FLAT",
                   "SECTION A", "SECTION B", "P1 IDENTITY", "P3 MONOTONICITY",
                   "LEG 1 BLINDS", "LEG 2 CHEAPER/FLAT", "LEG 3 NON-INFERIOR",
                   "FLAGGED", "PROVISIONAL", "INVALID, sd(q) moved",
                   "VIOLATION", "NOT A CURVE", "passing width",
                   "analytic cap INVALID here"):
        assert marker in out, \
            "the output path never printed %r; a branch is unexercised" % (
                marker,)
    n_lines = len(out.splitlines())
    assert n_lines > 400, "the smoke test only produced %d lines" % n_lines
    print("  t_output_smoke          PASS  (%d lines rendered, %d markers, "
          "no format raised)" % (n_lines, 16))


def selftest():
    print("=" * 74)
    print("phase7_width.py self-test")
    print("=" * 74)
    print("  %d seeds, %d widths, Z_FAMILY = %.2f (Bonferroni over %d), "
          "margin %.2f R2" % (len(SEEDS), len(WIDTHS), Z_FAMILY,
                              N_COMPARISONS, MARGIN_R2))
    for i, ratio in enumerate(W_OVER_SD):
        print("    w/sd = %.2f  ->  W_Q = %.6f BTC   analytic cap %.4f"
              % (ratio, WIDTHS[i], analytic_cap(WIDTHS[i])))
    print("")
    t0 = time.time()
    t_identity()
    t_widths_and_cap()
    t_margin_arithmetic()
    t_nan_halts()
    t_sign_statistics()
    t_flag_fires()
    t_verdict_labels()
    t_preconditions()
    t_output_smoke()
    t_capture_is_transparent()
    t_determinism_and_nonempty()
    t_no_default_moved()
    print("")
    print("  all self-tests PASS in %.0fs" % (time.time() - t0))
    print("=" * 74)


# main
def print_header():
    bar = "=" * 126
    print(bar)
    print("Phase 7: THE WIDTH SWEEP. Is there an interior point?")
    print(bar)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (BLIND.LAM, BLIND.P_MARKET, BLIND.LIFE, BLIND.DISP))
    print("k=%.5f. gamma=%.2e. quote_size=%.3f BTC = DEFAULT_QUOTE_SIZE, "
          "untouched." % (K, GAMMA_CAP, QUOTE_SIZE))
    print("%d seeds x %.0fs x %d arms. Forward %d/%d time split."
          % (len(SEEDS), T, len(ARMS), int(100 * TRAIN_FRAC),
             int(100 * (1 - TRAIN_FRAC))))
    print("")
    print("Widths from the committed sd(q) = %.5f BTC:" % SD_Q_REF)
    for i, ratio in enumerate(W_OVER_SD):
        print("  %-6s  w/sd = %.2f  ->  W_Q = %.6f BTC   analytic cap %.4f"
              % (WIDTH_ARMS[i], ratio, WIDTHS[i], analytic_cap(WIDTHS[i])))
    print("  Band from the cap: w/sd = sqrt(12*(1-cap)); caps")
    print("  0.70-0.95 at w/sd in [0.775, 1.897]. Arm A's widths are")
    print("  outside it: cap %.4f (w/sd 0.34) and cap %.4f (w/sd 2.50);"
          % (analytic_cap(0.34 * SD_Q_REF), analytic_cap(2.50 * SD_Q_REF)))
    print("  not re-run or corrected here.")
    print("")


def print_preregistration():
    print("Pre-registered before any arm ran.")
    print("  Z_FAMILY = %.2f. Bonferroni over %d widths: 0.05/%d = "
          "0.01 two-sided. Nominal Z = %.2f printed"
          % (Z_FAMILY, N_COMPARISONS, N_COMPARISONS, Z_NOMINAL))
    print("  for comparison with arm A only.")
    print("")
    print("  LEG 1  blinds at all          R2(arm) - R2(BASELINE) < 0 at "
          "Z_FAMILY.")
    print("  LEG 2  cheaper than FLAT      R_direct(arm) - R_direct(FLAT) > 0 "
          "at Z_FAMILY.")
    print("  LEG 3  NOT WORSE THAN FLAT    non-inferiority: the one-sided "
          "upper bound")
    print("                                mean(d3) + %.2f*SE < %.2f, where "
          "d3 = R2(arm) - R2(FLAT)." % (Z_FAMILY, MARGIN_R2))
    print("                                Margin %.2f: an equivalent "
          "arm can only" % MARGIN_R2)
    print("                                pass if the margin exceeds "
          "Z_FAMILY*SE = %.2f*%.5f = %.5f."
          % (Z_FAMILY, SE_R2_COMMITTED_8 * math.sqrt(8.0 / len(SEEDS)),
             Z_FAMILY * SE_R2_COMMITTED_8 * math.sqrt(8.0 / len(SEEDS))))
    print("                                If the realised SE is "
          "larger, LEG 3 is unpassable:")
    print("                                reported as a power limit; "
          "margin not raised.")
    print("")
    print("  Every leg prints for every width; only the final label "
          "short-circuits")
    print("  (arm A returned NO INTERIOR without consulting the blind leg;")
    print("  not repeated here).")
    print("")
    print("  Headline = paired per-seed mean +/- SE. Companion = "
          "paired sign count (bounded;")
    print("  immune to a diverged fit's magnitude). Sign count near-vacuous")
    print("  against BASELINE (R2 = 1.000000, so every difference")
    print("  is negative); informative only on Legs 2 and 3 and P3.")
    print("")
    print("  P5 flags: sd(pred)/sd(y_test) >= %.1f, or max|pred|/max|q| > "
          "%.1f. Measured separation:" % (FLAG_RATIO, FLAG_MAGN))
    print("  sound fits 0.47-1.24 and <=1.15; the one known divergence "
          "15.54 and 15.8. Thresholds")
    print("  set by inspecting that divergence; they change")
    print("  labels only, never pooling (flagged fits stay in every mean and SE).")
    print("")
    print("  Power: removing skew costs +0.0337 +/- 0.0082 in "
          "R_direct; a partial defense costs")
    print("  a fraction of that: NO INTERIOR = not distinguishable at this power.")
    print("")


def print_arm_timing(arm, secs, res_arm):
    r2m, r2s = mean_se([r["r2_mul"] for r in res_arm])
    rdm, rds = mean_se([r["r_direct"] for r in res_arm])
    nflag = sum(1 for r in res_arm if r["flagged"])
    print("  %-9s measured in %5.0fs   R2(mul)=%12.6f+/-%.6f   "
          "R_direct=%.4f+/-%.4f   FLAGGED %d/%d"
          % (arm, secs, r2m, r2s, rdm, rds, nflag, len(res_arm)))


def print_section_a(res):
    """Per-fit diagnostics, one line per seed."""
    bar = "-" * 126
    print("=" * 126)
    print("SECTION A: per-fit diagnostics (P5). Nothing dropped.")
    print("=" * 126)
    print("  arm         seed        R2(mul)   sd(pred)/sd(y)   "
          "max|pred|/max|q|   out-of-train-range   std pivot ratio   flag")
    for arm in ARMS:
        for i, s in enumerate(SEEDS):
            r = res[arm][i]
            print("  %-9s %5d  %13.6f   %14.2f   %16.2f   %18.3f   "
                  "%15.4e   %s"
                  % (arm, s, r["r2_mul"], r["diag_ratio"], r["diag_magn"],
                     r["diag_outfrac"], r["diag_pivot_std"],
                     "FLAGGED" if r["flagged"] else ""))
        print(bar)
    print("  Standardized pivot ratio: reported only; nothing fitted on")
    print("  standardized columns (rescaling preserves the estimator")
    print("  in exact arithmetic;")
    print("  ridge would not). Benign: 1e2-1e3.")
    print("")


def print_section_b(res):
    """Preconditions P1-P3"""
    print("=" * 126)
    print("SECTION B; preconditions")
    print("=" * 126)
    p1_ok, p1_m, p1_se = p1_identity(res[BASE])
    print("  P1 IDENTITY          BASELINE R2 = %.9f +/- %.9f   target "
          "%.6f +/- %.0e   %s"
          % (p1_m, p1_se, P1_TARGET, P1_TOL, "HELD" if p1_ok else "FAILED"))

    p2_ok, bm, bse, bmu, cm, cse, cmu = p2_endpoint(res[BASE], res[FLAT])
    print("  P2 endpoint          FLAT blinding %+.6f +/- %.6f  (%.2f SE)   "
          "FLAT cost %+.6f +/- %.6f  (%.2f SE)   %s"
          % (bm, bse, bmu, cm, cse, cmu, "HELD" if p2_ok else "FAILED"))

    p3_ok, p3_rows = p3_monotone(res)
    print("  P3 MONOTONICITY      blinding must not fall as w widens "
          "(paired, %d adjacent pairs):" % len(p3_rows))
    for a, b, m, se, mult, pos, viol in p3_rows:
        print("       %-6s -> %-6s   blinding rises by %+.6f +/- %.6f  "
              "(%.2f SE)   %2d/%d seeds rise   %s"
              % (a, b, m, se, mult, pos, len(SEEDS),
                 "VIOLATION" if viol else "ok"))
    print("       overall: %s"
          % ("HELD" if p3_ok else "FAILED -> NOT A CURVE"))
    print("")
    return p1_ok, p2_ok, p3_ok


def print_width_block(i, res):
    """One width: P4 and all three legs. Returns its record."""
    arm = WIDTH_ARMS[i]
    w = WIDTHS[i]
    cap = analytic_cap(w)
    p4_ok, v_m, v_se, v_mult = p4_var_q(res[arm], res[BASE])
    nflag = sum(1 for r in res[arm] if r["flagged"])
    r2m, r2s = mean_se([r["r2_mul"] for r in res[arm]])

    d1 = [x["r2_mul"] - y["r2_mul"] for x, y in zip(res[arm], res[BASE])]
    d2 = [x["r_direct"] - y["r_direct"] for x, y in zip(res[arm], res[FLAT])]
    d3 = [x["r2_mul"] - y["r2_mul"] for x, y in zip(res[arm], res[FLAT])]
    l1m, l1se, l1mu, l1n, l1p, l1ok = leg(d1, -1)
    l2m, l2se, l2mu, l2n, l2p, l2ok = leg(d2, +1)
    l3m, l3se, l3ub, l3n, l3p, l3ok = leg3_noninferiority(d3)

    print("  %-6s  w/sd %.2f   W_Q %.6f BTC   analytic cap %.4f %s"
          % (arm, W_OVER_SD[i], w, cap,
             "" if p4_ok else "<-- INVALID, sd(q) moved"))
    print("          measured R2(mul) %+.6f +/- %.6f      FLAGGED fits "
          "%d/%d %s"
          % (r2m, r2s, nflag, len(SEEDS), "-> PROVISIONAL" if nflag else ""))
    print("          P4 sd(q) vs BASELINE  %+.6f +/- %.6f  (%.2f SE)   %s"
          % (v_m, v_se, v_mult,
             "stable, cap valid" if p4_ok
             else "moved, cap not compared against"))
    print("          LEG 1 BLINDS        d = %+12.6f +/- %.6f  "
          "(%5.2f SE)   sign %2d/%d neg  p=%.2e   %s"
          % (l1m, l1se, l1mu, l1n, len(SEEDS), l1p,
             "PASS" if l1ok else "fail"))
    print("          LEG 2 CHEAPER/FLAT  d = %+12.6f +/- %.6f  "
          "(%5.2f SE)   sign %2d/%d pos  p=%.2e   %s"
          % (l2m, l2se, l2mu, l2n, len(SEEDS), l2p,
             "PASS" if l2ok else "fail"))
    print("          LEG 3 NON-INFERIOR  d = %+12.6f +/- %.6f  "
          "upper %+.6f vs margin %.2f   sign %2d/%d below  p=%.2e   %s"
          % (l3m, l3se, l3ub, MARGIN_R2, l3n, len(SEEDS), l3p,
             "PASS" if l3ok else "fail"))
    passes_all = l1ok and l2ok and l3ok
    print("          all three legs: %s%s"
          % ("PASS" if passes_all else "NOT PASSED",
             "  (PROVISIONAL; sign counts carry it)"
             if (passes_all and nflag) else ""))
    print("-" * 126)
    return {"arm": arm, "passes_all": passes_all,
            "provisional": nflag > 0, "cap_valid": p4_ok}


def print_verdict(v, records):
    print("=" * 126)
    print("Verdict")
    print("=" * 126)
    print("  %s" % v)
    print("")
    if v == "INTERIOR POINT EXISTS" or v == "INTERIOR POINT PROVISIONAL":
        for r in records:
            if r["passes_all"]:
                print("    passing width: %s%s"
                      % (r["arm"], "" if r["cap_valid"]
                         else "  (analytic cap INVALID here)"))
    print("  NO INTERIOR = not distinguishable at this power; not")
    print("  proof that only the endpoints exist.")
    print("  Arm A's two widths were not re-run or corrected")
    print("  here; phase7_blind_results.txt stands.")
    print("=" * 126)


def main():
    print_header()
    print_preregistration()

    res = {}
    t_all = time.time()
    for arm in ARMS:
        t0 = time.time()
        if arm == BASE or arm == FLAT:
            res[arm] = [run_diag(s, arm) for s in SEEDS]
        else:
            w = WIDTHS[WIDTH_ARMS.index(arm)]
            res[arm] = [run_diag(s, arm, mm=make_width_mm(w)) for s in SEEDS]
        print_arm_timing(arm, time.time() - t0, res[arm])
    print("  total %.0fs" % (time.time() - t_all))
    print("")

    print_section_a(res)
    p1_ok, p2_ok, p3_ok = print_section_b(res)

    print("=" * 126)
    print("Section C: the three legs (all printed for every width).")
    print("=" * 126)
    records = [print_width_block(i, res) for i in range(len(WIDTH_ARMS))]
    print("")

    print_verdict(sweep_verdict(p1_ok, p2_ok, p3_ok, records), records)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        try:
            main()
        except DegenerateComparison as exc:
            print("")
            print("=" * 126)
            print("Verdict")
            print("=" * 126)
            print("  INCONCLUSIVE: a decision rule got a value it")
            print("  cannot compare. Exception:")
            print("")
            print("    %s" % exc)
            print("")
            print("  Pre-registered handling of a non-finite")
            print("  statistic: halt and print; never report")
            print("  no VIOLATION found.")
            print("=" * 126)
            sys.exit(1)
