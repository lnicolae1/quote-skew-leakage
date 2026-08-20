#!/usr/bin/env python3
# phase7_nobs.py: Phase 7 arm B: the smoothing axis (recovery vs destruction)
# - JITTER adds noise to the reservation price: y = f(q) + eps; averaging should recover the signal
# - QUANT buckets inventory: y = f(bucket(q)), many-to-one; averaging should not recover it
# - attacker smooths every skew-derived column with a causal trailing mean over M observations
#   (M = 1, 3, 10, 30, 100, 300; M = 1 must reproduce the unsmoothed fit exactly)
# - arms: BASELINE, JITTER (sigma_eps = one time-averaged sd of skew), QUANT (W_Q = arm A's QUANT-LOW width)
# - 8 seeds x 86400 s; all comparisons paired per seed, Z_SE = 2.0
# - pre-registered:
#   Q1a: R2(M=30) - R2(M=1) for JITTER > 0 at >= 2.0 SE
#   Q1b: JITTER R2(M=30) within 2.0 SE of BASELINE at the same M (REFUTED if >= 2.0 SE below); Q1 HELD iff both
#   Q2: QUANT R2(M=300) - R2(M=1) flat within 2.0 SE -> HELD; REFUTED (RISES) or REFUTED (FALLS) otherwise
#   Q3 (descriptive): TURNED OVER if JITTER R2(M=300) < R2(M=100) by >= 2.0 SE
# - NaN inputs to any scorer raise

import math
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import phase7_blind as BLIND
import sniffer_skew as SKMOD

from clipped_mm_check import _resid_best, MM_ID
from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA, SECONDS_PER_YEAR
from phase7_blind import (DegenerateComparison, QuantizedMM, W_Q_LOW, mean_se,
                          require_finite, se_mult)
from sniffer_fill_only import PublicPrint, TRAIN_FRAC
from sniffer_skew import build_features, fit_score, TAU_FLOOR_FRAC

# is-identity
assert build_features is SKMOD.build_features, \
    "build_features must be sniffer_skew's own object, not a copy"
assert fit_score is SKMOD.fit_score, \
    "fit_score must be sniffer_skew's own object, not a copy"
assert QuantizedMM is BLIND.QuantizedMM, (
    "QuantizedMM must be phase7_blind's own class, so both arms test the "
    "same defense")
assert require_finite is BLIND.require_finite, (
    "require_finite must be phase7_blind's own object so both files halt on "
    "degenerate comparisons by the SAME rule")
assert W_Q_LOW == BLIND.W_Q_LOW, "W_Q_LOW must be Arm A's own width"
assert issubclass(QuantizedMM, MarketMaker), \
    "QuantizedMM must subclass the committed MarketMaker"

# constants
T = 86400.0
N_SEC = int(T)
SEEDS = list(range(8))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
QUOTE_SIZE = 0.020
GAMMA_CAP = 9.4e-6
SD_Q_REF = 0.04604
REF_MID = 62000.0
SIGMA_REL = 0.3721

# sigma_eps = one time-averaged sd of the maker's skew (from committed constants)
TAU0 = T / SECONDS_PER_YEAR
MEAN_TAU = 0.5 * TAU0
SIG_ABS = SIGMA_REL * REF_MID
SIGMA_EPS_MULT = 1.0
SD_SKEW_REF = SD_Q_REF * GAMMA_CAP * SIG_ABS * SIG_ABS * MEAN_TAU
SIGMA_EPS = SIGMA_EPS_MULT * SD_SKEW_REF

JITTER_SEED_BASE = 90000
                                   # own RNG stream, uncorrelated with the world's

# smoothing axis
MA_WINDOWS = [1, 3, 10, 30, 100, 300]
M_CONTROL = 1
M_RISE = 30
M_TOP = 300
M_TURN = 100

# predicted JITTER R2 from sqrt(M) noise reduction (printed only; no rule uses them)
PRED = {1: 0.500, 30: 0.968, 300: 0.997}

SKEW_PREFIX = "skew_"
Z_SE = 2.0

ARMS = ("BASELINE", "JITTER", "QUANT")

SELFTEST_SHORT = 86400

assert M_CONTROL in MA_WINDOWS and M_RISE in MA_WINDOWS, "scored M missing"
assert M_TOP in MA_WINDOWS and M_TURN in MA_WINDOWS, "scored M missing"
assert MA_WINDOWS == sorted(MA_WINDOWS), "MA_WINDOWS must be ascending"
assert set(PRED) <= set(MA_WINDOWS), "a prediction names an unswept M"


# JITTER defense
class JitteredMM(MarketMaker):
    """AS maker with mean-zero Gaussian noise on the reservation price; quoted width unchanged."""

    def __init__(self, sigma_eps, jitter_seed, **kw):
        MarketMaker.__init__(self, **kw)
        if sigma_eps < 0:
            raise ValueError("sigma_eps must be non-negative, got %r"
                             % (sigma_eps,))
        self.sigma_eps = sigma_eps
        self.rng = random.Random(jitter_seed)
        self.n_draws = 0

    def reservation_price(self, mid, t):
        r = MarketMaker.reservation_price(self, mid, t)
        self.n_draws += 1
        return r + self.rng.gauss(0.0, self.sigma_eps)


def make_mm(arm, seed, gamma=GAMMA_CAP):
    if arm == "BASELINE":
        return MarketMaker(horizon=T, k=K, gamma=gamma, quote_size=QUOTE_SIZE)
    if arm == "JITTER":
        return JitteredMM(SIGMA_EPS, JITTER_SEED_BASE + seed, horizon=T, k=K,
                          gamma=gamma, quote_size=QUOTE_SIZE)
    if arm == "QUANT":
        return QuantizedMM(W_Q_LOW, horizon=T, k=K, gamma=gamma,
                           quote_size=QUOTE_SIZE)
    raise ValueError("unknown arm %r" % (arm,))


# smoothing attacker
def trailing_mean(col, m):
    """Causal trailing mean over m observations (partial windows at the start)."""
    if m < 1:
        raise ValueError("window must be >= 1, got %r" % (m,))
    if m == 1:
        return list(col)
    out = [0.0] * len(col)
    run = 0.0
    for i, v in enumerate(col):
        run += v
        if i >= m:
            run -= col[i - m]
        out[i] = run / (m if i >= m - 1 else i + 1)
    return out


def smooth_skew_columns(rows, names, m):
    """Rows with every skew-derived column replaced by its trailing mean at window m."""
    if m == 1:
        return [list(r) for r in rows]
    targets = [i for i, nm in enumerate(names) if nm.startswith(SKEW_PREFIX)]
    if not targets:
        raise DegenerateComparison(
            "no column name starts with %r, so smoothing would be a no-op "
            "and the whole M axis would be flat by construction: %r"
            % (SKEW_PREFIX, names))
    out = [list(r) for r in rows]
    for c in targets:
        sm = trailing_mean([r[c] for r in rows], m)
        for i, v in enumerate(sm):
            out[i][c] = v
    return out


# simulation loop
def run(seed, arm, gamma=GAMMA_CAP, horizon=T, mm_override=None):
    """One arm, one seed, then the M sweep on that run's samples (loop transcribed from sniffer_skew.run)."""
    n_sec = int(horizon)
    _p, vf, eng, noise, informed = make_world(seed, horizon, LAM, P_MARKET,
                                              LIFE, DISP, "join")
    mm = mm_override if mm_override is not None else make_mm(arm, seed, gamma)

    views, prints, q_series = [], [], []
    n_mm_fills = 0

    t = 0.0
    while t < horizon:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                for f in fills:
                    prints.append(PublicPrint(t, f.price, f.size, r.side))
                    if f.counterparty_id == MM_ID:
                        n_mm_fills += 1
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue

        mb = ma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mb = o.price if mb is None else max(mb, o.price)
            else:
                ma = o.price if ma is None else min(ma, o.price)
        if mb is None or ma is None:
            continue
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is None or ra is None or ra <= rb:
            continue
        views.append(SKMOD.SkewView(t, mb, ma, 0.5 * (bb + ba),
                                    0.5 * (rb + ra),
                                    mm.time_remaining_years(t)))
        q_series.append(mm.q)

    # sniffer sees only views + prints from here on
    rows, names, keep = build_features(views, prints, n_sec)
    y = q_series
    n = min(len(rows), len(y))
    rows, y, keep = rows[:n], y[:n], keep[:n]
    n_kept_total = sum(1 for x in keep if x)

    mul_cols = [i for i, nm in enumerate(names) if nm != "skew_norm_fullmid"]
    c = {nm: i for i, nm in enumerate(names)}

    # unsmoothed fit: T-mirror anchor; M=1 must reproduce it
    r2_unsmoothed, _ru = fit_score(rows, y, keep, mul_cols)
    r2_norm_full, _rn = fit_score(rows, y, keep, [c["const"], c["skew_norm"]])

    # every M fitted on the full kept sample
    ma_curve = []
    for m in MA_WINDOWS:
        sm_rows = smooth_skew_columns(rows, names, m)
        r2, _r = fit_score(sm_rows, y, keep, mul_cols)
        ma_curve.append((m, r2))

    assert ma_curve[0][0] == M_CONTROL, ma_curve[0]
    assert ma_curve[0][1] == r2_unsmoothed, (
        "M=1 gave R2 %.17g but the unsmoothed fit gave %.17g. A trailing "
        "mean over ONE observation must be the identity, so these must agree "
        "bit for bit." % (ma_curve[0][1], r2_unsmoothed))

    return {
        "ma_curve": ma_curve,
        "r2_unsmoothed": r2_unsmoothed,
        "r2_norm_full": r2_norm_full,
        "n_kept_total": float(n_kept_total),
        "n_mm_fills": float(n_mm_fills),
        "n_views": float(len(views)),
        "n_prints": float(len(prints)),
        "sd_q": statistics.stdev(y) if len(y) > 1 else 0.0,
        "n_draws": float(getattr(mm, "n_draws", 0)),
    }


# curve bookkeeping
def curve_at(p, m):
    """That seed's R2 at window m; raises if missing."""
    for mm_, r2 in p["ma_curve"]:
        if mm_ == m:
            return r2
    raise DegenerateComparison(
        "no M=%r in this seed's curve %r" % (m, [w for w, _ in p["ma_curve"]]))


def curve_values(per_seed, m):
    return [curve_at(p, m) for p in per_seed]


def check_windows(per_seed, label):
    """All seeds must have swept the same windows."""
    ref = [w for w, _ in per_seed[0]["ma_curve"]]
    for i, p in enumerate(per_seed[1:], start=1):
        got = [w for w, _ in p["ma_curve"]]
        if got != ref:
            raise DegenerateComparison(
                "%s seed %d swept %r against seed 0's %r"
                % (label, i, got, ref))
    return ref


def paired_between(per_seed, m_lo, m_hi):
    """Paired per-seed change from m_lo to m_hi."""
    d = [curve_at(p, m_hi) - curve_at(p, m_lo) for p in per_seed]
    m, se = mean_se(d)
    return m, se, se_mult(m, se)


def paired_arms(a, b, m):
    """Paired per-seed difference a - b at one window."""
    if len(a) != len(b):
        raise DegenerateComparison(
            "paired_arms got %d and %d seeds" % (len(a), len(b)))
    d = [curve_at(x, m) - curve_at(y, m) for x, y in zip(a, b)]
    mm_, se = mean_se(d)
    return mm_, se, se_mult(mm_, se)


# scoring
def score_q1(jit, base):
    """Q1: JITTER rises with M and approaches BASELINE at the same M. Returns (verdict, rise, approach)."""
    check_windows(jit, "JITTER")
    check_windows(base, "BASELINE")
    require_finite("Q1 JITTER curve", *curve_values(jit, M_CONTROL))
    require_finite("Q1 JITTER curve", *curve_values(jit, M_RISE))
    require_finite("Q1 BASELINE curve", *curve_values(base, M_RISE))

    rm, rse, rmult = paired_between(jit, M_CONTROL, M_RISE)
    rise_ok = (rm > 0.0) and (rmult >= Z_SE)

    am, ase, amult = paired_arms(jit, base, M_RISE)
    approach_ok = not ((am < 0.0) and (amult >= Z_SE))

    if rise_ok and approach_ok:
        v = "HELD"
    elif not rise_ok and not approach_ok:
        v = "REFUTED (no rise, and does not approach BASELINE)"
    elif not rise_ok:
        v = "REFUTED (no rise)"
    else:
        v = "REFUTED (rises, but stays below BASELINE)"
    return v, (rm, rse, rmult), (am, ase, amult)


def score_q2(quant):
    """Q2: QUANT flat within Z_SE from M_CONTROL to M_TOP, paired per seed."""
    check_windows(quant, "QUANT")
    require_finite("Q2 QUANT curve", *curve_values(quant, M_CONTROL))
    require_finite("Q2 QUANT curve", *curve_values(quant, M_TOP))

    m, se, mult = paired_between(quant, M_CONTROL, M_TOP)
    if mult < Z_SE:
        return "HELD", m, se, mult
    if m > 0.0:
        return "REFUTED (RISES)", m, se, mult
    return "REFUTED (FALLS)", m, se, mult


def score_q3(jit):
    """Q3: turnover of the JITTER climb (descriptive)."""
    check_windows(jit, "JITTER")
    require_finite("Q3 JITTER curve", *curve_values(jit, M_TURN))
    require_finite("Q3 JITTER curve", *curve_values(jit, M_TOP))

    m, se, mult = paired_between(jit, M_TURN, M_TOP)
    if m < 0.0 and mult >= Z_SE:
        return "TURNED OVER", m, se, mult
    return "NO TURNOVER", m, se, mult


# self-tests
def t_identity():
    assert build_features is SKMOD.build_features
    assert fit_score is SKMOD.fit_score
    assert QuantizedMM is BLIND.QuantizedMM
    assert mean_se is BLIND.mean_se
    assert se_mult is BLIND.se_mult
    assert require_finite is BLIND.require_finite
    assert TAU_FLOOR_FRAC is SKMOD.TAU_FLOOR_FRAC
    assert TRAIN_FRAC == 0.70, \
        "TRAIN_FRAC moved off 0.70; every committed R2 assumed 0.70"
    assert SELFTEST_SHORT == int(T), (
        "SELFTEST_SHORT and T have diverged. make_mm builds every maker with "
        "horizon=T, so a self-test run at a different horizon would give the "
        "maker a clock that disagrees with its own loop.")
    print("  t_identity              PASS  (7 objects, all is-identity)")


def t_trailing_mean():
    """Smoother is causal, correct on partial windows, identity at m=1."""
    col = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert trailing_mean(col, 1) == col, trailing_mean(col, 1)
    got = trailing_mean(col, 2)
    want = [1.0, 1.5, 2.5, 3.5, 4.5]
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-12, (got, want)
    got = trailing_mean(col, 3)
    want = [1.0, 1.5, 2.0, 3.0, 4.0]
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-12, (got, want)
    # window longer than the data: all partial windows
    got = trailing_mean(col, 99)
    want = [1.0, 1.5, 2.0, 2.5, 3.0]
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-12, (got, want)
    # causality: a later value must not move an earlier output
    a = trailing_mean([1.0, 2.0, 3.0, 4.0], 3)
    b = trailing_mean([1.0, 2.0, 3.0, 999.0], 3)
    assert a[:3] == b[:3], (a, b)
    try:
        trailing_mean(col, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("window 0 must raise")
    print("  t_trailing_mean         PASS  (m=1 identity, partial windows, "
          "causality, m=0 raises)")


def t_ma_m1_is_unsmoothed():
    """m=1 rows bit-identical to build_features output."""
    names = ["const", "skew_norm", "skew_raw", "spread_width",
             "skew_norm_fullmid", "skew_vel_60", "fillvol_60"]
    rows = [[1.0, 0.1 * i, 0.2 * i, 0.3, 0.4 * i, 0.5 * i, 0.6 * i]
            for i in range(50)]
    got = smooth_skew_columns(rows, names, 1)
    assert got == rows, "m=1 must be the identity on the rows"
    assert got is not rows, "must return a COPY, not alias the caller's rows"
    assert got[0] is not rows[0], "rows must be copied, not aliased"
    print("  t_ma_m1_is_unsmoothed   PASS  (50 rows bit-identical at m=1, "
          "and copied not aliased)")


def t_ma_targets_only_skew():
    """Smoothing hits every skew-derived column and nothing else."""
    names = ["const", "skew_norm", "skew_raw", "spread_width",
             "skew_norm_fullmid", "skew_vel_60", "skew_vel_300",
             "fillvol_60", "fillvol_300"]
    rows = [[1.0, float(i), float(i), 7.0, float(i), float(i), float(i),
             float(i), float(i)] for i in range(20)]
    got = smooth_skew_columns(rows, names, 5)
    smoothed = [j for j, nm in enumerate(names)
                if any(got[k][j] != rows[k][j] for k in range(20))]
    want = [j for j, nm in enumerate(names) if nm.startswith(SKEW_PREFIX)]
    assert smoothed == want, (
        "smoothed columns %r but the skew-derived columns are %r"
        % ([names[j] for j in smoothed], [names[j] for j in want]))
    assert len(want) == 5, "expected 5 skew columns, got %d" % len(want)
    for k in range(20):
        assert got[k][0] == 1.0, "the constant column was smoothed"
        assert got[k][3] == 7.0, "spread_width was smoothed"
        assert got[k][7] == float(k), "fillvol_60 was smoothed"
    # no skew column must raise
    try:
        smooth_skew_columns([[1.0]], ["const"], 5)
    except DegenerateComparison:
        pass
    else:
        raise AssertionError(
            "a name set with no skew column must raise, not silently return "
            "an unsmoothed matrix that would pin the M axis flat")
    print("  t_ma_targets_only_skew  PASS  (5 skew columns smoothed, const / "
          "spread_width / fillvol untouched)")


def t_sigma_eps_arithmetic():
    """SIGMA_EPS re-derived from the MarketMaker's own methods."""
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA_CAP, quote_size=QUOTE_SIZE)
    mm.q = SD_Q_REF
    skew_t0 = abs(mm.inventory_skew(REF_MID, 0.0))
    want = 0.5 * skew_t0
    assert abs(SIGMA_EPS - want) < 1e-9, (
        "SIGMA_EPS = %.9f but one time-averaged sd of skew, computed from "
        "MarketMaker's own inventory_skew, is %.9f" % (SIGMA_EPS, want))
    assert 0.0 < SIGMA_EPS < 1.0, (
        "SIGMA_EPS = %.6f is outside any plausible range for a dollar "
        "displacement of a $62,000 quote centre" % SIGMA_EPS)
    print("  t_sigma_eps_arithmetic  PASS  (SIGMA_EPS = $%.6f = one "
          "time-averaged sd of skew)" % SIGMA_EPS)


def t_predicted_values():
    """Predicted R2 values re-derived: ceiling S^2/(S^2+N^2), noise variance / M."""
    for m, pred in sorted(PRED.items()):
        want = m / float(m + 1)
        assert abs(pred - want) < 0.0015, (
            "PRED[%d] = %.4f but M/(M+1) = %.6f. The header's predictions "
            "must be the arithmetic, not a guess." % (m, pred, want))
    assert SIGMA_EPS_MULT == 1.0, (
        "the M/(M+1) form assumes noise sd EQUALS signal sd at M=1, i.e. "
        "SIGMA_EPS_MULT = 1.0, but it is %r" % (SIGMA_EPS_MULT,))
    print("  t_predicted_values      PASS  (M/(M+1) at M=1, 30, 300 -> "
          "%.3f, %.3f, %.3f)" % (PRED[1], PRED[30], PRED[300]))


def _mk(vals, spread=0.0001):
    """Constructed per-seed curves for the scorer tests."""
    return [{"ma_curve": list(zip(MA_WINDOWS,
                                  [v + spread * s for v in vals]))}
            for s in range(6)]


def t_scoring_rules():
    """Every scorer on constructed curves."""
    rising = _mk([0.50, 0.60, 0.80, 0.95, 0.99, 0.997])
    flat = _mk([0.50, 0.50, 0.50, 0.50, 0.50, 0.50])
    falling = _mk([0.99, 0.90, 0.80, 0.70, 0.60, 0.50])
    turning = _mk([0.50, 0.60, 0.80, 0.95, 0.99, 0.80])
    near_one = _mk([0.98, 0.98, 0.99, 0.995, 0.996, 0.997])

    v, rise, appr = score_q1(rising, near_one)
    assert v == "REFUTED (rises, but stays below BASELINE)", (v, rise, appr)
    v, rise, appr = score_q1(rising, rising)
    assert v == "HELD", (v, rise, appr)
    v, rise, appr = score_q1(flat, near_one)
    assert v.startswith("REFUTED (no rise"), (v, rise, appr)

    v, m, se, mult = score_q2(flat)
    assert v == "HELD", (v, m, se, mult)
    v, m, se, mult = score_q2(rising)
    assert v == "REFUTED (RISES)", (v, m, se, mult)
    v, m, se, mult = score_q2(falling)
    assert v == "REFUTED (FALLS)", (v, m, se, mult)

    v, m, se, mult = score_q3(turning)
    assert v == "TURNED OVER", (v, m, se, mult)
    v, m, se, mult = score_q3(rising)
    assert v == "NO TURNOVER", (v, m, se, mult)

    # different windows across seeds must raise
    bad = _mk([0.5] * 6)
    bad[2]["ma_curve"] = bad[2]["ma_curve"][:-1]
    try:
        score_q2(bad)
    except DegenerateComparison:
        pass
    else:
        raise AssertionError("mismatched windows across seeds must raise")
    print("  t_scoring_rules         PASS  (9 constructed curves; mismatched "
          "windows raise)")


def t_nan_halts():
    """NaN reaching any scorer raises."""
    nan = float("nan")
    good = _mk([0.50, 0.60, 0.80, 0.95, 0.99, 0.997])
    n_raised = 0

    def poisoned(base_curves, m_bad):
        out = [{"ma_curve": list(p["ma_curve"])} for p in base_curves]
        for p in out:
            p["ma_curve"] = [(w, nan if w == m_bad else r)
                             for w, r in p["ma_curve"]]
        return out

    for m_bad in (M_CONTROL, M_RISE):
        try:
            score_q1(poisoned(good, m_bad), good)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError(
                "score_q1 returned a verdict with M=%d NaN" % m_bad)

    try:
        score_q1(good, poisoned(good, M_RISE))
    except DegenerateComparison:
        n_raised += 1
    else:
        raise AssertionError("score_q1 accepted a NaN BASELINE")

    for m_bad in (M_CONTROL, M_TOP):
        try:
            score_q2(poisoned(good, m_bad))
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError(
                "score_q2 returned a verdict with M=%d NaN -- this is the "
                "exact failure the n-ladder shipped" % m_bad)

    for m_bad in (M_TURN, M_TOP):
        try:
            score_q3(poisoned(good, m_bad))
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError(
                "score_q3 returned a verdict with M=%d NaN" % m_bad)

    # all-NaN curve must raise
    all_nan = _mk([nan] * 6, spread=0.0)
    for scorer, args in ((score_q2, (all_nan,)), (score_q3, (all_nan,))):
        try:
            scorer(*args)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError(
                "a curve that is NaN at EVERY window returned a verdict")

    # guards must not fire on a legitimate curve
    v, _, _, _ = score_q2(_mk([0.5] * 6))
    assert v == "HELD", v
    print("  t_nan_halts             PASS  (%d degenerate inputs all raised; "
          "clean curves still score)" % n_raised)


def t_no_default_moved():
    assert DEFAULT_GAMMA == 1e-6, "DEFAULT_GAMMA moved"
    from market_maker import DEFAULT_QUOTE_SIZE, DEFAULT_SIGMA
    assert DEFAULT_QUOTE_SIZE == 0.02, "DEFAULT_QUOTE_SIZE moved"
    assert DEFAULT_SIGMA == SIGMA_REL, (
        "DEFAULT_SIGMA is %.6f but SIGMA_EPS was derived from %.6f"
        % (DEFAULT_SIGMA, SIGMA_REL))
    assert MarketMaker.total_spread is JitteredMM.total_spread, (
        "JitteredMM must NOT override total_spread -- the arms have to quote "
        "the same width or they win different flow")
    assert MarketMaker.total_spread is QuantizedMM.total_spread, \
        "QuantizedMM must NOT override total_spread either"
    print("  t_no_default_moved      PASS  (3 defaults intact; neither "
          "defense overrides total_spread)")


def t_zero_jitter_is_baseline():
    """Control: sigma_eps = 0 is bit-identical to BASELINE."""
    mm = JitteredMM(0.0, 12345, horizon=T, k=K, gamma=GAMMA_CAP,
                    quote_size=QUOTE_SIZE)
    a = run(0, "JITTER-ZERO", horizon=SELFTEST_SHORT, mm_override=mm)
    b = run(0, "BASELINE", horizon=SELFTEST_SHORT)
    assert a["n_mm_fills"] > 0, "zero-jitter arm got ZERO fills; test vacuous"
    assert a["n_draws"] > 0, "the jitter RNG was never called at all"
    for f in ("r2_unsmoothed", "r2_norm_full", "sd_q", "n_kept_total",
              "n_views", "n_mm_fills"):
        assert a[f] == b[f], (
            "sigma_eps=0 did not reproduce the baseline on %s: %.15g vs "
            "%.15g" % (f, a[f], b[f]))
    for (wa, ra), (wb, rb) in zip(a["ma_curve"], b["ma_curve"]):
        assert wa == wb, (wa, wb)
        assert not math.isnan(ra), (
            "M=%d gave NaN -- a curve of NaNs would compare False everywhere "
            "and be scored as HELD. This is the n-ladder failure." % wa)
        assert ra == rb, (
            "sigma_eps=0 diverged from BASELINE at M=%d: %.15g vs %.15g"
            % (wa, ra, rb))
    print("  t_zero_jitter_baseline  PASS  (%d draws, all %d windows "
          "bit-identical to BASELINE, none NaN)"
          % (int(a["n_draws"]), len(a["ma_curve"])))


def t_jitter_degrades():
    """Jitter lowers R2."""
    a = run(0, "JITTER", horizon=SELFTEST_SHORT)
    b = run(0, "BASELINE", horizon=SELFTEST_SHORT)
    assert b["r2_unsmoothed"] > 0.99, (
        "BASELINE unsmoothed R2 is %.6f, not the ~1.0 the algebraic identity "
        "requires. Something upstream is wrong and no comparison below means "
        "anything." % b["r2_unsmoothed"])
    assert a["r2_unsmoothed"] < b["r2_unsmoothed"] - 0.05, (
        "SIGMA_EPS = $%.6f moved unsmoothed R2 only from %.6f to %.6f. The "
        "jitter is too weak for the curve to have any range to climb, and "
        "the test would be vacuous." % (SIGMA_EPS, b["r2_unsmoothed"],
                                        a["r2_unsmoothed"]))
    print("  t_jitter_degrades       PASS  (unsmoothed R2 %.6f -> %.6f, a "
          "drop of %.6f)" % (b["r2_unsmoothed"], a["r2_unsmoothed"],
                             b["r2_unsmoothed"] - a["r2_unsmoothed"]))


def t_smoothing_moves_jitter():
    """Smoothing moves the JITTER curve."""
    a = run(0, "JITTER", horizon=SELFTEST_SHORT)
    lo = curve_at(a, M_CONTROL)
    hi = curve_at(a, M_RISE)
    assert not math.isnan(lo) and not math.isnan(hi), (lo, hi)
    assert hi > lo, (
        "smoothing at M=%d did not raise JITTER's R2 above its M=1 value "
        "(%.6f -> %.6f). Either the smoother is not reaching the columns the "
        "fit uses, or the draws are not independent across views."
        % (M_RISE, lo, hi))
    print("  t_smoothing_moves       PASS  (JITTER M=1 %.6f -> M=%d %.6f, "
          "range %.6f)" % (lo, M_RISE, hi, hi - lo))


def t_determinism_and_nonempty():
    a = run(1, "JITTER", horizon=SELFTEST_SHORT)
    b = run(1, "JITTER", horizon=SELFTEST_SHORT)
    for f in ("r2_unsmoothed", "r2_norm_full", "sd_q", "n_kept_total"):
        assert a[f] == b[f], \
            "NOT DETERMINISTIC on %s: %.15g vs %.15g" % (f, a[f], b[f])
    assert a["ma_curve"] == b["ma_curve"], "NOT DETERMINISTIC on the M-curve"
    assert a["n_mm_fills"] > 0, (
        "ZERO MM FILLS at horizon %d. Every downstream check is vacuous. "
        "This is the failure mode SELFTEST_SHORT=86400 exists to avoid."
        % SELFTEST_SHORT)
    assert a["n_views"] > 0, "ZERO quote views; the sniffer saw nothing"
    assert a["n_prints"] > 0, "ZERO public prints; the book never traded"
    for w, r2 in a["ma_curve"]:
        assert not math.isnan(r2), (
            "M=%d produced NaN on a real run. fit_score returns NaN when the "
            "test labels have zero variance (sniffer_skew.py:226); scoring "
            "that would report HELD off no data." % w)
    assert a["n_kept_total"] > 10 * max(MA_WINDOWS), (
        "only %.0f kept samples against a top window of %d. The trailing "
        "mean needs many windows' worth of data or the partial-window head "
        "dominates the sample."
        % (a["n_kept_total"], max(MA_WINDOWS)))
    print("  t_determinism_nonempty  PASS  (%.0f MM fills, %.0f views, %.0f "
          "kept, no NaN at any window; repeated run identical)"
          % (a["n_mm_fills"], a["n_views"], a["n_kept_total"]))


def t_mirror():
    """Transcribed loop reproduces sniffer_skew.run at its own gamma."""
    mine = run(0, "BASELINE", gamma=DEFAULT_GAMMA)
    theirs = SKMOD.run(0, K)
    for f, g in (("r2_unsmoothed", "r2_mul"), ("r2_norm_full", "r2_norm"),
                 ("sd_q", "sd_q")):
        a, b = mine[f], theirs[g]
        assert abs(a - b) < 1e-12, (
            "T-MIRROR BROKEN on %s vs %s: this file %.12f vs "
            "sniffer_skew.run %.12f" % (f, g, a, b))
    assert mine["n_kept_total"] == theirs["n_kept"], \
        "T-MIRROR BROKEN on kept count: %.0f vs %.0f" % (mine["n_kept_total"],
                                                         theirs["n_kept"])
    print("  t_mirror                PASS  (r2 %.9f, r2_norm %.9f, sd_q "
          "%.9f all match sniffer_skew.run)"
          % (mine["r2_unsmoothed"], mine["r2_norm_full"], mine["sd_q"]))


def selftest():
    print("=" * 74)
    print("phase7_nobs.py self-test")
    print("=" * 74)
    print("  SIGMA_EPS = $%.6f  (%.1f x one time-averaged sd of skew)"
          % (SIGMA_EPS, SIGMA_EPS_MULT))
    print("  W_Q (QUANT) = %.6f BTC, imported from phase7_blind" % W_Q_LOW)
    print("  M sweep = %r, all fitted on the full kept sample" % (MA_WINDOWS,))
    print("")
    t0 = time.time()
    t_identity()
    t_trailing_mean()
    t_ma_m1_is_unsmoothed()
    t_ma_targets_only_skew()
    t_sigma_eps_arithmetic()
    t_predicted_values()
    t_scoring_rules()
    t_nan_halts()
    t_no_default_moved()
    t_zero_jitter_is_baseline()
    t_jitter_degrades()
    t_smoothing_moves_jitter()
    t_determinism_and_nonempty()
    t_mirror()
    print("")
    print("  all self-tests PASS in %.0fs" % (time.time() - t0))
    print("=" * 74)


# main
def main():
    bar = "=" * 118
    print(bar)
    print("Phase 7 arm B: smoothing axis. Recovery vs destruction.")
    print(bar)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present." % (LAM, P_MARKET, LIFE,
                                                     DISP))
    print("k=%.5f. gamma=%.2e. quote_size=%.3f BTC = DEFAULT_QUOTE_SIZE, "
          "untouched. %d seeds x %.0fs per arm."
          % (K, GAMMA_CAP, QUOTE_SIZE, len(SEEDS), T))
    print("")
    print("Axis: M, the attacker's trailing-mean window.")
    print("  (Not n: fit_score is a")
    print("  per-observation OLS, so errors-in-variables caps R2 independently")
    print("  of n. M is how")
    print("  much the attacker averages, which the two")
    print("  defenses respond to differently.")
    print("")
    print("The two defenses:")
    print("  JITTER  mean-zero Gaussian on the reservation price, "
          "sigma_eps = $%.6f" % SIGMA_EPS)
    print("          = %.1f x one time-averaged sd of the maker's own skew "
          "($%.6f)." % (SIGMA_EPS_MULT, SD_SKEW_REF))
    print("          Derived, not tuned; puts the M=1 ceiling near 0.5.")
    print("  QUANT   inventory bucketed at W_Q = %.6f BTC = %.2f x sd(q), arm A's" % (W_Q_LOW, W_Q_LOW / SD_Q_REF))
    print("          QUANT-LOW width, imported from phase7_blind.")
    print("")
    print("BASELINE R2 at M=1 is 1.000000 by construction, not by "
          "measurement.")
    print("  skew = -q*gamma*sigma^2*(T-t): the level feature is an")
    print("  invertible function of q.")
    print("  (BASELINE R2 = 1 by construction.)")
    print("  BASELINE is smoothed at the same M, so its")
    print("  R2 falls with M: the control for how much decline")
    print("  is the cost of smoothing itself.")
    print("")

    res = {}
    for arm in ARMS:
        t0 = time.time()
        res[arm] = [run(s, arm) for s in SEEDS]
        m, se = mean_se([r["r2_unsmoothed"] for r in res[arm]])
        print("  %-9s measured in %4.0fs   unsmoothed R2 = %.6f +/- %.6f   "
              "kept = %.0f" % (arm, time.time() - t0, m, se,
                               res[arm][0]["n_kept_total"]))
    print("")

    windows = check_windows(res["BASELINE"], "BASELINE")
    for arm in ARMS[1:]:
        got = check_windows(res[arm], arm)
        if got != windows:
            raise DegenerateComparison(
                "arm %s swept %r against BASELINE's %r" % (arm, got, windows))

    print(bar)
    print("Out-of-sample R2 against the trailing-mean window M")
    print(bar)
    print("  %6s %24s %24s %24s %11s"
          % ("M", "BASELINE", "JITTER", "QUANT", "PRED(JIT)"))
    for w in windows:
        cells = []
        for arm in ARMS:
            vals = curve_values(res[arm], w)
            require_finite("curve %s at M=%d" % (arm, w), *vals)
            m, se = mean_se(vals)
            cells.append("%12.6f+/-%9.6f" % (m, se))
        pred = ("%11.3f" % PRED[w]) if w in PRED else " " * 11
        print("  %6d %24s %24s %24s %s"
              % (w, cells[0], cells[1], cells[2], pred))
    print("")
    print("  Cell = mean across %d seeds of each seed's"
          % len(SEEDS))
    print("  out-of-sample R2 at that window, with the SE across seeds.")
    print("  Every window fitted on the full kept sample;")
    print("  M=1 asserted equal to the unsmoothed fit.")
    print("  PRED(JIT): pre-registered M/(M+1) prediction for JITTER.")
    print("")
    print("  Paired step changes per seed, window to window:")
    print("  %-14s %22s %22s %22s" % ("step", "BASELINE", "JITTER", "QUANT"))
    for j in range(len(windows) - 1):
        cells = []
        for arm in ARMS:
            m, se, mult = paired_between(res[arm], windows[j], windows[j + 1])
            cells.append("%+9.6f+/-%7.6f %4.1f" % (m, se, mult))
        print("  %5d->%-7d %22s %22s %22s"
              % (windows[j], windows[j + 1], cells[0], cells[1], cells[2]))
    print("")
    print("  Trailing number = |mean| / SE.")
    print("")

    print(bar)
    print("The pre-registered predictions")
    print(bar)

    v1, rise, appr = score_q1(res["JITTER"], res["BASELINE"])
    rm, rse, rmult = rise
    am, ase, amult = appr
    print("  Q1  JITTER R2 rises with M and approaches BASELINE at the same M")
    print("      Q1a rise      M=%d -> M=%d: %+.6f +/- %.6f   %.2f SE"
          % (M_CONTROL, M_RISE, rm, rse, rmult))
    print("      Q1b approach  JITTER - BASELINE at M=%d: %+.6f +/- %.6f   "
          "%.2f SE" % (M_RISE, am, ase, amult))
    for w in sorted(PRED):
        m, se = mean_se(curve_values(res["JITTER"], w))
        print("      measured vs predicted at M=%-3d  %.6f  vs  %.3f   "
              "(miss %+.6f)" % (w, m, PRED[w], m - PRED[w]))
    print("      -> %s" % v1)
    print("")

    v2, m2, se2, mult2 = score_q2(res["QUANT"])
    print("  Q2  QUANT R2 does not rise with M")
    print("      Scored: flat within %.1f SE from M=%d to M=%d, paired."
          % (Z_SE, M_CONTROL, M_TOP))
    print("      QUANT M=%d -> M=%d: %+.6f +/- %.6f   %.2f SE"
          % (M_CONTROL, M_TOP, m2, se2, mult2))
    print("      -> %s" % v2)
    if v2.startswith("REFUTED (RISES)"):
        print("")
        print("      Destruction claim REFUTED: averaging recovered")
        print("      signal from a quantized quote.")
        print("      Hypothesis (not scored):")
        print("      quantization error is a deterministic sawtooth in q, and")
        print("      with W_Q = %.6f against sd(q) = %.5f the inventory"
              % (W_Q_LOW, SD_Q_REF))
        print("      crosses bucket boundaries often enough that averaging")
        print("      across crossings cancels part of the bucket error.")
    print("")

    v3, m3, se3, mult3 = score_q3(res["JITTER"])
    print("  Q3  climb turns over (descriptive)")
    print("      AR1_DT = 60s, so M=%d is five AR(1) steps and the trailing"
          % M_TOP)
    print("      mean blurs signal as well as noise.")
    print("      JITTER M=%d -> M=%d: %+.6f +/- %.6f   %.2f SE"
          % (M_TURN, M_TOP, m3, se3, mult3))
    print("      -> %s" % v3)
    print("")

    print(bar)
    print("Shapes of the two curves")
    print(bar)
    print("  Expected contrast: JITTER climbs")
    print("  with M (independent noise averages away); QUANT does not")
    print("  (its error is a deterministic function of q")
    print("  that is never emitted).")
    print("")
    jm, jse, jmult = paired_between(res["JITTER"], M_CONTROL, M_TOP)
    qm, qse, qmult = paired_between(res["QUANT"], M_CONTROL, M_TOP)
    bm, bse, bmult = paired_between(res["BASELINE"], M_CONTROL, M_TOP)
    print("  total climb M=%d -> M=%d, paired per seed:" % (M_CONTROL, M_TOP))
    print("    JITTER    %+.6f +/- %.6f   %.2f SE" % (jm, jse, jmult))
    print("    QUANT     %+.6f +/- %.6f   %.2f SE" % (qm, qse, qmult))
    print("    BASELINE  %+.6f +/- %.6f   %.2f SE" % (bm, bse, bmult))
    print("")
    print("  BASELINE's change = cost of smoothing itself. If")
    print("  negative, part of any decline in the")
    print("  defended arms is smoothing, so")
    print("  read the defended curves against")
    print("  BASELINE at the same M, not against 1.000000.")
    print("")
    print("  Unsmoothed single-feature (skew_norm only) scores, for "
          "comparison:")
    for arm in ARMS:
        m, se = mean_se([r["r2_norm_full"] for r in res[arm]])
        print("    %-9s %.6f +/- %.6f" % (arm, m, se))
    print("")

    print(bar)
    print("Limitations")
    print(bar)
    print("  L1  Non-adaptive attacker: fixed window,")
    print("      no bucket-boundary detection (the best")
    print("      response to quantization;")
    print("      not implemented).")
    print("      Every R2 here is an upper bound on how well each defense")
    print("      works against a real adversary.")
    print("  L2  Defenses not calibrated to equal strength:")
    print("      JITTER at one sd of skew and QUANT at %.2f sd of inventory"
          % (W_Q_LOW / SD_Q_REF))
    print("      different amounts of hiding; levels of the two")
    print("      curves not comparable, only their shapes in M.")
    print("      only shape is scored.")
    print("  L3  Trailing mean uses a running sum over up to")
    print("      %.0f rows (floating-point drift vs a"
          % res["BASELINE"][0]["n_views"])
    print("      pairwise or Kahan sum); M=1 is")
    print("      special-cased to the identity so the control")
    print("      demand bit-equality.")
    print("  L4  At large M the smoothed skew columns are highly collinear;")
    print("      the committed ols skips singular pivots,")
    print("      so a large-M fit may use fewer")
    print("      effective columns.")
    print("  L5  sigma_eps derived at the time-averaged tau; realized")
    print("      noise-to-signal is better than designed early in the")
    print("      day and worse late; the regression pools both.")
    print("      M/(M+1) prediction inherits that approximation.")
    print("  L6  No committed module or default modified; both defenses")
    print("      are subclasses.")
    print("      does with FlatQuoteMM.")
    print("")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
