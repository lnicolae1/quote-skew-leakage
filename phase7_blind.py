#!/usr/bin/env python3
# phase7_blind.py: Phase 7 arm A: quantized skew; interior point or only two endpoints?
# - quantization: maker prices off q rounded to the nearest multiple of W_Q
# - arms: BASELINE (AS), QUANT-LOW (w/sd(q) = 0.34), QUANT-HIGH (w/sd(q) = 2.50), FLAT (no skew)
# - widths from the measured sd(q) = 0.04604 BTC; defense overrides reservation_price only
# - measures: sniffer R2 (sniffer_skew.fit_score), R_direct (participation_skew.series_stats), sd(q), max|q|
# - decision rule (fixed before any arm ran; paired per seed, Z_SE = 2.0):
#   NO INTERIOR if R_direct(QUANT-HIGH) - R_direct(FLAT) is within 2.0 SE of zero
#   INTERIOR if that difference is >= 2.0 SE above zero and R2(QUANT-HIGH) - R2(BASELINE) is >= 2.0 SE below zero
#   NEITHER otherwise: (a) QUANT-HIGH costs more than FLAT, (b) cheaper but does not blind
# - power: removing skew costs +0.0337 +/- 0.0082 (quote_size 0.020); NO INTERIOR = not distinguishable at this power
# - NaN inputs to any rule raise; the sniffer is non-adaptive (no boundary-crossing detector)

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import participation_skew as PSMOD
import sniffer_skew as SKMOD

from clipped_mm_check import _resid_best, MM_ID
from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA
from participation_skew import FlatQuoteMM, series_stats
from sniffer_fill_only import PublicPrint, TRAIN_FRAC
from sniffer_skew import build_features, fit_score, TAU_FLOOR_FRAC

# is-identity: scoring functions are the committed objects
assert build_features is SKMOD.build_features, \
    "build_features must be sniffer_skew's own object, not a copy"
assert fit_score is SKMOD.fit_score, \
    "fit_score must be sniffer_skew's own object, not a copy"
assert FlatQuoteMM is PSMOD.FlatQuoteMM, \
    "FlatQuoteMM must be participation_skew's own class, not a copy"
assert series_stats is PSMOD.series_stats, \
    "series_stats (R_direct) must be participation_skew's own object"
assert TAU_FLOOR_FRAC is SKMOD.TAU_FLOOR_FRAC, \
    "TAU_FLOOR_FRAC must be sniffer_skew's own value"
assert issubclass(FlatQuoteMM, MarketMaker), \
    "FlatQuoteMM must subclass the committed MarketMaker"

# constants (committed values or arithmetic on them)
T = 86400.0
N_SEC = int(T)
SEEDS = list(range(8))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
QUOTE_SIZE = 0.020
AR1_DT = 60

# gate-2 viability cap where the sd(q) anchor was measured
GAMMA_CAP = 9.4e-6

# measured sd(q), 8 seeds, this configuration
SD_Q_REF = 0.04604
MAX_ABS_Q_REF = 0.15129

# sniffer_saturation: R2 ~0.985 at w/sd 0.34; below 0.5 at w/sd ~2.5
W_OVER_SD_LOW = 0.34
W_OVER_SD_HIGH = 2.5
W_Q_LOW = W_OVER_SD_LOW * SD_Q_REF
W_Q_HIGH = W_OVER_SD_HIGH * SD_Q_REF

Z_SE = 2.0

ARMS = ("BASELINE", "QUANT-LOW", "QUANT-HIGH", "FLAT")

# self-test constants
SELFTEST_SHORT = 86400
# worst-case centre displacement (W_Q/2) * C_at_t0 under one tick
W_Q_TINY = 0.0005
# > 2 * max|q|: bucket(q) = 0, QuantizedMM == FlatQuoteMM
W_Q_HUGE = 1.0
TICK = 0.01
DEGENERATE_R2_MAX = 0.5


# defense
class QuantizedMM(MarketMaker):
    """AS maker pricing off q rounded to the nearest multiple of w_q."""

    def __init__(self, w_q, **kw):
        MarketMaker.__init__(self, **kw)
        if w_q <= 0:
            raise ValueError("w_q must be positive, got %r" % (w_q,))
        self.w_q = w_q

    def bucket_q(self):
        """q rounded half-up to the w_q grid; 0 when |q| < w_q/2."""
        return self.w_q * math.floor(self.q / self.w_q + 0.5)

    def reservation_price(self, mid, t):
        sig = self.sigma_absolute(mid)
        tau = self.time_remaining_years(t)
        return mid - self.bucket_q() * self.gamma * sig * sig * tau

    def inventory_skew(self, mid, t):
        """Skew actually quoted: -bucket_q * C."""
        sig = self.sigma_absolute(mid)
        tau = self.time_remaining_years(t)
        return -self.bucket_q() * self.gamma * sig * sig * tau


def make_mm(arm, gamma=GAMMA_CAP, quote_size=QUOTE_SIZE):
    if arm == "BASELINE":
        return MarketMaker(horizon=T, k=K, gamma=gamma, quote_size=quote_size)
    if arm == "QUANT-LOW":
        return QuantizedMM(W_Q_LOW, horizon=T, k=K, gamma=gamma,
                           quote_size=quote_size)
    if arm == "QUANT-HIGH":
        return QuantizedMM(W_Q_HIGH, horizon=T, k=K, gamma=gamma,
                           quote_size=quote_size)
    if arm == "FLAT":
        return FlatQuoteMM(horizon=T, k=K, gamma=gamma, quote_size=quote_size)
    raise ValueError("unknown arm %r" % (arm,))


# simulation loop
def run(seed, arm, gamma=GAMMA_CAP, horizon=T, mm_override=None,
        collect_quotes=False):
    """One arm, one seed; loop transcribed from sniffer_skew.run plus 60 s inventory sampling."""
    n_sec = int(horizon)
    _p, vf, eng, noise, informed = make_world(seed, horizon, LAM, P_MARKET,
                                              LIFE, DISP, "join")
    mm = mm_override if mm_override is not None else make_mm(arm, gamma)

    views, prints, q_series, q_sampled = [], [], [], []
    # q_series: view-gated (as sniffer_skew.run); q_every: one per second (as participation_skew)
    q_every = []
    quotes_seen = []
    n_mm_fills = 0
    last_mid = None

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
        row = mm.requote(eng, t)
        if collect_quotes:
            quotes_seen.append((row.quoted_bid, row.quoted_ask))

        q_every.append(mm.q)
        if int(t) % AR1_DT == 0:
            q_sampled.append(mm.q)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        last_mid = 0.5 * (bb + ba)

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

    c = {nm: i for i, nm in enumerate(names)}
    mul_cols = [i for i, nm in enumerate(names) if nm != "skew_norm_fullmid"]
    r2_mul, _rm = fit_score(rows, y, keep, mul_cols)
    r2_nrm, _rn = fit_score(rows, y, keep, [c["const"], c["skew_norm"]])

    st = series_stats(q_sampled)

    return {
        "r2_mul": r2_mul,
        "r2_norm": r2_nrm,
        "r_direct": st["r_direct"] if st else float("nan"),
        "r_ar1": st["r_ar1"] if st else float("nan"),
        "phi": st["phi"] if st else float("nan"),
        # sd_q over the view-gated series; sd_q_all / max_abs_q over the unconditional one
        "sd_q": statistics.stdev(y) if len(y) > 1 else 0.0,
        "sd_q_all": statistics.stdev(q_every) if len(q_every) > 1 else 0.0,
        "max_abs_q": max(abs(v) for v in q_every) if q_every else 0.0,
        "pnl": mm.mark_to_market(last_mid) if last_mid is not None else 0.0,
        "n_mm_fills": float(n_mm_fills),
        "n_views": float(len(views)),
        "n_kept": float(sum(1 for x in keep if x)),
        "n_prints": float(len(prints)),
        "quotes": quotes_seen,
    }


# statistics
class DegenerateComparison(ValueError):
    """Raised when a decision rule gets a value it cannot compare."""


def require_finite(label, *vals):
    """Every value must be a finite float; NaN and inf raise."""
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise DegenerateComparison(
                "%s: value %d is %r, not a number" % (label, i, v))
        if math.isnan(v):
            raise DegenerateComparison(
                "%s: value %d is NaN. A NaN reaching a decision rule would "
                "compare False against every threshold and be reported as "
                "'no violation'. Halting instead." % (label, i))
        if math.isinf(v):
            raise DegenerateComparison(
                "%s: value %d is %r. An infinite INPUT to a rule is not a "
                "measurement." % (label, i, v))
    return True


def mean_se(xs):
    if not xs:
        raise DegenerateComparison(
            "mean_se got an EMPTY sample. An empty comparison is the failure "
            "mode that passed as [] == [] once already in this project.")
    require_finite("mean_se sample", *xs)
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, statistics.stdev(xs) / math.sqrt(len(xs))


def se_mult(d, se):
    """|mean| / SE; 0 for 0/0, inf for nonzero/0."""
    require_finite("se_mult", d, se)
    if se < 0.0:
        raise DegenerateComparison("se_mult got a NEGATIVE SE: %r" % (se,))
    if se > 0.0:
        return abs(d) / se
    return float("inf") if d != 0.0 else 0.0


def paired(a, b, field):
    """Per-seed differences, then their mean and SE."""
    if len(a) != len(b):
        raise DegenerateComparison(
            "paired() got %d and %d seeds on %s; a paired difference needs "
            "the same seeds on both sides" % (len(a), len(b), field))
    d = [x[field] - y[field] for x, y in zip(a, b)]
    m, se = mean_se(d)
    return d, m, se


# decision rule
def verdict(d_cost, se_cost, d_r2, se_r2):
    """Decision rule. d_cost = R_direct(QUANT-HIGH) - R_direct(FLAT); d_r2 = R2(QUANT-HIGH) - R2(BASELINE)."""
    require_finite("verdict inputs (d_cost, se_cost, d_r2, se_r2)",
                   d_cost, se_cost, d_r2, se_r2)
    m_cost = se_mult(d_cost, se_cost)
    m_r2 = se_mult(d_r2, se_r2)

    cheaper = (d_cost > 0.0) and (m_cost >= Z_SE)
    costlier = (d_cost < 0.0) and (m_cost >= Z_SE)
    blinds = (d_r2 < 0.0) and (m_r2 >= Z_SE)

    if m_cost < Z_SE:
        return "NO INTERIOR", m_cost, m_r2
    if cheaper and blinds:
        return "INTERIOR", m_cost, m_r2
    if costlier:
        return "NEITHER (a): QUANT-HIGH costs MORE than FLAT", m_cost, m_r2
    return "NEITHER (b): cheaper than FLAT but does not blind", m_cost, m_r2


# self-tests
def t_identity():
    """Imported names are the modules' own objects."""
    assert build_features is SKMOD.build_features
    assert fit_score is SKMOD.fit_score
    assert FlatQuoteMM is PSMOD.FlatQuoteMM
    assert series_stats is PSMOD.series_stats
    assert TRAIN_FRAC == 0.70, \
        "TRAIN_FRAC moved off 0.70; every committed R2 assumed 0.70"
    assert PSMOD.AR1_DT == AR1_DT, (
        "participation_skew's inventory spacing moved; series_stats would be "
        "reading a series sampled at a different rate than it assumes")
    assert SELFTEST_SHORT == int(T), (
        "SELFTEST_SHORT and T have diverged. make_mm builds every maker with "
        "horizon=T, so a self-test run at a different horizon would give the "
        "maker a clock that disagrees with its own loop.")
    print("  t_identity              PASS  (6 objects, all is-identity)")


def t_bucket_arithmetic():
    """bucket_q rounds half-up; zero inside +/- w_q/2."""
    mm = QuantizedMM(0.10, horizon=T, k=K, gamma=GAMMA_CAP,
                     quote_size=QUOTE_SIZE)
    cases = [(0.0, 0.0), (0.04, 0.0), (0.049999, 0.0), (0.06, 0.1),
             (0.14, 0.1), (-0.04, 0.0), (-0.06, -0.1), (0.26, 0.3)]
    for q, want in cases:
        mm.q = q
        got = mm.bucket_q()
        assert abs(got - want) < 1e-12, \
            "bucket_q(%.6f) = %.6f, expected %.6f" % (q, got, want)
    print("  t_bucket_arithmetic     PASS  (%d cases on the grid)"
          % len(cases))


def t_bound():
    """Centre displacement bounded by (w_q/2) * C."""
    base = make_mm("BASELINE")
    worst = 0.0
    for w_q in (W_Q_TINY, W_Q_LOW, W_Q_HIGH):
        qm = QuantizedMM(w_q, horizon=T, k=K, gamma=GAMMA_CAP,
                         quote_size=QUOTE_SIZE)
        for q in [-0.3, -0.16, -0.05, -0.004, 0.0, 0.004, 0.05, 0.16, 0.3]:
            for t in (0.0, 12345.0, 43200.0, 80000.0):
                base.q = qm.q = q
                sig = qm.sigma_absolute(62000.0)
                c_t = qm.gamma * sig * sig * qm.time_remaining_years(t)
                d = abs(qm.reservation_price(62000.0, t)
                        - base.reservation_price(62000.0, t))
                lim = 0.5 * w_q * c_t + 1e-9
                assert d <= lim, \
                    ("centre displacement %.8f exceeds the structural bound "
                     "%.8f at w_q=%.6f q=%.4f t=%.0f" % (d, lim, w_q, q, t))
                worst = max(worst, d)
    print("  t_bound                 PASS  (worst displacement $%.6f over "
          "108 states)" % worst)


def t_tiny_is_baseline():
    """Control 1: tiny w_q stays within one tick of the baseline for a full day."""
    a = run(0, "BASELINE", horizon=SELFTEST_SHORT, collect_quotes=True)
    mm = QuantizedMM(W_Q_TINY, horizon=T, k=K, gamma=GAMMA_CAP,
                     quote_size=QUOTE_SIZE)
    b = run(0, "QUANT-TINY", horizon=SELFTEST_SHORT, mm_override=mm,
            collect_quotes=True)
    assert a["n_mm_fills"] > 0, "baseline got ZERO fills; control is vacuous"
    assert len(a["quotes"]) == len(b["quotes"]), \
        "quote counts differ: %d vs %d" % (len(a["quotes"]), len(b["quotes"]))
    worst = 0.0
    for (ab, aa), (bb, ba) in zip(a["quotes"], b["quotes"]):
        worst = max(worst, abs(ab - bb), abs(aa - ba))
    assert worst < TICK, \
        ("w_q=%.5f BTC displaced a quote by $%.6f, which is at or above one "
         "tick ($%.2f). Sub-tick control FAILED." % (W_Q_TINY, worst, TICK))
    print("  t_tiny_is_baseline      PASS  (%d quotes, worst gap $%.6f < one "
          "tick $%.2f)" % (len(a["quotes"]), worst, TICK))


def t_huge_is_flat():
    """Control 2: w_q > 2*max|q| is bit-identical to FlatQuoteMM."""
    assert W_Q_HUGE > 2.0 * MAX_ABS_Q_REF, \
        "W_Q_HUGE must exceed 2*max|q| or the bucket is not identically zero"
    mm = QuantizedMM(W_Q_HUGE, horizon=T, k=K, gamma=GAMMA_CAP,
                     quote_size=QUOTE_SIZE)
    a = run(0, "QUANT-HUGE", horizon=SELFTEST_SHORT, mm_override=mm,
            collect_quotes=True)
    b = run(0, "FLAT", horizon=SELFTEST_SHORT, collect_quotes=True)
    assert a["n_mm_fills"] > 0, "degenerate arm got ZERO fills; test vacuous"
    assert a["max_abs_q"] < 0.5 * W_Q_HUGE, \
        ("inventory reached %.5f, which is at or beyond half the bucket "
         "%.5f -- the bucket is NOT identically zero and this control does "
         "not test what it claims" % (a["max_abs_q"], 0.5 * W_Q_HUGE))
    assert len(a["quotes"]) == len(b["quotes"])
    for i, ((ab, aa), (bb, ba)) in enumerate(zip(a["quotes"], b["quotes"])):
        assert ab == bb and aa == ba, \
            ("quote %d differs from FlatQuoteMM: (%.10f, %.10f) vs "
             "(%.10f, %.10f)" % (i, ab, aa, bb, ba))
    assert a["r2_mul"] < DEGENERATE_R2_MAX, \
        ("a zero-skew maker scored R2 = %.6f, at or above %.2f. Something "
         "other than the skew is predicting inventory and no R2 in this file "
         "means what it claims." % (a["r2_mul"], DEGENERATE_R2_MAX))
    print("  t_huge_is_flat          PASS  (%d quotes bit-identical to "
          "FlatQuoteMM; R2 collapsed to %.6f)"
          % (len(a["quotes"]), a["r2_mul"]))


def t_mirror():
    """Transcribed loop reproduces sniffer_skew.run."""
    mine = run(0, "BASELINE", gamma=DEFAULT_GAMMA)
    theirs = SKMOD.run(0, K)
    for f in ("r2_mul", "r2_norm", "sd_q"):
        a, b = mine[f], theirs[f]
        assert abs(a - b) < 1e-12, \
            ("T-MIRROR BROKEN on %s: this file %.12f vs sniffer_skew.run "
             "%.12f. The transcribed loop has drifted." % (f, a, b))
    assert mine["n_kept"] == theirs["n_kept"], \
        "T-MIRROR BROKEN on n_kept: %.0f vs %.0f" % (mine["n_kept"],
                                                     theirs["n_kept"])
    print("  t_mirror                PASS  (r2_mul %.9f, r2_norm %.9f, sd_q "
          "%.9f all match sniffer_skew.run)"
          % (mine["r2_mul"], mine["r2_norm"], mine["sd_q"]))


def t_determinism_and_nonempty():
    """Deterministic and non-empty."""
    a = run(1, "QUANT-LOW", horizon=SELFTEST_SHORT)
    b = run(1, "QUANT-LOW", horizon=SELFTEST_SHORT)
    for f in ("r2_mul", "r2_norm", "r_direct", "sd_q", "sd_q_all",
              "max_abs_q", "pnl"):
        assert a[f] == b[f], \
            "NOT DETERMINISTIC on %s: %.15g vs %.15g" % (f, a[f], b[f])
    assert a["n_mm_fills"] > 0, \
        ("ZERO MM FILLS at horizon %d. Every downstream check is vacuous. "
         "This is the failure mode SELFTEST_SHORT=86400 exists to avoid."
         % SELFTEST_SHORT)
    assert a["n_views"] > 0, "ZERO quote views; the sniffer saw nothing"
    assert a["n_kept"] > 0, "ZERO kept samples; fit_score had no data"
    assert a["n_prints"] > 0, "ZERO public prints; the book never traded"
    print("  t_determinism_nonempty  PASS  (%.0f MM fills, %.0f views, %.0f "
          "kept, %.0f prints; repeated run identical)"
          % (a["n_mm_fills"], a["n_views"], a["n_kept"], a["n_prints"]))


def t_verdict_rule():
    """Decision rule on constructed inputs, all branches."""
    v, _, _ = verdict(0.001, 0.010, -0.40, 0.02)
    assert v == "NO INTERIOR", v
    v, _, _ = verdict(0.030, 0.005, -0.40, 0.02)
    assert v == "INTERIOR", v
    v, _, _ = verdict(0.030, 0.005, -0.01, 0.02)
    assert v.startswith("NEITHER (b)"), v
    v, _, _ = verdict(-0.030, 0.005, -0.40, 0.02)
    assert v.startswith("NEITHER (a)"), v
    # boundary counts as meeting it, both directions
    v, _, _ = verdict(0.020, 0.010, -0.40, 0.20)
    assert v == "INTERIOR", "Z_SE boundary must be inclusive, got %s" % v
    v, _, _ = verdict(0.0199, 0.010, -0.40, 0.20)
    assert v == "NO INTERIOR", "just inside 2 SE must be NO INTERIOR"
    print("  t_verdict_rule          PASS  (6 constructed cases, all four "
          "branches plus both boundaries)")


def t_nan_halts():
    """NaN reaching any rule raises."""
    nan = float("nan")
    n_raised = 0

    for args in ((nan, 0.01, -0.40, 0.20),
                 (0.02, nan, -0.40, 0.20),
                 (0.02, 0.01, nan, 0.20),
                 (0.02, 0.01, -0.40, nan)):
        try:
            verdict(*args)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError(
                "verdict(%r) returned a verdict on a NaN input instead of "
                "halting" % (args,))

    for args in ((nan, 0.01), (0.02, nan)):
        try:
            se_mult(*args)
        except DegenerateComparison:
            n_raised += 1
        else:
            raise AssertionError("se_mult(%r) accepted a NaN" % (args,))

    try:
        mean_se([0.1, nan, 0.3])
    except DegenerateComparison:
        n_raised += 1
    else:
        raise AssertionError("mean_se accepted a NaN sample")

    try:
        mean_se([])
    except DegenerateComparison:
        n_raised += 1
    else:
        raise AssertionError(
            "mean_se accepted an EMPTY sample -- the [] == [] failure mode")

    # guards must not fire on legitimate values (incl. se_mult's inf)
    assert se_mult(0.5, 0.0) == float("inf")
    assert se_mult(0.0, 0.0) == 0.0
    v, _, _ = verdict(0.02, 0.01, -0.40, 0.20)
    assert v == "INTERIOR", v

    print("  t_nan_halts             PASS  (%d degenerate inputs all raised; "
          "legitimate inf still allowed)" % n_raised)


def t_no_default_moved():
    """No committed default mutated by import."""
    assert DEFAULT_GAMMA == 1e-6, "DEFAULT_GAMMA moved"
    from market_maker import DEFAULT_QUOTE_SIZE, DEFAULT_TICK
    assert DEFAULT_QUOTE_SIZE == 0.02, "DEFAULT_QUOTE_SIZE moved"
    assert DEFAULT_TICK == 0.01, "DEFAULT_TICK moved"
    assert MarketMaker.reservation_price is not QuantizedMM.reservation_price
    assert MarketMaker.total_spread is QuantizedMM.total_spread, \
        ("QuantizedMM must NOT override total_spread -- the arms have to quote "
         "the same width or they win different flow")
    assert MarketMaker.total_spread is FlatQuoteMM.total_spread, \
        "FlatQuoteMM must NOT override total_spread either"
    print("  t_no_default_moved      PASS  (3 defaults intact; neither "
          "defense overrides total_spread)")


def selftest():
    print("=" * 74)
    print("phase7_blind.py self-test")
    print("=" * 74)
    print("  W_Q_LOW  = %.6f BTC  (%.2f x sd(q) = %.5f)"
          % (W_Q_LOW, W_OVER_SD_LOW, SD_Q_REF))
    print("  W_Q_HIGH = %.6f BTC  (%.2f x sd(q) = %.5f)"
          % (W_Q_HIGH, W_OVER_SD_HIGH, SD_Q_REF))
    print("")
    t0 = time.time()
    t_identity()
    t_bucket_arithmetic()
    t_bound()
    t_verdict_rule()
    t_nan_halts()
    t_no_default_moved()
    t_tiny_is_baseline()
    t_huge_is_flat()
    t_determinism_and_nonempty()
    t_mirror()
    print("")
    print("  all self-tests PASS in %.0fs" % (time.time() - t0))
    print("=" * 74)


# main
def main():
    bar = "=" * 126
    print(bar)
    print("Phase 7 arm A: interior point or only two endpoints?")
    print(bar)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present." % (LAM, P_MARKET, LIFE,
                                                     DISP))
    print("k=%.5f. gamma=%.2e (gate-2 viability cap, where sd(q) was measured)." % (K, GAMMA_CAP))
    print("quote_size=%.3f BTC = DEFAULT_QUOTE_SIZE, untouched. %d seeds x "
          "%.0fs per arm." % (QUOTE_SIZE, len(SEEDS), T))
    print("")
    print("Bucket widths from the measured sd(q) = %.5f BTC:" % SD_Q_REF)
    print("  QUANT-LOW   w / sd(q) = %.2f  ->  W_Q = %.6f BTC   "
          "(saturation R2 ~ 0.985)"
          % (W_OVER_SD_LOW, W_Q_LOW))
    print("  QUANT-HIGH  w / sd(q) = %.2f  ->  W_Q = %.6f BTC   (blinding width)"
          % (W_OVER_SD_HIGH, W_Q_HIGH))
    print("")
    print("BASELINE R2 = 1.000000 is by construction, not by measurement.")
    print("  skew = -q*gamma*sigma^2*(T-t): the level feature is an")
    print("  invertible function of q (BASELINE R2 = 1 by construction).")
    print("")
    print("Power:")
    print("  removing skew costs +0.0337 +/- 0.0082 at this quote")
    print("  size (not distinguishable from zero at the other two sizes).")
    print("  A partial defense costs a fraction")
    print("  of that: power to detect NO INTERIOR; may")
    print("  miss a small INTERIOR. NO INTERIOR =")
    print("  not distinguishable at this power.")
    print("")

    res = {}
    for arm in ARMS:
        t0 = time.time()
        res[arm] = [run(s, arm) for s in SEEDS]
        r2m, r2s = mean_se([r["r2_mul"] for r in res[arm]])
        rdm, rds = mean_se([r["r_direct"] for r in res[arm]])
        print("  %-11s measured in %4.0fs   R2(mul)=%.6f+/-%.6f   "
              "R_direct=%.4f+/-%.4f"
              % (arm, time.time() - t0, r2m, r2s, rdm, rds))
    print("")

    print(bar)
    print("A. Inference axis: sniffer R2 (sniffer_skew.fit_score)")
    print(bar)
    print("  %-12s %22s %22s" % ("arm", "R2 multi-feature", "R2 skew_norm "
                                 "only"))
    for arm in ARMS:
        a, ase = mean_se([r["r2_mul"] for r in res[arm]])
        b, bse = mean_se([r["r2_norm"] for r in res[arm]])
        print("  %-12s %11.6f+/-%9.6f %11.6f+/-%9.6f"
              % (arm, a, ase, b, bse))
    print("")
    print("  Multi-feature = adversary's model: all 12 columns except")
    print("  the full-mid sensitivity.")
    print("  Primary for the rule below.")
    print("")
    print("  Paired against BASELINE, per seed:")
    print("  %-12s %24s %9s" % ("arm", "d R2(mul)", "SE mult"))
    for arm in ARMS[1:]:
        _d, m, se = paired(res[arm], res["BASELINE"], "r2_mul")
        print("  %-12s %12.6f +/- %8.6f %9.2f"
              % (arm, m, se, se_mult(m, se)))
    print("")

    print(bar)
    print("B. Cost axis: R_direct (participation_skew.series_stats)")
    print(bar)
    print("  R_direct = -slope of (q[t+H] - q[t]) on demeaned q[t], "
          "H=%.0fs, train slice only." % PSMOD.H)
    print("  Higher = more inventory reverts within the hour.")
    print("")
    print("  %-12s %22s %22s" % ("arm", "R_direct", "R_ar1 (continuity)"))
    for arm in ARMS:
        a, ase = mean_se([r["r_direct"] for r in res[arm]])
        b, bse = mean_se([r["r_ar1"] for r in res[arm]])
        print("  %-12s %11.4f+/-%9.4f %11.4f+/-%9.4f"
              % (arm, a, ase, b, bse))
    print("")
    print("  paired against FLAT (no skew), per seed.")
    print("  %-12s %24s %9s" % ("arm", "d R_direct", "SE mult"))
    for arm in ("BASELINE", "QUANT-LOW", "QUANT-HIGH"):
        _d, m, se = paired(res[arm], res["FLAT"], "r_direct")
        print("  %-12s %12.4f +/- %8.4f %9.2f"
              % (arm, m, se, se_mult(m, se)))
    print("")
    print("  BASELINE minus FLAT corresponds to the +0.0337")
    print("  reference, at gamma=%.2e here vs %.2e there"
          % (GAMMA_CAP, PSMOD.GAMMA_CAP))
    print("  and %d seeds vs %d: not the same"
          % (len(SEEDS), len(PSMOD.SEEDS)))
    print("  experiment; no comparison asserted.")
    print("")

    print(bar)
    print("C. Feedback: does the defense change Var(q)?")
    print(bar)
    print("  Analytic cap R2 = 1 - (w^2/12)/Var(q) assumes Var(q) fixed;")
    print("  quantized quotes feed back into q, so")
    print("  inventory can park inside a bucket. If sd(q)")
    print("  moves across arms, the analytic prediction does not apply")
    print("  to the real defense.")
    print("")
    print("  %-12s %20s %20s %14s" % ("arm", "sd(q) BTC", "max|q| BTC",
                                      "sd/BASELINE"))
    base_sd, _ = mean_se([r["sd_q_all"] for r in res["BASELINE"]])
    for arm in ARMS:
        a, ase = mean_se([r["sd_q_all"] for r in res[arm]])
        b, bse = mean_se([r["max_abs_q"] for r in res[arm]])
        print("  %-12s %9.5f+/-%9.5f %9.5f+/-%9.5f %14.4f"
              % (arm, a, ase, b, bse, a / base_sd if base_sd else float("nan")))
    print("")
    print("  paired against BASELINE, per seed:")
    print("  %-12s %24s %9s" % ("arm", "d sd(q)", "SE mult"))
    for arm in ARMS[1:]:
        _d, m, se = paired(res[arm], res["BASELINE"], "sd_q_all")
        print("  %-12s %12.5f +/- %8.5f %9.2f"
              % (arm, m, se, se_mult(m, se)))
    print("")
    print("  Committed reference for BASELINE: sd(q) = %.5f, max|q| = %.5f"
          % (SD_Q_REF, MAX_ABS_Q_REF))
    print("  (8 seeds, gamma 9.40e-06).")
    print("")

    print(bar)
    print("D. MM PnL; reported, not scored")
    print(bar)
    print("  Control arm loses -525.18 +/- 162.43 on 8 of 8 seeds")
    print("  (damage_split_results.txt); PnL differences are")
    print("  perturbations of a baseline")
    print("  that is itself negative; printed for")
    print("  completeness and adjudicates nothing.")
    print("")
    print("  %-12s %24s %26s" % ("arm", "MM PnL ($)", "d vs BASELINE ($)"))
    for arm in ARMS:
        a, ase = mean_se([r["pnl"] for r in res[arm]])
        if arm == "BASELINE":
            print("  %-12s %12.2f +/- %8.2f %26s" % (arm, a, ase, "--"))
        else:
            _d, m, se = paired(res[arm], res["BASELINE"], "pnl")
            print("  %-12s %12.2f +/- %8.2f %14.2f +/- %8.2f"
                  % (arm, a, ase, m, se))
    print("")

    print(bar)
    print("The pre-registered verdict")
    print(bar)
    d_cost_v, d_cost, se_cost = paired(res["QUANT-HIGH"], res["FLAT"],
                                       "r_direct")
    d_r2_v, d_r2, se_r2 = paired(res["QUANT-HIGH"], res["BASELINE"], "r2_mul")
    v, m_cost, m_r2 = verdict(d_cost, se_cost, d_r2, se_r2)

    print("  cost  R_direct(QUANT-HIGH) - R_direct(FLAT)     "
          "%+.4f +/- %.4f   %.2f SE" % (d_cost, se_cost, m_cost))
    print("        per seed: " + "  ".join("%+.4f" % x for x in d_cost_v))
    print("  Blind R2(QUANT-HIGH) - R2(BASELINE)             "
          "%+.6f +/- %.6f   %.2f SE" % (d_r2, se_r2, m_r2))
    print("        per seed: " + "  ".join("%+.4f" % x for x in d_r2_v))
    print("")
    print("  Threshold for every branch: %.1f SE." % Z_SE)
    print("")
    print("  VERDICT: %s" % v)
    print("")
    if v == "NO INTERIOR":
        print("  The blinding width costs the maker as much as")
        print("  abandoning skew. No useful partial defense on this book:")
        print("  two endpoints,")
        print("  no interior.")
        print("")
        print("  Not distinguishable at this power (not")
        print("  proven identical). Removing skew costs")
        print("  +0.0337 +/- 0.0082 at this size; an interior point worth")
        print("  a fraction of that would not clear %.1f SE at %d seeds and"
              % (Z_SE, len(SEEDS)))
        print("  this design would miss it.")
    elif v == "INTERIOR":
        print("  The defense blinds the sniffer and costs the maker")
        print("  less than abandoning skew: interior")
        print("  point exists.")
        print("  Both conditions fired.")
        print("  would have been enough.")
    else:
        print("  Rule did not fire cleanly; no estimate quoted.")
        print("  Nothing concluded about the interior")
        print("  in either direction.")
    print("")

    print(bar)
    print("Limitations")
    print(bar)
    print("  L1  Non-adaptive sniffer (committed OLS; no boundary-crossing")
    print("      detector, no smoother). Transition times would localize q")
    print("      at bucket boundaries; that attack is not implemented,")
    print("      so each R2 is an upper bound on defense")
    print("      performance")
    print("      and a lower bound")
    print("      on the leak.")
    print("  L2  Two widths, not a sweep (both pinned to committed")
    print("      numbers); the shape of R2 against W_Q")
    print("      is not measured.")
    print("  L3  T=%.0fs per arm, one day. Inventory half-lives on this book"
          % T)
    print("      are 3-9h; sd(q) and")
    print("      R_direct could differ over seven days.")
    print("  L4  gamma held at the viability cap, not swept (sd(q) moves")
    print("      1.33x across five decades of gamma).")
    print("  L5  Cost in dollars not stated while the control")
    print("      arm loses money at 3.23 SE; R_direct is an")
    print("      inventory-dynamics quantity.")
    print("  L6  No committed module or default modified; the defense is")
    print("      a subclass.")
    print("")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
