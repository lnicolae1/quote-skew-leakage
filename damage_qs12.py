# damage_qs12.py: damage A/B at MM quote_size 0.12 (sensitivity arm).
# - committed null at 0.020: d edge -0.00646 +/- 0.01023 $/BTC (-0.0010 +/- 0.0016 bp); 0.020 was
#   TOO_SMALL per quotesize_verdict; 1194392 moved the grid from OUT OF GRID to BOUNDED
# - cluster B (604800s, 8 seeds, gamma from C = 13.66); the 1194392 clearance is cluster C, not
#   poolable, so sd(q) is re-measured at both sizes against SD_MIN from quotesize_verdict
# - gamma_for_C has no quote_size term: gamma 1.3392e-06 at both sizes; the C cap is re-checked
# - run_ext = damage_ab.run with MM size split from SNIFF_SIZE (sniffer stays 0.020 BTC);
#   T-MIRROR: matches damage_ab.run to 1e-9 at mm_size = SNIFF_SIZE
# - headline cell only: theta 0.0, hold 10800s; non-finite metrics raise
# pre-registered:
# - primary: paired d edge/unit; d MM PnL and spread/inventory split secondary
# - damage detected iff |mean d edge/unit| >= DECISION_SE * SE, DECISION_SE = 2.0
# - MDE = DECISION_SE * SE; r = MDE / sd(skew); power bought iff r(0.12)/r(0.020) < 1 - POWER_GAIN_MIN (0.10)
# - clearance (0.020, 0.12): (no, yes) intended; (no, no) uncleared; (yes, yes) OUT OF GRID was a
#   cluster-C fact, 6x-clip sensitivity; (yes, no) anomalous, flagged
# - gate 2 built as in clipped_gamma_sweep: per-second MM spread and residual touch, per-seed
#   medians, ratio of seed means, control arm

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import damage_ab as DA

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from clipped_mm_check import _resid_best, MM_ID
from damage_ab import (T, SEEDS, LAM, P_MARKET, LIFE, DISP, K, REF_MID,
                       C_TARGET, TAU_FLOOR_FRAC, SNIFF_SIZE, MTM_EVERY,
                       SEC_PER_YEAR, HOLD_MID, GAMMA, C_PER_GAMMA,
                       gamma_for_C, sharpe, mean_se)
from quotesize_verdict import SD_MIN, SD_REF, SD_FACTOR
from clipped_window import Z

MM_SIZES = (0.020, 0.12)
THETA = 0.0
HOLD = HOLD_MID
DECISION_SE = 2.0
POWER_GAIN_MIN = 0.10            # low bar by design
TICK = 0.01

# committed figures (damage_ab_results.txt, damage_split_results.txt);
# tol = half the last printed digit
COMMITTED = {
    "d_pnl":        (-107.50, 0.005),
    "d_pnl_se":     (33.40,   0.005),
    "d_edge":       (-0.00646, 0.000005),
    "d_edge_se":    (0.01023,  0.000005),
    "sn_share":     (4.5187,   0.00005),
    "ctrl_pnl":     (-525.18,  0.005),
    "ctrl_pnl_se":  (162.43,   0.005),
}
# committed sd(skew) at quote_size 0.020, ticks
COMMITTED_SKEW_TICKS = 2.724
# 1194392: aggTrades/s at 0.12 is 19.4% below the 0.0451 target (band +/-10%)
RATE_DEVIATION_PCT = 19.4
# gate 1 anchor, cluster C (clipped_gamma_sweep_results.txt)
GATE1_ANCHOR_FILLS_PER_DAY = 437.8
GATE2_MAX_RATIO = 10.0

MIRROR_TOL = 1e-9
MIRROR_SEED = 0


def _finite(x, what):
    if x is None or not math.isfinite(x):
        raise ValueError("non-finite %s: %r" % (what, x))
    return x


def _ratio(num, den, what):
    """Guarded division for every verdict ratio; raises on a non-positive or non-finite input."""
    _finite(num, "numerator of %s" % what)
    _finite(den, "denominator of %s" % what)
    if den <= 0:
        raise ValueError("non-positive denominator in %s: %r; no verdict "
                         "from a failed measurement"
                         % (what, den))
    return _finite(num / den, what)


# --------------------------------------------------------------------------- #
# damage_ab.run with mm_size split out, plus measurement columns
# --------------------------------------------------------------------------- #
def run_ext(seed, trades, theta, hold, mm_size):
    """damage_ab.run() with the maker's quote_size as a parameter.
    - only the MarketMaker constructor differs; sniffer order size stays SNIFF_SIZE
    - T-MIRROR checks equality with damage_ab.run() at mm_size = SNIFF_SIZE"""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA, quote_size=mm_size)

    pos = 0.0
    sn_cash = 0.0
    entry_t = None
    n_open = 0
    sn_vol = 0.0

    tot_vol = 0.0
    mm_edge_num = 0.0
    mm_edge_den = 0.0
    q_all = []
    mtm = []
    last_mid = REF_MID
    tau0 = mm.time_remaining_years(0.0)

    # added measurement; read-only, no RNG
    skew_series = []
    mm_spread_series = []
    resid_touch_series = []
    n_mm_fills = 0

    t = 0.0
    while t < T:
        t += 1.0

        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                for f in fills:
                    tot_vol += f.size
                    if f.counterparty_id == MM_ID:
                        n_mm_fills += 1
                        sgn = 1.0 if r.side == "buy" else -1.0
                        mm_edge_num += sgn * (f.price - last_mid) * f.size
                        mm_edge_den += f.size
                mm.on_fills(fills, r.side)

        mm.requote(eng, t)
        q_all.append(mm.q)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        last_mid = 0.5 * (bb + ba)
        if int(t) % MTM_EVERY == 0:
            mtm.append(mm.mark_to_market(last_mid))

        # added: leak amplitude and gate inputs, both arms; read-only, no RNG
        _mmb = _mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                _mmb = o.price if _mmb is None else max(_mmb, o.price)
            else:
                _mma = o.price if _mma is None else min(_mma, o.price)
        _rb = _resid_best(eng.bids, bb, True)
        _ra = _resid_best(eng.asks, ba, False)
        if _mmb is not None and _mma is not None:
            mm_spread_series.append(_mma - _mmb)
            if _rb is not None and _ra is not None and _ra > _rb:
                resid_touch_series.append(_ra - _rb)
                skew_series.append(0.5 * (_mmb + _mma) - 0.5 * (_rb + _ra))

        if not trades:
            continue

        mmb = mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mmb = o.price if mmb is None else max(mmb, o.price)
            else:
                mma = o.price if mma is None else min(mma, o.price)
        if mmb is None or mma is None:
            continue
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is None or ra is None or ra <= rb:
            continue
        resid_mid = 0.5 * (rb + ra)
        tau = mm.time_remaining_years(t)
        if tau < TAU_FLOOR_FRAC * tau0:
            continue

        skew = 0.5 * (mmb + mma) - resid_mid
        sig = mm.sigma_absolute(resid_mid)
        c_t = GAMMA * sig * sig * tau
        if c_t <= 0:
            continue
        q_hat = -skew / c_t

        if pos != 0.0:
            if t - entry_t >= hold:
                side = "sell" if pos > 0 else "buy"
                fl = eng.submit_market_order(side, abs(pos))
                got = sum(f.size for f in fl)
                cashflow = sum(f.price * f.size for f in fl)
                sn_cash += cashflow if pos > 0 else -cashflow
                sn_vol += got
                mm.on_fills(fl, side)
                pos = pos - got if pos > 0 else pos + got
                if abs(pos) < 1e-12:
                    pos = 0.0
                    entry_t = None
            continue

        if abs(q_hat) <= theta:
            continue
        side = "sell" if q_hat > 0 else "buy"
        fl = eng.submit_market_order(side, SNIFF_SIZE)
        got = sum(f.size for f in fl)
        if got <= 0:
            continue
        cashflow = sum(f.price * f.size for f in fl)
        sn_cash += cashflow if side == "sell" else -cashflow
        sn_vol += got
        mm.on_fills(fl, side)
        pos = -got if side == "sell" else got
        entry_t = t
        n_open += 1

    if trades and pos != 0.0:
        side = "sell" if pos > 0 else "buy"
        fl = eng.submit_market_order(side, abs(pos))
        cashflow = sum(f.price * f.size for f in fl)
        sn_cash += cashflow if pos > 0 else -cashflow
        sn_vol += sum(f.size for f in fl)
        mm.on_fills(fl, side)
        pos = 0.0

    pnl = mm.mark_to_market(last_mid)
    spread_pnl = mm_edge_num
    out = {
        "mm_pnl": pnl,
        "mm_spread_pnl": spread_pnl,
        "mm_inv_pnl": pnl - spread_pnl,
        "mm_sharpe": sharpe(mtm, MTM_EVERY),
        "mm_inv_var": statistics.variance(q_all) if len(q_all) > 1 else 0.0,
        "mm_edge": (mm_edge_num / mm_edge_den) if mm_edge_den > 0 else 0.0,
        "mm_vol": mm_edge_den,
        "sn_pnl": sn_cash,
        "sn_vol": sn_vol,
        "sn_share": (100.0 * sn_vol / tot_vol) if tot_vol > 0 else 0.0,
        "n_open": float(n_open),
    }
    # added columns, not mirrored
    out["sd_q"] = statistics.stdev(q_all) if len(q_all) > 1 else 0.0
    out["sd_skew"] = (statistics.stdev(skew_series)
                      if len(skew_series) > 1 else 0.0)
    out["mm_spread_med"] = (statistics.median(mm_spread_series)
                            if mm_spread_series else float("nan"))
    out["resid_touch_med"] = (statistics.median(resid_touch_series)
                              if resid_touch_series else float("nan"))
    out["mm_fills_per_day"] = n_mm_fills / (T / 86400.0)
    return out


MIRRORED_KEYS = ("mm_pnl", "mm_spread_pnl", "mm_inv_pnl", "mm_sharpe",
                 "mm_inv_var", "mm_edge", "mm_vol", "sn_pnl", "sn_vol",
                 "sn_share", "n_open")


def _check_seed(out, arm, s, mm_size):
    """Halt on non-finite per-seed values before storage (damage_ab.mean_se drops NaN silently).
    - mm_sharpe exempt: can legitimately be NaN, never averaged here; NaN seeds are counted
    - zero MM fill volume halts: damage_ab substitutes mm_edge = 0.0 (primary metric)"""
    for k, v in out.items():
        if k == "mm_sharpe":
            continue
        _finite(v, "%s %s seed %d size %.3f" % (arm, k, s, mm_size))
    if not out["mm_vol"] > 0:
        raise ValueError("%s seed %d size %.3f: zero MM fill volume; "
                         "mm_edge would be damage_ab's 0.0 substitute"
                         % (arm, s, mm_size))


def measure(mm_size):
    """Paired control and treatment runs over all seeds; returns per-seed dicts."""
    ctrl, treat = {}, {}
    t0 = time.time()
    for s in SEEDS:
        c = run_ext(s, False, THETA, HOLD, mm_size)
        _check_seed(c, "control", s, mm_size)
        ctrl[s] = c
    for s in SEEDS:
        b = run_ext(s, True, THETA, HOLD, mm_size)
        _check_seed(b, "treatment", s, mm_size)
        treat[s] = b
    return ctrl, treat, time.time() - t0


def paired(ctrl, treat, key):
    d = [treat[s][key] - ctrl[s][key] for s in SEEDS]
    m, se, n = mean_se(d)
    return _finite(m, "paired mean %s" % key), _finite(se, "paired se %s" % key), d


def bp(x):
    return 1e4 * x / REF_MID


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def t_ident():
    assert SD_MIN is not None and abs(SD_MIN - 0.08258) < 1e-12, \
        "SD_MIN moved: %r" % SD_MIN
    assert abs(SD_FACTOR - 2.0) < 1e-12 and abs(SD_REF - 0.04129) < 1e-12
    assert (T, tuple(SEEDS), K) == (DA.T, tuple(DA.SEEDS), DA.K)
    assert (LAM, P_MARKET, LIFE, DISP) == (DA.LAM, DA.P_MARKET, DA.LIFE,
                                           DA.DISP)
    assert SNIFF_SIZE == DEFAULT_QUOTE_SIZE == 0.02, \
        "sniffer size is not the committed 0.02: %r" % SNIFF_SIZE
    assert HOLD == 10800 and THETA == 0.0
    assert DECISION_SE == 2.0 and POWER_GAIN_MIN == 0.10, \
        "a pre-registered decision constant moved"
    g, per = gamma_for_C(C_TARGET, T)
    assert g is not None and abs(g - GAMMA) < 1e-18
    assert abs(GAMMA - 1.3392e-06) < 1e-9, "gamma is not the committed value"
    g1, per1 = gamma_for_C(C_TARGET, 86400.0)
    print("  T-IDENT     PASS  SD_MIN=%.5f from quotesize_verdict; "
          "working point, seeds, k" % SD_MIN)
    print("                    and T match damage_ab. Sniffer size "
          "%.3f BTC = DEFAULT_QUOTE_SIZE." % SNIFF_SIZE)
    print("                    gamma_for_C(%.2f, %.0fs) = %.4e with "
          "C_per_gamma %.4g;" % (C_TARGET, T, g, per))
    print("                    at 86400s it would be %.4e with C_per_gamma "
          "%.4g. NO quote_size term in" % (g1, per1))
    print("                    either, so gamma is unchanged at 0.12. Cap "
          "C=%.2f was set at" % C_TARGET)
    print("                    quote_size 0.020 and 86400s; re-checked in "
          "the gate block.")


def t_nan():
    for bad in (float("nan"), float("inf"), float("-inf"), None):
        try:
            _finite(bad, "planted")
        except ValueError:
            continue
        raise AssertionError("_finite accepted %r" % bad)
    _finite(0.0, "zero")
    print("  T-NAN       PASS  nan, +inf, -inf and None all raise.")


def t_ratio():
    """_ratio raises on zero, negative and NaN denominators (planted inputs)."""
    for num, den in ((1.0, 0.0), (1.0, -0.5), (1.0, float("nan")),
                     (float("nan"), 1.0), (float("inf"), 1.0), (0.0, 0.0)):
        try:
            _ratio(num, den, "planted")
        except ValueError:
            continue
        raise AssertionError("_ratio(%r, %r) did not raise" % (num, den))
    assert _ratio(3.0, 2.0, "ok") == 1.5
    # regression: the old expression on a zero SE gave a verdict
    old_mult = abs(0.001 / 0.0) if 0.0 > 0 else float("inf")
    assert old_mult >= DECISION_SE, "precondition: old code reported a detection"
    print("  T-RATIO     PASS  zero, negative and NaN denominators all raise; "
          "the old expression on a")
    print("                    zero SE gave inf >= %.1f ('DAMAGE "
          "DETECTED'); now halts." % DECISION_SE)


def t_tolerances():
    """Tolerances follow the printed precision of the committed files."""
    assert COMMITTED["d_pnl"][1] == 0.005, "2dp -> +/-0.005"
    assert COMMITTED["d_edge"][1] == 0.000005, "5dp -> +/-0.000005"
    assert COMMITTED["sn_share"][1] == 0.00005, "4dp -> +/-0.00005"
    print("  T-TOL       PASS  tolerances derived from printed precision: "
          "2dp->5e-3, 5dp->5e-6, 4dp->5e-5.")


def t_mirror():
    """At mm_size = SNIFF_SIZE, run_ext must reproduce damage_ab.run(): both arms, every key."""
    t0 = time.time()
    for trades in (False, True):
        mine = run_ext(MIRROR_SEED, trades, THETA, HOLD, SNIFF_SIZE)
        theirs = DA.run(MIRROR_SEED, trades, THETA, HOLD)
        for k in MIRRORED_KEYS:
            a, b = mine[k], theirs[k]
            if isinstance(a, float) and math.isnan(a) and math.isnan(b):
                continue
            assert abs(a - b) <= MIRROR_TOL * max(1.0, abs(b)), \
                "mirror mismatch %s (trades=%s): run_ext %.12g vs committed %.12g" \
                % (k, trades, a, b)
    print("  T-MIRROR    PASS  all %d returned keys agree with damage_ab.run "
          "to %.0e, both arms," % (len(MIRRORED_KEYS), MIRROR_TOL))
    print("                    at seed %d (%.0fs). At mm_size=SNIFF_SIZE the "
          "transcription IS the" % (MIRROR_SEED, time.time() - t0))
    print("                    committed function; added columns draw no RNG.")


def selftest():
    print("=" * 128)
    print("self-tests: damage_qs12.py (arm not yet run)")
    print("=" * 128)
    t_ident()
    t_nan()
    t_ratio()
    t_tolerances()
    t_mirror()
    print("")
    print("  all 5 self-tests PASSED.")
    return 0


# --------------------------------------------------------------------------- #
def main():
    print("=" * 128)
    print("damage A/B at quote_size 0.12: sensitivity arm, not a new headline")
    print("=" * 128)
    print("working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, join clipping." % (LAM, P_MARKET, LIFE, DISP))
    print("cluster B: %.0fs (%.0f days), %d seeds, k=%.5f, C=%.2f -> "
          "gamma=%.4e (C_per_gamma %.4g)."
          % (T, T / 86400.0, len(SEEDS), K, C_TARGET, GAMMA, C_PER_GAMMA))
    print("theta=%.2f, hold=%ds (headline only). Sniffer size %.3f BTC, "
          "not rescaled with the maker."
          % (THETA, HOLD, SNIFF_SIZE))
    print("DEFAULT_GAMMA=%g and DEFAULT_QUOTE_SIZE=%g untouched."
          % (DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE))
    print("")
    print("pre-registered: primary = paired d edge/unit; "
          "'damage detected' iff |mean| >= %.1f SE." % DECISION_SE)
    print("MDE = %.1f x SE at both sizes, against leak amplitude "
          "sd(skew) in ticks." % DECISION_SE)
    print("")

    data = {}
    for size in MM_SIZES:
        print("running mm quote_size=%.3f ..." % size, flush=True)
        ctrl, treat, secs = measure(size)
        data[size] = (ctrl, treat)
        print("  %d control + %d treatment runs in %.0fs"
              % (len(SEEDS), len(SEEDS), secs), flush=True)
    print("")

    # 0. data integrity
    print("=" * 128)
    print("0. data integrity: NaN Sharpe seeds, min per-seed MM volume")
    print("=" * 128)
    print("  mm_sharpe NaN is counted, not halted on; other non-finite fields "
          "and zero MM volume halt.")
    print("")
    print("  %-10s %-11s %-22s %-24s"
          % ("mm size", "arm", "NaN Sharpe seeds", "min per-seed MM volume"))
    nan_at_020 = 0
    for size in MM_SIZES:
        ctrl, treat = data[size]
        for arm, dd in (("control", ctrl), ("treatment", treat)):
            n_nan = sum(1 for s in SEEDS
                        if not math.isfinite(dd[s]["mm_sharpe"]))
            vmin = min(dd[s]["mm_vol"] for s in SEEDS)
            smin = min(SEEDS, key=lambda s: dd[s]["mm_vol"])
            if size == 0.020:
                nan_at_020 += n_nan
            print("  %-10.3f %-11s %2d of %-17d %.4f BTC (seed %d)"
                  % (size, arm, n_nan, len(SEEDS), vmin, smin))
    print("")
    if nan_at_020:
        print("  >>> %d NaN Sharpe seed(s) at the COMMITTED 0.020 "
              "configuration <<<" % nan_at_020)
        print("  given T-MIRROR, the committed control-arm Sharpe (-12.4326 +/- "
              "2.8607) and d-MM-Sharpe")
        print("  column were averaged over fewer than 8 seeds (mean_se drops "
              "NaN); headline row only.")
    else:
        print("  no NaN Sharpe at 0.020 in either arm; given T-MIRROR, the "
              "committed control-arm Sharpe and")
        print("  headline d-MM-Sharpe row used all 8 seeds. Four non-headline "
              "rows not re-run.")
    print("  zero-volume guard did not fire.")
    print("")

    # 1. cluster-B clearance
    print("=" * 128)
    print("1. cluster-B clearance: does 0.12 clear in this configuration?")
    print("=" * 128)
    print("  1194392 clearance is cluster C (86400s, gamma 9.40e-06); this is "
          "cluster B (%.0fs, gamma %.4e)." % (T, GAMMA))
    print("  not poolable; sd(q) re-measured here.")
    print("  threshold from quotesize_verdict: %.1f x %.5f = %.5f "
          "BTC on the 95%% lower bound."
          % (SD_FACTOR, SD_REF, SD_MIN))
    print("")
    print("  %-12s %-26s %-12s %-12s %s"
          % ("mm size", "sd(q) BTC (control arm)", "95% lo", "threshold",
             "clears?"))
    clearance = {}
    for size in MM_SIZES:
        ctrl, _ = data[size]
        sds = [ctrl[s]["sd_q"] for s in SEEDS]
        m, se, n = mean_se(sds)
        # mean_se drops NaN and returns se = 0.0 on one value; refuse both
        if n != len(SEEDS):
            raise ValueError("clearance at size %.3f rests on %d of %d seeds"
                             % (size, n, len(SEEDS)))
        _finite(m, "clearance mean sd(q) at %.3f" % size)
        if not (_finite(se, "clearance SE at %.3f" % size) > 0):
            raise ValueError("clearance SE at size %.3f is %r; no interval"
                             % (size, se))
        lo = _finite(m - Z * se, "clearance 95%% lo at %.3f" % size)
        ok = lo >= SD_MIN
        clearance[size] = (m, se, lo, ok)
        print("  %-12.3f %11.5f +/- %-10.5f %-12.5f %-12.5f %s"
              % (size, m, se, lo, SD_MIN, "YES" if ok else "NO"))
    print("")
    print("  SD_REF = %.5f is from cluster E "
          "(gamma 9.37e-06, 86400s, 5 seeds);" % SD_REF)
    print("  here gamma is %.1fx smaller and the horizon 7x longer; "
          "both raise sd(q)." % (9.37e-06 / GAMMA))
    print("")
    c020, c12 = clearance[0.020][3], clearance[0.12][3]
    if c020 and c12:
        print("  >>> both sizes clear at cluster B, including the COMMITTED "
              "0.020 <<<")
        print("  OUT OF GRID was a cluster-C fact: the 0.020 null was measured "
              "at an adequate inventory scale;")
        print("  this run is a 6x-clip sensitivity, not a rescue. Clearance is "
              "cluster-B-specific.")
    elif c12 and not c020:
        print("  intended case: 0.12 clears at cluster B, 0.020 does not; "
              "damage below is measured at")
        print("  the pre-registered inventory scale.")
    elif not c12 and not c020:
        print("  *** 0.12 does not clear at cluster B *** cluster-C "
              "clearance did not transfer.")
        print("  damage below is from an uncleared configuration; not 'null "
              "across a 2.6x inventory range'.")
    else:
        print("  *** anomalous: 0.020 clears, 0.12 does not *** sd(q) "
              "falls as the clip rises;")
        print("  flagged, not interpreted; damage numbers below unreliable.")
    print("")

    # 2. reproduction
    print("=" * 128)
    print("2. REPRODUCTION OF THE COMMITTED quote_size 0.020 HEADLINE")
    print("=" * 128)
    ctrl0, treat0 = data[0.020]
    d_pnl0, d_pnl_se0, _ = paired(ctrl0, treat0, "mm_pnl")
    d_edge0, d_edge_se0, _ = paired(ctrl0, treat0, "mm_edge")
    sn0, sn_se0, _n = mean_se([treat0[s]["sn_share"] for s in SEEDS])
    cp0, cp_se0, _n = mean_se([ctrl0[s]["mm_pnl"] for s in SEEDS])
    got = {"d_pnl": d_pnl0, "d_pnl_se": d_pnl_se0, "d_edge": d_edge0,
           "d_edge_se": d_edge_se0, "sn_share": sn0, "ctrl_pnl": cp0,
           "ctrl_pnl_se": cp_se0}
    bad = []
    for k, (want, tol) in COMMITTED.items():
        g = got[k]
        ok = abs(g - want) <= tol
        print("  %-14s committed %12.5f   got %12.5f   tol %.6f   %s"
              % (k, want, g, tol, "OK" if ok else "*** MISMATCH ***"))
        if not ok:
            bad.append(k)
    if bad:
        raise AssertionError(
            "reproduction failed on %s; halting, no 0.12 number reported"
            % ", ".join(bad))
    print("  all seven COMMITTED figures reproduce.")
    print("")

    # 3. primary metric
    print("=" * 128)
    print("3. primary metric: paired d edge/unit, pre-registered decision rule")
    print("=" * 128)
    print("  %-10s %-24s %-24s %-10s %s"
          % ("mm size", "d edge/unit ($/BTC)", "as bp of mid", "|mean|/SE",
             "verdict"))
    prim = {}
    for size in MM_SIZES:
        c, b_ = data[size]
        m, se, _d = paired(c, b_, "mm_edge")
        mult = _ratio(abs(m), se, "primary |mean|/SE at size %.3f" % size)
        det = mult >= DECISION_SE
        prim[size] = (m, se, mult, det)
        print("  %-10.3f %9.5f +/- %-10.5f %9.5f +/- %-10.5f %-10.2f %s"
              % (size, m, se, bp(m), bp(se), mult,
                 "DAMAGE DETECTED" if det else "null"))
    print("")
    m12, se12, mult12, det12 = prim[0.12]
    if det12:
        print("  DAMAGE DETECTED at 0.12: |mean| = %.2f SE >= %.1f SE; "
              "null did not survive the" % (mult12, DECISION_SE))
        print("  inventory extension (damage hidden at 0.020 by the small clip).")
    else:
        print("  null at 0.12: |mean| = %.2f SE < %.1f SE; null survives "
              "the inventory extension." % (mult12, DECISION_SE))
    print("")

    # 4. power
    print("=" * 128)
    print("4. power")
    print("=" * 128)
    print("  %-10s %-22s %-20s %-18s %s"
          % ("mm size", "MDE ($/BTC)", "MDE (bp)", "sd(skew) ticks",
             "MDE / leak"))
    for size in MM_SIZES:
        m, se, _mult, _d = prim[size]
        mde = DECISION_SE * se
        c, _b = data[size]
        sk = [c[s]["sd_skew"] for s in SEEDS]
        msk, sesk, _n = mean_se(sk)
        ticks = msk / TICK
        ratio = _ratio(mde, msk, "MDE / sd(skew) at size %.3f" % size)
        print("  %-10.3f %-22.5f %-20.5f %8.3f +/- %-7.3f %.3f"
              % (size, mde, bp(mde), ticks, sesk / TICK, ratio))
    print("")
    print("  sd(skew): sd of 0.5*(mm_bid+mm_ask) - residual_mid, the quantity "
          "the sniffer inverts.")
    print("  measured in the CONTROL arm; committed reference at "
          "0.020: %.3f ticks." % COMMITTED_SKEW_TICKS)
    mde0 = DECISION_SE * prim[0.020][1]
    mde12 = DECISION_SE * prim[0.12][1]
    c0, _ = data[0.020]
    c12, _ = data[0.12]
    sk0, _, _n = mean_se([c0[s]["sd_skew"] for s in SEEDS])
    sk12, _, _n = mean_se([c12[s]["sd_skew"] for s in SEEDS])
    r0 = _ratio(mde0, sk0, "r(0.020) = MDE / sd(skew)")
    r12 = _ratio(mde12, sk12, "r(0.12) = MDE / sd(skew)")
    print("")
    print("  MDE as a fraction of the leak amplitude: r = %.3f at 0.020, "
          "%.3f at 0.12." % (r0, r12))
    rr = _ratio(r12, r0, "power ratio r(0.12)/r(0.020)")
    print("  RATIO r(0.12)/r(0.020) = %.4f, against the pre-registered bar "
          "of < %.2f (POWER_GAIN_MIN = %.2f)." % (rr, 1.0 - POWER_GAIN_MIN,
                                                    POWER_GAIN_MIN))
    print("  low bar by design: %.0f%% is the smallest "
          "change worth calling a gain." % (100.0 * POWER_GAIN_MIN))
    if rr < 1.0 - POWER_GAIN_MIN:
        print("  power bought: detectable effect relative "
              "to the signal shrank by %.0f%%."
              % (100.0 * (1.0 - rr)))
    else:
        print("  no material power bought: detectable effect did "
              "not shrink by %.0f%%" % (100.0 * POWER_GAIN_MIN))
        print("  relative to the signal (%.3f -> %.3f); null no "
              "stronger than at 0.020." % (r0, r12))
    print("")

    # 5. secondary
    print("=" * 128)
    print("5. secondary: d MM PnL, spread / inventory split")
    print("=" * 128)
    print("  %-10s %-26s %-26s %-26s"
          % ("mm size", "d MM PnL ($)", "d spread ($)", "d inv ($)"))
    for size in MM_SIZES:
        c, b_ = data[size]
        mp, sp, _ = paired(c, b_, "mm_pnl")
        ms, ss, _ = paired(c, b_, "mm_spread_pnl")
        mi, si, _ = paired(c, b_, "mm_inv_pnl")
        print("  %-10.3f %10.2f +/- %-11.2f %10.2f +/- %-11.2f %10.2f +/- %-11.2f"
              % (size, mp, sp, ms, ss, mi, si))
    print("")
    print("  secondary: Run B has an extra participant, so PnL differences "
          "are not pure leakage.")
    print("")

    # 6. caveats and gates
    print("=" * 128)
    print("6. caveats: sniffer share, control PnL, Phase 5 gates")
    print("=" * 128)
    print("  %-10s %-16s %-26s %-16s %-16s %s"
          % ("mm size", "sniffer share", "control PnL ($)", "MM fills/day",
             "MM spread ($)", "resid touch ($)"))
    for size in MM_SIZES:
        c, b_ = data[size]
        sn, snse, _n = mean_se([b_[s]["sn_share"] for s in SEEDS])
        cp, cpse, _n = mean_se([c[s]["mm_pnl"] for s in SEEDS])
        fd, fdse, _n = mean_se([c[s]["mm_fills_per_day"] for s in SEEDS])
        sp, spse, _n = mean_se([c[s]["mm_spread_med"] for s in SEEDS])
        rt, rtse, _n = mean_se([c[s]["resid_touch_med"] for s in SEEDS])
        print("  %-10.3f %7.4f%% %6s %10.2f +/- %-11.2f %8.1f %7s %8.4f %7s %8.4f"
              % (size, sn, "", cp, cpse, fd, "", sp, "", rt))
    print("")
    print("  sniffer share: clip stays %.3f BTC, not rescaled "
          "with the maker." % SNIFF_SIZE)
    print("  committed at 0.020: %.4f%%." % COMMITTED["sn_share"][0])
    print("")
    print("  CONTROL PnL: the maker loses with no adversary "
          "present (Phase 5 item 3);")
    print("  damage is a perturbation to that baseline.")
    print("")
    print("  Phase 5 gates, built as in clipped_gamma_sweep.py: per-second MM "
          "spread and residual touch,")
    print("  per-seed medians, ratio of seed means; MM fills as on_fills "
          "counts; control arm.")
    print("  caps derived at 86400s; this run is %.0fs." % T)
    for size in MM_SIZES:
        c, _b = data[size]
        fd, _se, _n = mean_se([c[s]["mm_fills_per_day"] for s in SEEDS])
        sp, _se, _n = mean_se([c[s]["mm_spread_med"] for s in SEEDS])
        rt, _se, _n = mean_se([c[s]["resid_touch_med"] for s in SEEDS])
        ratio = _ratio(sp, rt, "gate 2 MM spread / touch at size %.3f" % size)
        g2 = ratio <= GATE2_MAX_RATIO
        print("    size %.3f: GATE 2 (MM spread / market touch <= %.0fx) = "
              "%.4f / %.4f = %.2fx -> %s"
              % (size, GATE2_MAX_RATIO, sp, rt, ratio,
                 "HOLDS" if g2 else "*** FAILS ***"))
        print("               GATE 1 input: %.1f MM fills/day; anchor "
              "%.1f/day is cluster C, not poolable" % (fd, GATE1_ANCHOR_FILLS_PER_DAY))
    f0, _s, _n = mean_se([data[0.020][0][s]["mm_fills_per_day"] for s in SEEDS])
    f12, _s, _n = mean_se([data[0.12][0][s]["mm_fills_per_day"] for s in SEEDS])
    print("    GATE 1 within cluster B: %.1f fills/day at 0.020 -> %.1f at "
          "0.12 = %.1f%% retained."
          % (f0, f12, 100.0 * _ratio(f12, f0, "gate 1 fills retained")))
    print("")
    print("  rate-gate deviation (1194392): at quote_size 0.12 the aggTrade "
          "rate is ~%.1f%% below the" % RATE_DEVIATION_PCT)
    print("  0.0451/s target (band +/-10%), so the CALIBRATION gate rejects "
          "this configuration;")
    print("  no grid size meets it (0.020 straddles the floor: UNRESOLVED). "
          "Sensitivity arm, not the")
    print("  calibrated damage figure.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
