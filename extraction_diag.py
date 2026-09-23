# extraction_diag.py: why is extraction null? structural null vs strategy null (diagnostic, not a fix)
# - cluster B, maker sizes 0.020 and 0.12, 8 seeds; transcribes damage_qs12.run_ext plus read-only captures
# - (A) who the sniffer trades against: share of its fill volume by counterparty
# - (B) does true q predict future mid moves: slope over non-overlapping windows, SE across seeds
# - (C) top-minus-bottom quintile move (TMB) vs measured round-trip cost of a market order
# - pre-registered:
#   (P1) predicted slope sign: negative
#   (P2) per horizon, 95% bounds (clipped_window.Z): BELOW iff |TMB| upper < cost lower; ABOVE iff |TMB| lower > cost upper; else STRADDLE
#        STRATEGY NULL iff any horizon ABOVE; STRUCTURAL NULL iff every horizon BELOW; else INCONCLUSIVE
#        overall: STRATEGY if either size is; STRUCTURAL if both are; else INCONCLUSIVE
#   (P3) companion: same rule with TMB and its SE halved (one tail per trade); reported, not substituted
#   (P4) power: STRUCTURAL NULL unreachable at 8 seeds for h >= 900 s
#   (P5) profit is not damage; (A) bounds what can come from the maker
# - limitation L1: Stambaugh bias (q persistent, innovations correlated with the mid)

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import damage_qs12 as QS

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_SIGMA
from matching_engine import MatchingEngine
from clipped_mm_check import _resid_best, MM_ID
from damage_ab import (T, SEEDS, LAM, P_MARKET, LIFE, DISP, K, REF_MID,
                       TAU_FLOOR_FRAC, SNIFF_SIZE, MTM_EVERY, SEC_PER_YEAR,
                       GAMMA, sharpe, mean_se)
from damage_qs12 import _finite, _ratio, MM_SIZES, THETA, HOLD
from clipped_window import Z

HORIZONS = (60, 300, 900, 3600, 10800)
SAMPLE = 60
PREDICTED_SIGN = -1
COMPANION_FACTOR = 0.5
N_SEEDS = len(SEEDS)

# committed damage_qs12 figures; tolerance from printed precision (5 dp -> 5e-6, 4 dp -> 5e-5, 2 dp -> 5e-3)
COMMITTED = {
    0.020: {"d_edge": (-0.00646, 5e-6), "d_edge_se": (0.01023, 5e-6),
            "sd_q": (0.09099, 5e-6), "sd_q_se": (0.00798, 5e-6),
            "sn_share": (4.5187, 5e-5),
            "ctrl_pnl": (-525.18, 5e-3), "ctrl_pnl_se": (162.43, 5e-3)},
    0.12: {"d_edge": (0.00563, 5e-6), "d_edge_se": (0.00629, 5e-6),
           "sd_q": (0.26929, 5e-6), "sd_q_se": (0.00871, 5e-6),
           "sn_share": (4.5186, 5e-5),
           "ctrl_pnl": (-1058.55, 5e-3), "ctrl_pnl_se": (388.57, 5e-3)},
}

MIRROR_SEED = 0
MIRROR_SIZE = 0.12
MIRROR_TOL = 1e-9
CP_BUCKETS = ("MM", "noise", "informed", "seed", "other")


# helpers
def bucket(agent_id):
    a = str(agent_id)
    if a == MM_ID:
        return "MM"
    if a.startswith("noise"):
        return "noise"
    if a == "informed":
        return "informed"
    if a == "seed":
        return "seed"
    return "other"


def walk_cost(eng, size, mid):
    """Round-trip cost in bp of a market order of size (buy VWAP up the asks minus sell VWAP down the bids, over the mid); read-only."""
    def vwap(book, ascending):
        need, spent = size, 0.0
        for p in sorted(book, reverse=not ascending):
            for o in book[p]:
                take = o.size if o.size < need else need
                spent += take * p
                need -= take
                if need <= 1e-12:
                    break
            if need <= 1e-12:
                break
        if need > 1e-12:
            raise ValueError("book too thin to fill %.4f BTC -- cost "
                             "undefined, refusing to invent one" % size)
        return _ratio(spent, size, "VWAP")
    va = vwap(eng.asks, True)
    vb = vwap(eng.bids, False)
    return _ratio(1e4 * (va - vb), mid, "round-trip cost / mid")


def strict_mean_se(values, what):
    """Mean/SE over all N_SEEDS finite values, positive SE required."""
    if len(values) != N_SEEDS:
        raise ValueError("%s: %d values, expected %d" % (what, len(values),
                                                        N_SEEDS))
    for v in values:
        _finite(v, what)
    m, se, n = mean_se(values)
    if n != N_SEEDS:
        raise ValueError("%s: mean_se kept %d of %d" % (what, n, N_SEEDS))
    _finite(m, "mean of " + what)
    if not _finite(se, "SE of " + what) > 0:
        raise ValueError("%s: SE is %r -- no interval, no verdict" % (what, se))
    return m, se


def windows(samples, h):
    """Non-overlapping [t, t+h] windows: x = q at start, y = mid change in bp."""
    xs, ys = [], []
    t = SAMPLE
    while t + h <= T:
        a, b = samples.get(t), samples.get(t + h)
        if a is not None and b is not None:
            xs.append(a[0])
            ys.append(_ratio(1e4 * (b[1] - a[1]), a[1], "return / mid_t"))
        t += h
    return xs, ys


def ols(xs, ys, what):
    """Slope, conventional SE, R^2, n; guarded divisions."""
    n = len(xs)
    if n < 3:
        raise ValueError("%s: %d windows, cannot regress" % (what, n))
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = _ratio(sxy, sxx, what + " slope")
    ssr = sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(_ratio(_ratio(ssr, n - 2, what + " resid var"), sxx,
                          what + " slope var"))
    if not se > 0:
        raise ValueError("%s: slope SE is exactly zero -- a degenerate fit, "
                         "not a measurement" % what)
    r2 = 1.0 - _ratio(ssr, syy, what + " R2")
    return b, se, r2, n


def tmb(xs, ys, what):
    """Top-quintile minus bottom-quintile mean future return, bp."""
    n = len(xs)
    k = n // 5
    if k < 2:
        raise ValueError("%s: quintile of %d windows is too small" % (what, k))
    order = sorted(range(n), key=lambda i: xs[i])
    bot = [ys[i] for i in order[:k]]
    top = [ys[i] for i in order[-k:]]
    return statistics.fmean(top) - statistics.fmean(bot)


def horizon_call(m, se, c, se_c, factor):
    mag = abs(m) * factor
    s = se * factor
    mag_hi = mag + Z * s
    mag_lo = max(0.0, mag - Z * s)
    c_hi = c + Z * se_c
    c_lo = c - Z * se_c
    for v, nm in ((mag_hi, "mag_hi"), (mag_lo, "mag_lo"), (c_hi, "c_hi"),
                  (c_lo, "c_lo")):
        _finite(v, nm)
    if mag_hi < c_lo:
        call = "BELOW"
    elif mag_lo > c_hi:
        call = "ABOVE"
    else:
        call = "STRADDLE"
    return call, mag, mag_lo, mag_hi, c_lo, c_hi


def size_verdict(calls):
    """Roll up per-horizon calls; STRATEGY takes precedence."""
    above = [h for h, c in calls if c == "ABOVE"]
    if above:
        return "STRATEGY NULL", above
    if all(c == "BELOW" for _h, c in calls):
        return "STRUCTURAL NULL", []
    return "INCONCLUSIVE", [h for h, c in calls if c == "STRADDLE"]


def overall_verdict(per_size):
    vs = list(per_size.values())
    if any(v == "STRATEGY NULL" for v in vs):
        return "STRATEGY NULL"
    if all(v == "STRUCTURAL NULL" for v in vs):
        return "STRUCTURAL NULL"
    return "INCONCLUSIVE"


# transcription: damage_qs12.run_ext plus capture
def run_diag(seed, trades, theta, hold, mm_size):
    """damage_qs12.run_ext with read-only captures (control: q, mid, cost; treatment: sniffer fill volume by counterparty)."""
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

    skew_series = []
    mm_spread_series = []
    resid_touch_series = []
    n_mm_fills = 0

    # capture, read-only
    samples = {}
    cp_vol = {b: 0.0 for b in CP_BUCKETS}

    def cap_fills(fl):
        for f in fl:
            cp_vol[bucket(f.counterparty_id)] += f.size

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

        # capture (control arm): q, mid, cost at the MTM point
        if not trades and int(t) % SAMPLE == 0:
            samples[int(t)] = (mm.q, last_mid,
                               walk_cost(eng, SNIFF_SIZE, last_mid))

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
                cap_fills(fl)
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
        cap_fills(fl)
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
        cap_fills(fl)
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
    out["sd_q"] = statistics.stdev(q_all) if len(q_all) > 1 else 0.0
    out["sd_skew"] = (statistics.stdev(skew_series)
                      if len(skew_series) > 1 else 0.0)
    out["mm_spread_med"] = (statistics.median(mm_spread_series)
                            if mm_spread_series else float("nan"))
    out["resid_touch_med"] = (statistics.median(resid_touch_series)
                              if resid_touch_series else float("nan"))
    out["mm_fills_per_day"] = n_mm_fills / (T / 86400.0)
    return out, {"samples": samples, "cp_vol": cp_vol}


def check_run(out, cap, arm, s, size):
    """Halt on non-finite values (except mm_sharpe), a zero-volume maker, or a non-finite capture."""
    for k, v in out.items():
        if k == "mm_sharpe":
            continue
        _finite(v, "%s %s seed %d size %.3f" % (arm, k, s, size))
    if not out["mm_vol"] > 0:
        raise ValueError("%s seed %d size %.3f: zero maker volume" %
                         (arm, s, size))
    for tt, (q, mid, c) in cap["samples"].items():
        _finite(q, "q at t=%d" % tt)
        _finite(mid, "mid at t=%d" % tt)
        _finite(c, "cost at t=%d" % tt)
    for b, v in cap["cp_vol"].items():
        _finite(v, "counterparty volume %s" % b)


# self-tests
def t_ident():
    assert _finite is QS._finite and _ratio is QS._ratio
    assert (T, tuple(SEEDS), K, GAMMA) == (QS.T, tuple(QS.SEEDS), QS.K,
                                          QS.GAMMA)
    assert abs(GAMMA - 1.3392e-06) < 1e-9 and T == 604800.0
    assert SNIFF_SIZE == 0.02 and THETA == 0.0 and HOLD == 10800
    assert tuple(MM_SIZES) == (0.020, 0.12)
    assert all(h % SAMPLE == 0 for h in HORIZONS)
    assert PREDICTED_SIGN == -1 and COMPANION_FACTOR == 0.5 and Z == 1.96
    print("  T-IDENT     PASS  _finite/_ratio are damage_qs12's; T, seeds, k, "
          "gamma, sniffer size, theta,")
    print("                    hold and both maker sizes equal the committed "
          "cluster-B values; every")
    print("                    horizon is a multiple of the 60s sample; "
          "pre-registered constants fixed.")


def t_guards():
    for num, den in ((1.0, 0.0), (1.0, -1.0), (1.0, float("nan")),
                     (float("inf"), 1.0)):
        try:
            _ratio(num, den, "planted")
        except ValueError:
            continue
        raise AssertionError("_ratio(%r, %r) did not raise" % (num, den))
    for bad in ([1.0] * (N_SEEDS - 1), [1.0] * N_SEEDS,
                [float("nan")] + [1.0] * (N_SEEDS - 1)):
        try:
            strict_mean_se(bad, "planted")
        except ValueError:
            continue
        raise AssertionError("strict_mean_se accepted %r" % bad)
    print("  T-guards    PASS  zero/negative/NaN/inf denominators raise; a "
          "7-seed mean, a zero-SE mean and")
    print("                    a mean containing NaN are all refused.")


def t_ols():
    xs = [float(i) for i in range(50)]
    ys = [3.0 - 0.5 * x for x in xs]
    try:
        ols(xs, ys, "planted exact")
        raise AssertionError("exact fit should refuse (zero residual SE)")
    except ValueError:
        pass
    ys2 = [3.0 - 0.5 * x + (0.3 if i % 2 else -0.3) for i, x in enumerate(xs)]
    b, se, r2, n = ols(xs, ys2, "planted")
    assert abs(b - (-0.5)) < 0.01 and 0.99 < r2 < 1.0 and n == 50
    mx = statistics.fmean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    my = statistics.fmean(ys2)
    ssr = sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys2))
    assert abs(se - math.sqrt(ssr / (n - 2) / sxx)) < 1e-12
    print("  T-OLS       PASS  recovers a planted slope of -0.5 (got %.4f, "
          "R2 %.4f); SE matches the" % (b, r2))
    print("                    textbook formula; a zero-residual fit is "
          "refused rather than given SE 0.")


def t_tmb():
    xs = [float(i) for i in range(100)]
    ys = [x * 0.1 for x in xs]
    got = tmb(xs, ys, "planted")
    want = statistics.fmean(ys[80:]) - statistics.fmean(ys[:20])
    assert abs(got - want) < 1e-12 and got > 0
    xs_r = list(reversed(xs))
    ys_r = list(reversed(ys))
    assert abs(tmb(xs_r, ys_r, "shuffled") - want) < 1e-12
    print("  T-TMB       PASS  top-minus-bottom quintile = %.4f on planted "
          "data, invariant to input order." % got)


def t_windows():
    samples = {t: (float(t), 100.0 + 0.001 * t, 0.5)
               for t in range(SAMPLE, int(T) + 1, SAMPLE)}
    for h in HORIZONS:
        xs, ys = windows(samples, h)
        expect = (int(T) - SAMPLE) // h
        assert len(xs) == expect, (h, len(xs), expect)
        starts = [int(x) for x in xs]
        assert all(b - a == h for a, b in zip(starts, starts[1:])), h
    del samples[SAMPLE + 300]
    xs, _ys = windows(samples, 300)
    assert len(xs) == (int(T) - SAMPLE) // 300 - 2
    print("  T-windows   PASS  window counts equal floor((T-60)/h) at every "
          "horizon; starts are exactly")
    print("                    h apart (non-overlapping); a missing sample "
          "drops exactly the 2 windows it ends/starts.")


def t_walk():
    eng = MatchingEngine()
    eng.add_limit_order("sell", 100.02, 0.010, "a1")
    eng.add_limit_order("sell", 100.05, 0.050, "a2")
    eng.add_limit_order("buy", 99.98, 0.015, "b1")
    eng.add_limit_order("buy", 99.90, 0.050, "b2")
    before = {oid: o.size for oid, o in eng.orders.items()}
    mid = 100.0
    got = walk_cost(eng, 0.020, mid)
    va = (0.010 * 100.02 + 0.010 * 100.05) / 0.020
    vb = (0.015 * 99.98 + 0.005 * 99.90) / 0.020
    want = 1e4 * (va - vb) / mid
    assert abs(got - want) < 1e-9, (got, want)
    after = {oid: o.size for oid, o in eng.orders.items()}
    assert before == after, "walk_cost MUTATED the book"
    try:
        walk_cost(eng, 1.0, mid)
        raise AssertionError("an unfillable size should raise")
    except ValueError:
        pass
    print("  T-walk      PASS  a hand-built book walks to %.4f bp, matching "
          "the VWAP arithmetic;" % got)
    print("                    every order size is unchanged afterwards, and "
          "an unfillable size raises.")


def t_verdict():
    assert horizon_call(0.1, 0.05, 1.0, 0.01, 1.0)[0] == "BELOW"
    assert horizon_call(5.0, 0.5, 1.0, 0.01, 1.0)[0] == "ABOVE"
    assert horizon_call(1.0, 0.5, 1.0, 0.01, 1.0)[0] == "STRADDLE"
    assert horizon_call(-5.0, 0.5, 1.0, 0.01, 1.0)[0] == "ABOVE"
    # companion case: ABOVE in full, STRADDLE when halved
    assert horizon_call(2.2, 0.2, 1.0, 0.01, 1.0)[0] == "ABOVE"
    assert horizon_call(2.2, 0.2, 1.0, 0.01, 0.5)[0] == "STRADDLE"
    assert size_verdict([(60, "BELOW"), (300, "BELOW")]) == \
        ("STRUCTURAL NULL", [])
    assert size_verdict([(60, "BELOW"), (300, "ABOVE"), (900, "STRADDLE")]) \
        == ("STRATEGY NULL", [300])
    assert size_verdict([(60, "BELOW"), (300, "STRADDLE")]) == \
        ("INCONCLUSIVE", [300])
    assert overall_verdict({0.02: "STRATEGY NULL",
                            0.12: "STRUCTURAL NULL"}) == "STRATEGY NULL"
    assert overall_verdict({0.02: "STRUCTURAL NULL",
                            0.12: "STRUCTURAL NULL"}) == "STRUCTURAL NULL"
    assert overall_verdict({0.02: "STRUCTURAL NULL",
                            0.12: "INCONCLUSIVE"}) == "INCONCLUSIVE"
    print("  T-VERDICT   PASS  BELOW / ABOVE / STRADDLE on hand cases, a "
          "negative move counted by magnitude,")
    print("                    the companion halving flips an ABOVE to a "
          "STRADDLE, and both roll-ups fire.")


def t_power():
    """(P4) benchmark SE of TMB per horizon, before any data."""
    print("  T-power     (P4) benchmark, from DEFAULT_SIGMA=%.4f and window "
          "counts only; no data:" % DEFAULT_SIGMA)
    out = {}
    for h in HORIZONS:
        sig_h = DEFAULT_SIGMA * math.sqrt(h / SEC_PER_YEAR) * 1e4
        n = (int(T) - SAMPLE) // h
        se = sig_h * math.sqrt(10.0 / n) / math.sqrt(N_SEEDS)
        out[h] = Z * se
        print("                    h=%5ds  sigma_h=%7.3f bp  n=%5d windows/seed"
              "  expected 95%% half-width %7.3f bp" % (h, sig_h, n, Z * se))
    assert out[900] > 1.0 and out[10800] > 10.0
    print("                    Against a cost near 0.4-0.6 bp, h>=900 "
          "cannot bound below")
    print("                    the cost: STRUCTURAL NULL is unreachable at %d "
          "seeds under (P2)." % N_SEEDS)


def t_mirror():
    """run_diag reproduces damage_qs12.run_ext."""
    t0 = time.time()
    keys = None
    for trades in (False, True):
        mine, cap = run_diag(MIRROR_SEED, trades, THETA, HOLD, MIRROR_SIZE)
        theirs = QS.run_ext(MIRROR_SEED, trades, THETA, HOLD, MIRROR_SIZE)
        assert set(mine) == set(theirs), "returned key sets differ"
        keys = sorted(theirs)
        for k in keys:
            a, b = mine[k], theirs[k]
            if math.isnan(a) and math.isnan(b):
                continue
            assert abs(a - b) <= MIRROR_TOL * max(1.0, abs(b)), \
                "MIRROR MISMATCH %s (trades=%s): %.12g vs %.12g" \
                % (k, trades, a, b)
        if not trades:
            assert len(cap["samples"]) > 0.99 * (T / SAMPLE), \
                "control capture has only %d samples" % len(cap["samples"])
        else:
            total = sum(cap["cp_vol"].values())
            assert abs(total - mine["sn_vol"]) <= 1e-9 * max(1.0, total), \
                "counterparty buckets %.12g != sniffer volume %.12g" \
                % (total, mine["sn_vol"])
            assert len(cap["samples"]) == 0
    print("  T-mirror    PASS  every one of the %d keys damage_qs12.run_ext "
          "returns agrees to %.0e," % (len(keys), MIRROR_TOL))
    print("                    in both arms, seed %d, maker size %.3f (%.0fs)."
          % (MIRROR_SEED, MIRROR_SIZE, time.time() - t0))
    print("  T-capture   PASS  the counterparty buckets sum EXACTLY to the "
          "sniffer's own fill volume --")
    print("                    no sniffer fill escapes the capture; and the "
          "control arm sampled >99% of")
    print("                    the 60s grid.")


def selftest():
    print("=" * 128)
    print("Self-tests: extraction_diag.py. The arm has not run.")
    print("=" * 128)
    t_ident()
    t_guards()
    t_ols()
    t_tmb()
    t_windows()
    t_walk()
    t_verdict()
    t_power()
    t_mirror()
    print("")
    print("  All 10 self-tests PASSED (T-mirror and T-capture share one run).")
    return 0


def main():
    print("=" * 128)
    print("Why is extraction null? Structural null vs strategy null, "
          "cluster B, maker size 0.020 and 0.12")
    print("=" * 128)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. T=%.0fs, %d seeds," %
          (LAM, P_MARKET, LIFE, DISP, T, N_SEEDS))
    print("gamma=%.4e, k=%.5f. Sniffer %.3f BTC, theta=%.2f, hold=%ds. "
          "Samples every %ds; horizons %s s." %
          (GAMMA, K, SNIFF_SIZE, THETA, HOLD, SAMPLE,
           ", ".join(str(h) for h in HORIZONS)))
    print("Pre-registered: predicted slope sign negative; the (P2) "
          "verdict; the (P3) companion")
    print("at half the statistic; the (P4) power limit; the (P5) "
          "profit-is-not-damage reading.")
    print("")

    data = {}
    for size in MM_SIZES:
        t0 = time.time()
        print("running maker size %.3f ..." % size, flush=True)
        ctrl, treat = {}, {}
        for s in SEEDS:
            out, cap = run_diag(s, False, THETA, HOLD, size)
            check_run(out, cap, "control", s, size)
            ctrl[s] = (out, cap)
        for s in SEEDS:
            out, cap = run_diag(s, True, THETA, HOLD, size)
            check_run(out, cap, "treatment", s, size)
            treat[s] = (out, cap)
        data[size] = (ctrl, treat)
        print("  %d control + %d treatment runs in %.0fs"
              % (N_SEEDS, N_SEEDS, time.time() - t0), flush=True)
    print("")

    # reproduction
    print("=" * 128)
    print("Reproduction of damage_qs12_results.txt; tolerances from "
          "its printed precision")
    print("=" * 128)
    bad = []
    for size in MM_SIZES:
        ctrl, treat = data[size]
        d = [treat[s][0]["mm_edge"] - ctrl[s][0]["mm_edge"] for s in SEEDS]
        de, dse = strict_mean_se(d, "d edge %.3f" % size)
        sq, sqse = strict_mean_se([ctrl[s][0]["sd_q"] for s in SEEDS],
                                  "sd(q) %.3f" % size)
        sn, _snse = strict_mean_se([treat[s][0]["sn_share"] for s in SEEDS],
                                   "sniffer share %.3f" % size)
        cp, cpse = strict_mean_se([ctrl[s][0]["mm_pnl"] for s in SEEDS],
                                  "control PnL %.3f" % size)
        got = {"d_edge": de, "d_edge_se": dse, "sd_q": sq, "sd_q_se": sqse,
               "sn_share": sn, "ctrl_pnl": cp, "ctrl_pnl_se": cpse}
        for k, (want, tol) in COMMITTED[size].items():
            ok = abs(got[k] - want) <= tol
            print("  size %.3f  %-12s committed %12.5f  got %12.5f  tol %.0e"
                  "  %s" % (size, k, want, got[k], tol,
                            "OK" if ok else "*** mismatch ***"))
            if not ok:
                bad.append("%s@%.3f" % (k, size))
    if bad:
        raise AssertionError("REPRODUCTION FAILED on %s -- this is not the "
                             "committed experiment. Halting." % ", ".join(bad))
    print("  all 14 committed figures reproduce.")
    print("")

    # (A)
    print("=" * 128)
    print("(A) Who does the sniffer trade against? Treatment arm, "
          "share of sniffer fill volume")
    print("=" * 128)
    a_res = {}
    for size in MM_SIZES:
        _c, treat = data[size]
        per = {b: [] for b in CP_BUCKETS}
        tot_all = {b: 0.0 for b in CP_BUCKETS}
        for s in SEEDS:
            cv = treat[s][1]["cp_vol"]
            tot = sum(cv.values())
            for b in CP_BUCKETS:
                per[b].append(100.0 * _ratio(cv[b], tot,
                                             "share %s seed %d" % (b, s)))
                tot_all[b] += cv[b]
        grand = sum(tot_all.values())
        print("  maker size %.3f   (sniffer fill volume, all 8 seeds: "
              "%.4f BTC)" % (size, grand))
        print("    %-10s %-26s %s" % ("counterparty", "share per seed, mean "
                                      "+/- SE", "pooled share"))
        for b in CP_BUCKETS:
            vals = per[b]
            if max(vals) == 0.0:
                print("    %-12s %8.3f%%  (zero on every seed)          %8.3f%%"
                      % (b, 0.0, 0.0))
                continue
            m, se = strict_mean_se(vals, "share %s %.3f" % (b, size))
            print("    %-12s %8.3f%% +/- %-8.3f           %8.3f%%"
                  % (b, m, se, 100.0 * _ratio(tot_all[b], grand, "pooled " + b)))
        a_res[size] = per["MM"]
        print("")
    m20 = statistics.fmean(a_res[0.020])
    m12 = statistics.fmean(a_res[0.12])
    print("  the maker's share of sniffer fill volume: %.3f%% at 0.020, "
          "%.3f%% at 0.12." % (m20, m12))
    print("  The sniffer can take the maker's edge directly only on "
          "that fraction of its")
    print("  volume; the rest is paid by, or to, the other side of the market.")
    print("")

    # (B)
    print("=" * 128)
    print("(B) Does q predict future mid moves? Control arm, "
          "non-overlapping windows, true q")
    print("=" * 128)
    print("  Predicted sign (P1): negative. Slope in bp of mid per BTC "
          "of inventory.")
    print("  Primary SE across the %d seeds; pooled OLS SE is naive." % N_SEEDS)
    b_res = {}
    for size in MM_SIZES:
        ctrl, _t = data[size]
        print("")
        print("  maker size %.3f" % size)
        print("    per-seed slope (bp/BTC):")
        print("    %-6s" % "seed" + "".join("%13s" % ("h=%ds" % h)
                                              for h in HORIZONS))
        per_h = {h: [] for h in HORIZONS}
        r2_h = {h: [] for h in HORIZONS}
        n_h = {h: [] for h in HORIZONS}
        pool = {h: ([], []) for h in HORIZONS}
        for s in SEEDS:
            smp = ctrl[s][1]["samples"]
            row = []
            for h in HORIZONS:
                xs, ys = windows(smp, h)
                b, _se, r2, n = ols(xs, ys, "seed %d h %d" % (s, h))
                per_h[h].append(b)
                r2_h[h].append(r2)
                n_h[h].append(n)
                pool[h][0].extend(xs)
                pool[h][1].extend(ys)
                row.append(b)
            print("    %-6d" % s + "".join("%13.3f" % v for v in row))
        print("")
        print("    %-7s %-24s %-9s %-10s %-10s %-24s %s"
              % ("h (s)", "slope, SE across seeds", "|t|", "sign ok?",
                 "mean R2", "pooled slope (naive SE)", "windows"))
        b_res[size] = {}
        for h in HORIZONS:
            m, se = strict_mean_se(per_h[h], "slope h%d %.3f" % (h, size))
            tt = _ratio(abs(m), se, "slope t h%d" % h)
            pb, pse, pr2, pn = ols(pool[h][0], pool[h][1], "pooled h%d" % h)
            ok = (m < 0) if PREDICTED_SIGN < 0 else (m > 0)
            b_res[size][h] = (m, se)
            print("    %-7d %9.3f +/- %-11.3f %-9.2f %-10s %-10.5f %9.3f +/- "
                  "%-11.3f %d/seed, %d total"
                  % (h, m, se, tt, "yes" if ok else "no",
                     statistics.fmean(r2_h[h]), pb, pse,
                     min(n_h[h]), sum(n_h[h])))
    print("")
    print("  L1 Stambaugh: q is persistent and its innovations (fills) "
          "correlate with mid innovations.")
    print("  Non-overlapping windows remove return overlap, not that "
          "bias; SE across seeds is")
    print("  robust to within-seed dependence only.")
    print("")

    # (C)
    print("=" * 128)
    print("(C) Can the predictable move pay for the trade? TMB vs "
          "measured round trip")
    print("=" * 128)
    verdicts, comp_verdicts = {}, {}
    for size in MM_SIZES:
        ctrl, _t = data[size]
        costs = [statistics.fmean(c for (_q, _m, c) in
                                  ctrl[s][1]["samples"].values())
                 for s in SEEDS]
        c, se_c = strict_mean_se(costs, "round-trip cost %.3f" % size)
        print("")
        print("  maker size %.3f; measured round-trip cost of a %.3f BTC "
              "market order: %.4f +/- %.4f bp"
              % (size, SNIFF_SIZE, c, se_c))
        print("    (cost 95%% band [%.4f, %.4f] bp; mean over %d samples/seed; "
              "excludes permanent impact, L2)"
              % (c - Z * se_c, c + Z * se_c,
                 len(ctrl[SEEDS[0]][1]["samples"])))
        print("    %-7s %-22s %-10s %-26s %-10s %s"
              % ("h (s)", "TMB bp, SE", "sign ok?", "|TMB| 95% band",
                 "(P2) call", "(P3) companion, |TMB|/2"))
        calls, ccalls = [], []
        for h in HORIZONS:
            vals = []
            for s in SEEDS:
                xs, ys = windows(ctrl[s][1]["samples"], h)
                vals.append(tmb(xs, ys, "seed %d h %d" % (s, h)))
            m, se = strict_mean_se(vals, "TMB h%d %.3f" % (h, size))
            call, mag, lo, hi, _cl, _ch = horizon_call(m, se, c, se_c, 1.0)
            ccall, cmag, clo, chi, _a, _b = horizon_call(
                m, se, c, se_c, COMPANION_FACTOR)
            ok = (m < 0) if PREDICTED_SIGN < 0 else (m > 0)
            calls.append((h, call))
            ccalls.append((h, ccall))
            print("    %-7d %8.4f +/- %-9.4f %-10s [%8.4f, %8.4f]      "
                  "%-10s %-9s [%.4f, %.4f]"
                  % (h, m, se, "yes" if ok else "no", lo, hi, call, ccall,
                     clo, chi))
        v, hs = size_verdict(calls)
        cv, chs = size_verdict(ccalls)
        verdicts[size] = v
        comp_verdicts[size] = cv
        print("    (P2) VERDICT at %.3f: %s%s" % (
            size, v, ("  -- horizon(s) %s" % ", ".join(str(x) for x in hs))
            if hs else ""))
        print("    (P3) companion  at %.3f: %s%s" % (
            size, cv, ("  -- horizon(s) %s" % ", ".join(str(x) for x in chs))
            if chs else ""))

    print("")
    print("=" * 128)
    print("Verdict")
    print("=" * 128)
    ov = overall_verdict(verdicts)
    cov = overall_verdict(comp_verdicts)
    for size in MM_SIZES:
        print("  maker size %.3f   (P2) %-16s   (P3 companion) %s"
              % (size, verdicts[size], comp_verdicts[size]))
    print("")
    print("  overall (P2): %s" % ov)
    print("  overall (P3 companion, one tail per trade): %s" % cov)
    if ov != cov:
        print("  The two disagree: (P2) compares the gap between two tails "
              "with one round trip; a single")
        print("  trade captures one tail. The companion is the per-trade "
              "reading; (P2)'s call")
        print("  rests on the ~2x bias noted in the header.")
    print("")
    print("  With (A) and (P5): a STRATEGY NULL would mean a better "
          "attacker could profit from q;")
    print("  only the maker's share")
    print("  of sniffer volume (%.3f%% at 0.020, %.3f%% at 0.12) could "
          "come from the maker directly." % (m20, m12))
    print("  (P4): STRUCTURAL NULL unreachable at %d seeds (recorded "
          "before the run)." % N_SEEDS)

    nan_sh = sum(1 for size in MM_SIZES for arm in (0, 1) for s in SEEDS
                 if math.isnan(data[size][arm][s][0]["mm_sharpe"]))
    print("")
    print("  NaN Sharpe seeds across all 32 runs (counted, not halted on): %d"
          % nan_sh)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
