# quotesize_extend.py: extends the quote_size grid to 0.12 and 0.16

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import quotesize_verdict as QV

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from clipped_mm_check import _resid_best, MM_ID
from quotesize_verdict import (
    classify, overall, clears_sd, log_slope,
    SD_REF, SD_FACTOR, SD_MIN, QUOTE_SIZES as COMMITTED_SIZES,
    T, SEEDS, LAM, P_MARKET, LIFE, DISP, K, GAMMA_CAP, SAMPLE_EVERY,
    C_PER_GAMMA,
)
from clipped_window import band, verdict, Z, TARGETS

NEW_SIZES = (0.12, 0.16)
ALL_SIZES = tuple(COMMITTED_SIZES) + NEW_SIZES

COMMITTED_EXPONENT = 0.5148
RUNAWAY = 50.0

COMMITTED_ROWS = {
    0.020: {"fill_share": 5.80, "vol_share": 14.30, "sd_q": 0.04604,
            "sd_q_se": 0.00507, "max_q": 0.15129, "median_bp": 0.32138,
            "levels_per_side": 550.98455, "agg_per_s": 0.04083,
            "shape": 2.17765},
    0.035: {"fill_share": 6.19, "vol_share": 19.41, "sd_q": 0.05857,
            "sd_q_se": 0.00605, "max_q": 0.20436, "median_bp": 0.33237,
            "levels_per_side": 550.57817, "agg_per_s": 0.03927,
            "shape": 2.10918},
    0.050: {"fill_share": 6.34, "vol_share": 22.77, "sd_q": 0.07211,
            "sd_q_se": 0.00831, "max_q": 0.24392, "median_bp": 0.33556,
            "levels_per_side": 550.63650, "agg_per_s": 0.03859,
            "shape": 2.08267},
    0.080: {"fill_share": 6.59, "vol_share": 27.16, "sd_q": 0.09359,
            "sd_q_se": 0.00895, "max_q": 0.31357, "median_bp": 0.33372,
            "levels_per_side": 550.61680, "agg_per_s": 0.03765,
            "shape": 2.08826},
}
REPRO_TOL = {"fill_share": 5e-3, "vol_share": 5e-3, "sd_q": 5e-6,
             "sd_q_se": 5e-6, "max_q": 5e-6, "median_bp": 5e-6,
             "levels_per_side": 5e-6, "agg_per_s": 5e-6, "shape": 5e-6}

MIRROR_TOL = 1e-9
SELFTEST_SEED = 0
SELFTEST_QSIZE = 0.02


def _finite(x, what):
    if x is None or not math.isfinite(x):
        raise ValueError("NON-FINITE %s: %r -- refusing to continue" % (what, x))
    return x


def run_ext(seed, qsize):
    """Transcription of quotesize_verdict.run(), asserted equal to it by"""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA_CAP, quote_size=qsize)

    n_agg = n_fill = n_mm_fill = 0
    vol_total = vol_mm = 0.0
    resid_bp, mm_spreads = [], []
    lvl_ns, near_ns, near_mm_l, near_all_l = [], [], [], []
    q_all = []
    allin_bp = []
    n_touch_secs = n_mm_sets_touch = 0

    t = 0.0
    while t < T:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                n_agg += len(set(f.price for f in fills))
                n_fill += len(fills)
                for f in fills:
                    vol_total += f.size
                    if f.counterparty_id == MM_ID:
                        n_mm_fill += 1
                        vol_mm += f.size
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)
        q_all.append(mm.q)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mid = 0.5 * (bb + ba)
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is not None and ra is not None and ra > rb:
            resid_bp.append(1e4 * (ra - rb) / (0.5 * (rb + ra)))
            if ba > bb:
                allin_bp.append(1e4 * (ba - bb) / mid)

        mmb = mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mmb = o.price if mmb is None else max(mmb, o.price)
            else:
                mma = o.price if mma is None else min(mma, o.price)
        if mmb is not None and mma is not None:
            mm_spreads.append(mma - mmb)

        n_touch_secs += 1
        if (mmb is not None and mmb >= bb) or (mma is not None and mma <= ba):
            n_mm_sets_touch += 1

        if t % SAMPLE_EVERY:
            continue

        l_ns = 0
        q_ns = q_mm = q_all_sz = 0.0
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        for book, is_bid in ((eng.bids, True), (eng.asks, False)):
            for p, q in book.items():
                live_ns = [o for o in q
                           if o.id not in seed_ids and o.agent_id != MM_ID]
                if live_ns:
                    l_ns += 1
                if (p >= lo) if is_bid else (p <= hi):
                    q_ns += sum(o.size for o in live_ns)
                    q_mm += sum(o.size for o in q if o.agent_id == MM_ID)
                    q_all_sz += sum(o.size for o in q if o.id not in seed_ids)
        lvl_ns.append(0.5 * l_ns)
        near_ns.append(0.5 * q_ns)
        near_mm_l.append(0.5 * q_mm)
        near_all_l.append(0.5 * q_all_sz)

    near_mm = statistics.mean(near_mm_l) if near_mm_l else 0.0
    near_all = statistics.mean(near_all_l) if near_all_l else 0.0

    if resid_bp:
        med = statistics.median(resid_bp)
        mean_bp = statistics.mean(resid_bp)
        shape = (mean_bp / med) if med else float("nan")
    else:
        med = mean_bp = shape = float("nan")

    return {
        "fill_share": 100.0 * n_mm_fill / n_fill if n_fill else 0.0,
        "vol_share": 100.0 * vol_mm / vol_total if vol_total else 0.0,
        "mm_fills": float(n_mm_fill),
        "mm_spread": statistics.median(mm_spreads) if mm_spreads else
        float("nan"),
        "width_bp": med,
        "levels": statistics.mean(lvl_ns) if lvl_ns else float("nan"),
        "agg_per_s": n_agg / T,
        "near_ns": statistics.mean(near_ns) if near_ns else float("nan"),
        "near_mm": near_mm,
        "near_mm_share": (100.0 * near_mm / near_all) if near_all > 0 else 0.0,
        "median_bp": med,
        "mean_bp": mean_bp,
        "shape": shape,
        "levels_per_side": statistics.mean(lvl_ns) if lvl_ns else float("nan"),
        "sd_q": statistics.stdev(q_all) if len(q_all) > 1 else 0.0,
        "max_q": max(abs(x) for x in q_all) if q_all else 0.0,
        "n_q": float(len(q_all)),
        "allin_bp": statistics.median(allin_bp) if allin_bp else float("nan"),
        "touch_set_share": (100.0 * n_mm_sets_touch / n_touch_secs)
        if n_touch_secs else 0.0,
    }


def measure_ext(qsize, seeds=SEEDS):
    """Per-seed then averaged, identical in form to quotesize_verdict.measure"""
    per = None
    t0 = time.time()
    for s in seeds:
        x = run_ext(s, qsize)
        for k, v in x.items():
            _finite(v, "%s at seed %d, quote_size %.3f" % (k, s, qsize))
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    out = {"qsize": qsize, "secs": time.time() - t0, "n": len(seeds)}
    n = len(seeds)
    for k, v in per.items():
        out[k] = _finite(statistics.mean(v), "mean %s at %.3f" % (k, qsize))
        out[k + "_se"] = _finite(
            (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0,
            "se %s at %.3f" % (k, qsize))
    return out


def runaway(r):
    """Pre-registered participation flag"""
    return (r["vol_share"] >= RUNAWAY or r["near_mm_share"] >= RUNAWAY
            or r["touch_set_share"] >= RUNAWAY)


def t_ident():
    """The threshold and the decision rule are quotesize_verdict's own objects"""
    assert classify is QV.classify and overall is QV.overall
    assert clears_sd is QV.clears_sd and log_slope is QV.log_slope
    assert SD_MIN is QV.SD_MIN and SD_REF is QV.SD_REF
    assert abs(SD_MIN - 0.08258) < 1e-12, "SD_MIN moved: %r" % SD_MIN
    assert abs(SD_FACTOR - 2.0) < 1e-12
    assert TARGETS is QV.TARGETS and verdict is QV.verdict and band is QV.band
    assert Z == 1.96
    assert tuple(COMMITTED_SIZES) == (0.020, 0.035, 0.050, 0.080)
    assert (T, tuple(SEEDS), K, GAMMA_CAP) == (QV.T, tuple(QV.SEEDS),
                                               QV.K, QV.GAMMA_CAP)
    assert (LAM, P_MARKET, LIFE, DISP) == (QV.LAM, QV.P_MARKET, QV.LIFE,
                                           QV.DISP)
    print("  T-IDENT       PASS  SD_MIN is quotesize_verdict's own object and "
          "equals %.5f BTC;" % SD_MIN)
    print("                      classify/overall/clears_sd/TARGETS/verdict/"
          "band are its objects too.")
    print("                      Working point, seeds, k and gamma all equal "
          "the committed values.")


def t_nan():
    """NaN and infinity raise rather than propagating"""
    for bad in (float("nan"), float("inf"), float("-inf"), None):
        try:
            _finite(bad, "planted")
        except ValueError:
            continue
        raise AssertionError("_finite accepted %r" % bad)
    _finite(0.0, "zero")
    _finite(-1.5, "negative")
    print("  T-NAN         PASS  nan, +inf, -inf and None all raise; finite "
          "values pass.")


def t_choice():
    """The 0.12 / 0.16 choice is arithmetic from the committed numbers, and the"""
    c80 = COMMITTED_ROWS[0.080]
    assert c80["sd_q"] > SD_MIN, "0.080's POINT estimate should already clear"
    lo80 = c80["sd_q"] - Z * c80["sd_q_se"]
    assert abs(lo80 - 0.07606) < 2e-5, "0.080 lower bound is %r" % lo80
    assert lo80 < SD_MIN, "0.080's lower bound should fall short"
    ratios = [COMMITTED_ROWS[q]["sd_q_se"] / COMMITTED_ROWS[q]["sd_q"]
              for q in COMMITTED_SIZES]
    assert 0.09 < min(ratios) and max(ratios) < 0.12, \
        "SE/sd outside the 9-12%% band used for the projection: %r" % ratios
    need = SD_MIN / (1.0 - Z * 0.105)
    proj = 0.080 * (need / c80["sd_q"]) ** (1.0 / COMMITTED_EXPONENT)
    assert 0.095 < proj < 0.101, "projected clearance point %r" % proj
    assert NEW_SIZES[0] > proj and NEW_SIZES[1] > proj
    need_exp = math.log(need / c80["sd_q"]) / math.log(NEW_SIZES[1] / 0.080)
    assert 0.14 < need_exp < 0.17, "exponent needed at 0.16 is %r" % need_exp
    print("  T-choice      PASS  0.080 point %.5f clears %.5f but its 95%% lo "
          "%.5f does not."
          % (c80["sd_q"], SD_MIN, lo80))
    print("                      SE/sd across the committed four is %.1f%%-"
          "%.1f%%; at 10.5%% clearance needs"
          % (100 * min(ratios), 100 * max(ratios)))
    print("                      sd(q) >= %.5f, reached at quote_size %.4f on "
          "the committed exponent" % (need, proj))
    print("                      %.4f. At 0.16 the exponent could fall to %.3f "
          "and still clear."
          % (COMMITTED_EXPONENT, need_exp))


def t_runaway():
    """The participation flag fires on each of its three legs, and only there"""
    base = {"vol_share": 10.0, "near_mm_share": 10.0, "touch_set_share": 10.0}
    assert not runaway(base)
    for leg in base:
        hot = dict(base)
        hot[leg] = RUNAWAY
        assert runaway(hot), "flag did not fire on %s" % leg
        cool = dict(base)
        cool[leg] = RUNAWAY - 0.01
        assert not runaway(cool), "flag fired below threshold on %s" % leg
    print("  T-runaway     PASS  flag fires at %.0f%% on each of volume, "
          "near-touch and touch-set," % RUNAWAY)
    print("                      and not at %.2f%%. It is reported, and does "
          "NOT enter the verdict." % (RUNAWAY - 0.01))


def t_mirror_ext():
    """The load-bearing test: run_ext is quotesize_verdict.run plus columns"""
    shared = ("fill_share", "vol_share", "mm_fills", "mm_spread", "width_bp",
              "levels", "agg_per_s", "near_ns", "near_mm", "near_mm_share",
              "median_bp", "mean_bp", "shape", "levels_per_side", "sd_q",
              "max_q", "n_q")
    t0 = time.time()
    mine = run_ext(SELFTEST_SEED, SELFTEST_QSIZE)
    theirs = QV.run(SELFTEST_SEED, SELFTEST_QSIZE)
    for k in shared:
        a, b = mine[k], theirs[k]
        assert abs(a - b) <= MIRROR_TOL * max(1.0, abs(b)), \
            "MIRROR MISMATCH on %s: mine %.12g vs committed %.12g" % (k, a, b)
    assert math.isfinite(mine["allin_bp"])
    assert 0.0 <= mine["touch_set_share"] <= 100.0
    assert mine["allin_bp"] <= mine["median_bp"] + 1e-9, \
        "all-in touch (%r) cannot be WIDER than the residual touch (%r)" \
        % (mine["allin_bp"], mine["median_bp"])
    print("  T-mirror-EXT  PASS  all %d shared quantities agree with "
          "quotesize_verdict.run to %.0e" % (len(shared), MIRROR_TOL))
    print("                      at seed %d, quote_size %.3f (%.0fs). The new "
          "columns add no RNG draw."
          % (SELFTEST_SEED, SELFTEST_QSIZE, time.time() - t0))
    print("                      all-in touch %.5f bp <= residual touch %.5f "
          "bp, maker sets the touch %.1f%% of seconds."
          % (mine["allin_bp"], mine["median_bp"], mine["touch_set_share"]))


def selftest():
    print("=" * 126)
    print("Self-tests; quotesize_extend.py. The arm has NOT run.")
    print("=" * 126)
    t_ident()
    t_nan()
    t_choice()
    t_runaway()
    t_mirror_ext()
    print("")
    print("  All 5 self-tests PASSED. T-REPRO runs inside main(), because it "
          "re-measures the four")
    print("  committed cells at full seed count and that is the run.")
    return 0


def main():
    print("=" * 126)
    print("Closing the quote_size grid; extending upward past the out of "
          "grid verdict")
    print("=" * 126)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("k=%.5f. gamma=%.2e (C=%.2f). DEFAULT_GAMMA=%g and "
          "DEFAULT_QUOTE_SIZE=%g untouched."
          % (K, GAMMA_CAP, GAMMA_CAP * C_PER_GAMMA, DEFAULT_GAMMA,
             DEFAULT_QUOTE_SIZE))
    print("%d seeds x %.0fs per size, book scanned every %ds. quote_size is "
          "the only thing that varies."
          % (len(SEEDS), T, SAMPLE_EVERY))
    print("")
    print("The threshold is unchanged and imported: sd(q) 95%% LOWER bound >= "
          "%.1f x %.5f = %.5f BTC."
          % (SD_FACTOR, SD_REF, SD_MIN))
    print("The committed four cells are re-run here, not copied, and asserted "
          "against")
    print("quotesize_verdict_results.txt. That is what makes the new rows "
          "poolable with the old.")
    print("")

    rows = []
    for qs in ALL_SIZES:
        r = measure_ext(qs)
        rows.append(r)
        tag = "committed" if qs in COMMITTED_ROWS else "NEW"
        print("  quote_size=%.3f BTC [%-9s] %4.0fs  fill=%5.2f%%  vol=%5.2f%%  "
              "near=%5.2f%%  touch-set=%5.2f%%  sd(q)=%.5f"
              % (qs, tag, r["secs"], r["fill_share"], r["vol_share"],
                 r["near_mm_share"], r["touch_set_share"], r["sd_q"]),
              flush=True)
    print("")

    print("=" * 126)
    print("T-REPRO; do the four committed cells reproduce "
          "quotesize_verdict_results.txt?")
    print("=" * 126)
    bad = 0
    for r in rows:
        if r["qsize"] not in COMMITTED_ROWS:
            continue
        want = COMMITTED_ROWS[r["qsize"]]
        diffs = []
        for k, w in want.items():
            got = r[k]
            if abs(got - w) > REPRO_TOL[k]:
                diffs.append("%s committed=%.5f got=%.5f" % (k, w, got))
        if diffs:
            bad += 1
            print("  quote_size=%.3f  *** mismatch ***" % r["qsize"])
            for d in diffs:
                print("      %s" % d)
        else:
            print("  quote_size=%.3f  OK; all %d committed fields reproduce"
                  % (r["qsize"], len(want)))
    if bad:
        raise AssertionError(
            "T-REPRO FAILED on %d committed cell(s). The new rows are NOT "
            "poolable with the committed ones and nothing below is reported."
            % bad)
    print("  all four committed cells reproduce. The new rows pool with them.")
    print("")

    print("=" * 126)
    print("1. THE FOUR COMMITTED TARGETS, scored by clipped_window.verdict "
          "(imported, not reimplemented)")
    print("=" * 126)
    cells = {}
    for r in rows:
        tag = "committed" if r["qsize"] in COMMITTED_ROWS else "NEW"
        print("  quote_size=%.3f  [%s]" % (r["qsize"], tag))
        vs = []
        for name, key, tgt, tol in TARGETS:
            pt, se = r[key], r[key + "_se"]
            v = verdict(pt, se, tgt, tol)
            vs.append(v)
            lo_b, hi_b = band(tgt, tol)
            print("    %-20s %10.5f +/- %-9.5f  95%% [%9.5f, %9.5f]  band "
                  "[%9.5f, %9.5f]  %s"
                  % (name, pt, se, pt - Z * se, pt + Z * se, lo_b, hi_b, v))
        n_met = sum(1 for v in vs if v == "MET")
        n_miss = sum(1 for v in vs if v == "MISSED")
        print("    -> %d/4 MET, %d MISSED, %d UNRESOLVED"
              % (n_met, n_miss, 4 - n_met - n_miss))
        cells[r["qsize"]] = vs
        print("")

    print("=" * 126)
    print("2. sd(q) and max|q| across the extended grid")
    print("=" * 126)
    print("  %11s %5s %20s %10s %14s %14s %13s"
          % ("quote_size", "", "sd(q) BTC", "95% lo", "sd(q)/qsize",
             "max|q| BTC", "max|q|/qsize"))
    for r in rows:
        tag = "   " if r["qsize"] in COMMITTED_ROWS else "NEW"
        lo = r["sd_q"] - Z * r["sd_q_se"]
        print("  %11.3f %5s %11.5f +/- %-6.5f %10.5f %14.3f %14.5f %13.3f"
              % (r["qsize"], tag, r["sd_q"], r["sd_q_se"], lo,
                 r["sd_q"] / r["qsize"], r["max_q"], r["max_q"] / r["qsize"]))
    print("")

    e_all = log_slope([r["qsize"] for r in rows], [r["sd_q"] for r in rows])
    old = [r for r in rows if r["qsize"] in COMMITTED_ROWS]
    e_old = log_slope([r["qsize"] for r in old], [r["sd_q"] for r in old])
    top = [r for r in rows if r["qsize"] >= 0.080]
    e_top = log_slope([r["qsize"] for r in top], [r["sd_q"] for r in top])
    print("  fitted exponent of sd(q) on quote_size")
    print("    committed range 0.020 -> 0.080 (re-fitted here): %.4f   "
          "(committed value %.4f)" % (e_old, COMMITTED_EXPONENT))
    print("    extended range  0.020 -> 0.160                  : %.4f   "
          "(change %+.4f)" % (e_all, e_all - COMMITTED_EXPONENT))
    print("    new segment     0.080 -> 0.160                  : %.4f   "
          "(change %+.4f)" % (e_top, e_top - COMMITTED_EXPONENT))
    print("")
    print("  The committed record (:65-69) already carries one failure of this "
          "power law to")
    print("  extrapolate: 0.74 fitted below 0.020 became 0.5148 over 0.020-"
          "0.080, a change of")
    print("  -0.2252 across non-overlapping ranges.")
    if abs(e_top - COMMITTED_EXPONENT) < 0.05:
        print("  This extension does not move it again: the new segment's "
              "exponent sits within 0.05")
        print("  of the committed one, so 0.020-0.160 is described by a single "
              "exponent.")
    else:
        print("  The exponent moves again. That is a third data point on the "
              "same phenomenon and")
        print("  belongs in the results inventory: sd(q) on quote_size is not "
              "a single power law")
        print("  over 0.005-0.160, and no exponent from any sub-range may be "
              "used to project.")
    print("")

    print("=" * 126)
    print("3. The pre-registered verdict; threshold imported, not moved")
    print("=" * 126)
    per_cell = []
    for r in rows:
        c = classify(cells[r["qsize"]], r["sd_q"], r["sd_q_se"])
        per_cell.append(c)
        lo = r["sd_q"] - Z * r["sd_q_se"]
        tag = "   " if r["qsize"] in COMMITTED_ROWS else "NEW"
        print("  quote_size=%.3f %s  sd(q) 95%% lo = %.5f vs threshold %.5f   "
              "-> %s" % (r["qsize"], tag, lo, SD_MIN, c))
    print("")
    out = overall(per_cell)
    print("  overall: %s" % out)
    print("")

    print("=" * 126)
    print("4. Participation; is a cell that clears still a market?")
    print("=" * 126)
    print("  Pre-registered before the run: runaway if volume share, "
          "near-touch share or")
    print("  touch-set share reaches %.0f%%. Reported separately; it does NOT "
          "enter the verdict above." % RUNAWAY)
    print("")
    print("  %11s %9s %9s %10s %11s %12s %12s %8s"
          % ("quote_size", "fill %", "vol %", "near %", "touch-set %",
             "resid bp", "all-in bp", "flag"))
    for r in rows:
        print("  %11.3f %9.2f %9.2f %10.2f %11.2f %12.5f %12.5f %8s"
              % (r["qsize"], r["fill_share"], r["vol_share"],
                 r["near_mm_share"], r["touch_set_share"], r["median_bp"],
                 r["allin_bp"], "runaway" if runaway(r) else "ok"))
    print("")
    a = rows[0]
    print("  Baseline at the committed 0.020: fill %.2f%%, volume %.2f%%, "
          "near-touch %.2f%%, touch-set %.2f%%."
          % (a["fill_share"], a["vol_share"], a["near_mm_share"],
             a["touch_set_share"]))
    for r in rows:
        if r["qsize"] in NEW_SIZES:
            print("  At %.3f (%.0fx the committed clip): fill %.2f%% (%.2fx), "
                  "volume %.2f%% (%.2fx), near-touch %.2f%% (%.2fx)."
                  % (r["qsize"], r["qsize"] / 0.020, r["fill_share"],
                     r["fill_share"] / a["fill_share"], r["vol_share"],
                     r["vol_share"] / a["vol_share"], r["near_mm_share"],
                     (r["near_mm_share"] / a["near_mm_share"])
                     if a["near_mm_share"] > 0 else float("nan")))
    flagged = [r["qsize"] for r in rows if runaway(r)]
    print("")
    if flagged:
        print("  runaway at: %s BTC. At these clips the maker is more than "
              "half of at least one of" % ", ".join("%.3f" % q for q in flagged))
        print("  volume, near-touch resting size, or touch-setting. "
              "the research plan")
        print("  names this case: if the maker is the sole liquidity provider, "
              "every print is")
        print("  attributable, q is recoverable by summation, and the research "
              "question does not")
        print("  exist. A cell flagged here is NOT a usable configuration for "
              "a damage A/B even if")
        print("  its sd(q) clears.")
    else:
        print("  No cell is flagged. The maker stays below %.0f%% on volume, "
              "near-touch size and" % RUNAWAY)
        print("  touch-setting at every size tested, so no cleared cell is "
              "cleared by becoming the")
        print("  market. The residual and all-in touch columns show how far "
              "inside the book it sits.")
    print("")

    print("=" * 126)
    print("What this run does not do")
    print("=" * 126)
    print("  N1  It does not measure damage. Phase 7 is not run. This "
          "establishes only whether an")
    print("      inventory scale exists at which a damage A/B would have room "
          "to show anything.")
    print("  N2  T=%.0fs per cell. The damage A/B runs 7 days, where sd(q) "
          "could be larger --" % T)
    print("      which makes this test conservative about reaching the "
          "threshold, not optimistic.")
    print("  N3  The four scored targets are unchanged and near-touch depth is "
          "still not among")
    print("      them. No new gate was invented inside this run.")
    print("  N4  gamma is held at the binding viability cap, not swept.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
