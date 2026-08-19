# quotesize_verdict.py: quote_size ceiling under the verdict standard; sd(q) by quote_size

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import clipped_window
import probe_quotesize_lever

from clipped_placement import make_world, TARGET as RATE_TARGET
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from clipped_mm_check import _resid_best, ABSENT, MM_ID
from clipped_window import band, verdict, margin_se, Z, TARGETS
from clipped_depth_sweep import (REAL_MEDIAN_BP, REAL_MEAN_BP, REAL_LEVELS,
                                 TOL_WIDTH, TOL_LEVELS, TOL_RATE)

T = 86400.0
SEEDS = list(range(8))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
SAMPLE_EVERY = 60
GAMMA_CAP = 9.4e-6
C_PER_GAMMA = 1.458e6

QUOTE_SIZES = (0.020, 0.035, 0.050, 0.080)

SD_REF = 0.04129
SD_FACTOR = 2.0
SD_MIN = SD_FACTOR * SD_REF

CLAIMED_EXPONENT = 0.74

SELFTEST_SEED = 0
SELFTEST_QSIZE = 0.02
MIRROR_TOL = 1e-9


def run(seed, qsize):
    """One cell"""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA_CAP, quote_size=qsize)

    n_agg = n_fill = n_mm_fill = 0
    vol_total = vol_mm = 0.0
    resid_bp, mm_spreads = [], []
    lvl_ns, near_ns, near_mm_l, near_all_l = [], [], [], []
    q_all = []

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
    }


def measure(qsize, seeds=SEEDS):
    per = None
    t0 = time.time()
    for s in seeds:
        x = run(s, qsize)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    out = {"qsize": qsize, "secs": time.time() - t0, "n": len(seeds)}
    n = len(seeds)
    for k, v in per.items():
        out[k] = statistics.mean(v)
        out[k + "_se"] = (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0
    return out


def clears_sd(sd, sd_se):
    """Whole-interval clearance, the same convention clipped_window uses"""
    return (sd - Z * sd_se) >= SD_MIN


def classify(verdicts, sd, sd_se):
    """Per-size outcome"""
    if not clears_sd(sd, sd_se):
        return "TOO_SMALL"
    if sum(1 for v in verdicts if v == "MET") == len(verdicts):
        return "REOPENS"
    if any(v == "MISSED" for v in verdicts):
        return "BOUNDED"
    return "UNRESOLVED"


def overall(cells):
    """cells is a list of per-size classifications"""
    if any(c == "REOPENS" for c in cells):
        return "REOPENED"
    if any(c == "UNRESOLVED" for c in cells):
        return "UNRESOLVED"
    if any(c == "BOUNDED" for c in cells):
        return "BOUNDED"
    return "OUT OF GRID"


def log_slope(xs, ys):
    """Least-squares slope of ln(y) on ln(x): the power-law exponent"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx if sxx else float("nan")


def t_ident():
    """T-IDENT: band/verdict/TARGETS are clipped_window's own objects, and the"""
    assert band is clipped_window.band, "band is not clipped_window.band"
    assert verdict is clipped_window.verdict, "verdict is not the imported one"
    assert margin_se is clipped_window.margin_se
    assert TARGETS is clipped_window.TARGETS, "TARGETS is not the committed list"
    assert Z == clipped_window.Z == 1.96
    expected = [("median width bp", "median_bp", REAL_MEDIAN_BP, TOL_WIDTH),
                ("levels/side", "levels_per_side", REAL_LEVELS, TOL_LEVELS),
                ("aggTrades/s", "agg_per_s", RATE_TARGET, TOL_RATE),
                ("mean/median shape", "shape",
                 REAL_MEAN_BP / REAL_MEDIAN_BP, 0.50)]
    assert len(TARGETS) == 4, "expected exactly four scored targets"
    for got, want in zip(TARGETS, expected):
        assert got == want, "committed target changed: %r vs %r" % (got, want)
    print("  T-IDENT     PASS  band/verdict/margin_se/TARGETS are "
          "clipped_window's own objects;")
    print("                    all four (target, tol) pairs equal the "
          "committed constants.")


def t_bands():
    """T-bands: the four numeric bands, against arithmetic done by hand"""
    got = [band(tgt, tol) for _n, _k, tgt, tol in TARGETS]
    want = [(0.2873 * 0.8, 0.2873 * 1.2),
            (666.34 * 0.8, 666.34 * 1.2),
            (0.0451 * 0.9, 0.0451 * 1.1),
            ((0.5436 / 0.2873) * 0.5, (0.5436 / 0.2873) * 1.5)]
    for (glo, ghi), (wlo, whi) in zip(got, want):
        assert abs(glo - wlo) < 1e-12 and abs(ghi - whi) < 1e-12, \
            "band mismatch: got (%r, %r) want (%r, %r)" % (glo, ghi, wlo, whi)
    assert abs(got[0][1] - 0.34476) < 1e-9, "width ceiling is not 0.34476"
    assert abs(got[2][0] - 0.04059) < 1e-9, "rate floor is not 0.04059"
    print("  T-bands     PASS  width [%.5f, %.5f]  levels [%.3f, %.3f]  "
          "rate [%.5f, %.5f]" % (got[0][0], got[0][1], got[1][0], got[1][1],
                                 got[2][0], got[2][1]))
    print("                    shape [%.5f, %.5f].  The two edges the probe's "
          "verdict turns on are"
          % (got[3][0], got[3][1]))
    print("                    the width ceiling 0.34476 and the rate floor "
          "0.04059.")


def t_verdict():
    """T-VERDICT: the three outcomes on cases whose answer is known by hand"""
    tgt, tol = REAL_MEDIAN_BP, TOL_WIDTH
    assert verdict(0.2873, 0.001, tgt, tol) == "MET"
    assert verdict(0.5000, 0.001, tgt, tol) == "MISSED"
    assert verdict(0.3400, 0.0100, tgt, tol) == "UNRESOLVED"
    assert verdict(0.35051, 0.0001, tgt, tol) == "MISSED"
    assert verdict(0.35051, 0.0100, tgt, tol) == "UNRESOLVED"
    print("  T-VERDICT   PASS  0.2873+-0.001 MET, 0.5000+-0.001 MISSED, "
          "0.3400+-0.010 UNRESOLVED.")
    print("                    The probe's 0.35051 reads MISSED at SE 0.0001 "
          "and UNRESOLVED at SE 0.0100,")
    print("                    so the SE it never reported is exactly what "
          "decides its verdict.")


def t_pass():
    """T-PASS: the decision rule is total and its branches are exclusive"""
    big, small = 0.20, 0.02
    m4 = ["MET"] * 4
    m3u = ["MET", "MET", "MET", "UNRESOLVED"]
    m3x = ["MET", "MET", "MET", "MISSED"]
    mix = ["MET", "UNRESOLVED", "MISSED", "MET"]
    assert classify(m4, big, 0.0) == "REOPENS"
    assert classify(m3u, big, 0.0) == "UNRESOLVED"
    assert classify(m3x, big, 0.0) == "BOUNDED"
    assert classify(mix, big, 0.0) == "BOUNDED"
    assert classify(m4, small, 0.0) == "TOO_SMALL"
    assert classify(m3x, small, 0.0) == "TOO_SMALL"
    assert clears_sd(SD_MIN + 0.001, 0.0) is True
    assert clears_sd(SD_MIN + 0.001, 0.01) is False
    assert overall(["REOPENS", "BOUNDED", "TOO_SMALL"]) == "REOPENED"
    assert overall(["UNRESOLVED", "BOUNDED", "TOO_SMALL"]) == "UNRESOLVED"
    assert overall(["BOUNDED", "TOO_SMALL"]) == "BOUNDED"
    assert overall(["TOO_SMALL", "TOO_SMALL"]) == "OUT OF GRID"
    print("  T-PASS      PASS  four per-size branches and four overall "
          "branches all fire correctly;")
    print("                    SD_MIN = %.5f BTC and a straddling interval "
          "does NOT clear it." % SD_MIN)


def t_exponent():
    """T-exponent: the log-log fit recovers a known exponent exactly"""
    xs = [0.020, 0.035, 0.050, 0.080]
    ys = [3.0 * (x ** 0.74) for x in xs]
    got = log_slope(xs, ys)
    assert abs(got - 0.74) < 1e-12, "log_slope returned %r" % got
    got3 = log_slope([0.005, 0.010, 0.020], [0.01479, 0.02255, 0.04129])
    assert abs(got3 - CLAIMED_EXPONENT) < 0.01, \
        "committed three points give %.4f, not ~%.2f" % (got3, CLAIMED_EXPONENT)
    print("  T-exponent  PASS  recovers 0.74 exactly on synthetic data; the "
          "committed three points")
    print("                    (0.01479 / 0.02255 / 0.04129) fit %.4f, which "
          "is the claimed %.2f." % (got3, CLAIMED_EXPONENT))


def t_mirror():
    """T-mirror: this loop reproduces probe_quotesize_lever.run exactly"""
    shared = ("fill_share", "vol_share", "mm_fills", "mm_spread", "width_bp",
              "levels", "agg_per_s", "near_ns", "near_mm", "near_mm_share")
    t0 = time.time()
    mine = run(SELFTEST_SEED, SELFTEST_QSIZE)
    theirs = probe_quotesize_lever.run(SELFTEST_SEED, SELFTEST_QSIZE)
    for k in shared:
        a, b = mine[k], theirs[k]
        assert abs(a - b) <= MIRROR_TOL * max(1.0, abs(b)), \
            "MIRROR MISMATCH on %s: mine %.12g vs probe %.12g" % (k, a, b)
    assert probe_quotesize_lever.T == T, "probe T differs from this module's"
    assert probe_quotesize_lever.GAMMA_CAP == GAMMA_CAP
    print("  T-mirror    PASS  all %d shared quantities agree with "
          "probe_quotesize_lever.run to %.0e" % (len(shared), MIRROR_TOL))
    print("                    at seed %d, quote_size %.3f (%.0fs). fill "
          "share %.4f both ways; width %.6f"
          % (SELFTEST_SEED, SELFTEST_QSIZE, time.time() - t0,
             mine["fill_share"], mine["width_bp"]))
    print("                    both ways. The transcribed loop is the probe's "
          "loop plus columns.")
    return mine


def t_nonempty(cell):
    """T-nonempty: the run produced non-degenerate output, so a clean-looking"""
    assert cell["mm_fills"] > 0, "the MM never filled"
    assert cell["n_q"] > 0.9 * T, "inventory sampled on only %.0f of %.0f secs" \
        % (cell["n_q"], T)
    assert cell["sd_q"] > 0.0, "inventory never moved"
    assert cell["max_q"] >= cell["sd_q"], "max|q| below sd(q) is impossible"
    assert cell["width_bp"] > 0.0 and math.isfinite(cell["width_bp"])
    assert math.isfinite(cell["shape"]) and cell["shape"] >= 1.0, \
        "mean/median shape %r is not >= 1 for a right-skewed spread" \
        % cell["shape"]
    assert cell["agg_per_s"] > 0.0
    print("  T-nonempty  PASS  %.0f MM fills, %.0f inventory samples, "
          "sd(q)=%.5f, max|q|=%.5f," % (cell["mm_fills"], cell["n_q"],
                                        cell["sd_q"], cell["max_q"]))
    print("                    width %.5f bp, shape %.4f, rate %.5f/s; "
          "nothing degenerate."
          % (cell["width_bp"], cell["shape"], cell["agg_per_s"]))


def selftest():
    print("=" * 126)
    print("Self-tests; quotesize_verdict.py. The arm has NOT run.")
    print("=" * 126)
    t_ident()
    t_bands()
    t_verdict()
    t_pass()
    t_exponent()
    cell = t_mirror()
    t_nonempty(cell)
    print("")
    print("  All 7 self-tests PASSED.")
    print("  Note what T-mirror costs and buys: it runs two full %.0fs "
          "simulations to prove the" % T)
    print("  transcribed loop is the probe's loop. Transcription is the defect "
          "class this project")
    print("  keeps hitting, and a visual diff has already failed to catch it "
          "once.")
    return 0


def main():
    print("=" * 126)
    print("Does the quote_size ceiling survive the project's own verdict "
          "standard?; 8 seeds, intervals, sd(q)")
    print("=" * 126)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("k=%.5f. gamma=%.2e (C=%.2f), the same binding viability cap the "
          "probe used, passed as a"
          % (K, GAMMA_CAP, GAMMA_CAP * C_PER_GAMMA))
    print("constructor argument. DEFAULT_GAMMA=%g and DEFAULT_QUOTE_SIZE=%g "
          "untouched. %d seeds x %.0fs"
          % (DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE, len(SEEDS), T))
    print("per size, book scanned every %ds. quote_size is the only thing that "
          "varies." % SAMPLE_EVERY)
    print("")
    print("Pre-registered before any cell ran: does any quote_size clear all "
          "four committed targets")
    print("while giving sd(q) at least %.1fx the committed %.5f BTC, i.e. "
          "%.5f BTC, with the whole"
          % (SD_FACTOR, SD_REF, SD_MIN))
    print("95% interval above that threshold? REOPENED / UNRESOLVED / BOUNDED "
          "/ OUT OF GRID, and a")
    print("BOUNDED result is a stronger claim than the null at one point, not "
          "a weaker one.")
    print("")

    rows = []
    for qs in QUOTE_SIZES:
        r = measure(qs)
        rows.append(r)
        print("  quote_size=%.3f BTC  measured in %4.0fs   fill share=%5.2f%%"
              "   vol share=%5.2f%%   sd(q)=%.5f"
              % (qs, r["secs"], r["fill_share"], r["vol_share"], r["sd_q"]),
              flush=True)
    print("")

    print("=" * 126)
    print("1. THE FOUR COMMITTED TARGETS, scored by clipped_window.verdict "
          "(imported, not reimplemented)")
    print("=" * 126)
    cells = {}
    for r in rows:
        print("  quote_size=%.3f" % r["qsize"])
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
    print("2. sd(q) and max|q|; never measured above quote_size=0.020 "
          "anywhere in this project")
    print("=" * 126)
    print("  %11s %20s %10s %14s %14s %13s"
          % ("quote_size", "sd(q) BTC", "95% lo", "sd(q)/qsize", "max|q| BTC",
             "max|q|/qsize"))
    for r in rows:
        lo = r["sd_q"] - Z * r["sd_q_se"]
        print("  %11.3f %11.5f +/- %-6.5f %10.5f %14.3f %14.5f %13.3f"
              % (r["qsize"], r["sd_q"], r["sd_q_se"], lo,
                 r["sd_q"] / r["qsize"], r["max_q"], r["max_q"] / r["qsize"]))
    print("")
    anchor = rows[0]
    print("  cross-check against the one recorded point. "
          "participation_skew_results.txt line 92")
    print("  records sd(q) = %.5f at quote_size 0.020 (5 seeds, gamma "
          "9.37e-06). Here, at 8 seeds" % SD_REF)
    print("  and gamma %.2e: %.5f +/- %.5f. Difference %+.5f BTC (%+.1f%% of "
          "the recorded value)."
          % (GAMMA_CAP, anchor["sd_q"], anchor["sd_q_se"],
             anchor["sd_q"] - SD_REF,
             100.0 * (anchor["sd_q"] - SD_REF) / SD_REF))
    print("  The two are not the same experiment; different seed count, "
          "marginally different gamma --")
    print("  so this is a plausibility check, not a replication, and it is not "
          "asserted anywhere.")
    print("")
    exps = log_slope([r["qsize"] for r in rows], [r["sd_q"] for r in rows])
    print("  Fitted exponent of sd(q) against quote_size over %.3f -> %.3f "
          "(a %.1fx range): %.4f"
          % (rows[0]["qsize"], rows[-1]["qsize"],
             rows[-1]["qsize"] / rows[0]["qsize"], exps))
    print("  The value claimed from the committed three points, over a "
          "different and non-overlapping")
    print("  4x range below this one, was %.2f. Difference %+.4f."
          % (CLAIMED_EXPONENT, exps - CLAIMED_EXPONENT))
    if abs(exps - CLAIMED_EXPONENT) < 0.05:
        print("  The power law extrapolates: one exponent describes sd(q) "
              "across 0.005 -> 0.080 BTC.")
    else:
        print("  The power law does NOT extrapolate cleanly across the two "
              "ranges, so any sd(q)")
        print("  projected beyond 0.080 BTC from either exponent is an "
              "extrapolation and not a measurement.")
    print("")

    print("=" * 126)
    print("3. The pre-registered verdict")
    print("=" * 126)
    per_cell = []
    for r in rows:
        c = classify(cells[r["qsize"]], r["sd_q"], r["sd_q_se"])
        per_cell.append(c)
        lo = r["sd_q"] - Z * r["sd_q_se"]
        print("  quote_size=%.3f   sd(q) 95%% lo = %.5f vs threshold %.5f   "
              "-> %s" % (r["qsize"], lo, SD_MIN, c))
    print("")
    out = overall(per_cell)
    print("  overall: %s" % out)
    print("")
    if out == "REOPENED":
        w = [rows[i]["qsize"] for i, c in enumerate(per_cell)
             if c == "REOPENS"]
        print("  At least one quote_size clears all four committed targets and "
              "reaches at least %.1fx the" % SD_FACTOR)
        print("  committed inventory scale: %s BTC."
              % ", ".join("%.3f" % x for x in w))
        print("  The damage axis has a benefit axis again. The maker can be "
              "run at that size inside the")
        print("  calibration, and a damage measurement there is not the same "
              "measurement as the null.")
    elif out == "UNRESOLVED":
        w = [rows[i]["qsize"] for i, c in enumerate(per_cell)
             if c == "UNRESOLVED"]
        print("  No size is 4/4 MET, but %s BTC clears the inventory threshold "
              "with no target MISSED --"
              % ", ".join("%.3f" % x for x in w))
        print("  only UNRESOLVED ones. The bound may not be claimed. The "
              "probe's ceiling does not survive")
        print("  the project's own standard, and it does not survive into a "
              "clean re-opening either.")
        print("  More seeds at those sizes would settle it; nothing else "
              "would.")
    elif out == "BOUNDED":
        w = [rows[i]["qsize"] for i, c in enumerate(per_cell)
             if c == "BOUNDED"]
        print("  No size is 4/4 MET, and every size reaching %.1fx the "
              "committed inventory scale (%s BTC)"
              % (SD_FACTOR, ", ".join("%.3f" % x for x in w)))
        print("  misses at least one committed target with its whole 95% "
              "interval outside the band.")
        print("  The bound is the result: the maker's inventory cannot be "
              "doubled without decalibrating")
        print("  the book, so damage is null across the whole "
              "calibration-feasible range of maker")
        print("  participation; not merely at one point. That is the "
              "stronger claim and it answers the")
        print("  obvious reviewer objection directly.")
    else:
        print("  No size tested reached the inventory threshold of %.5f BTC at "
              "all. The grid was chosen" % SD_MIN)
        print("  wrong and nothing is concluded in either direction. The "
              "exponent fitted above says where")
        print("  the threshold would be reached; that is the only thing this "
              "run establishes.")
    print("")

    print("=" * 126)
    print("What this run does not do, and what limits it")
    print("=" * 126)
    print("  L1  It does not measure damage. It measures the range over which "
          "a damage measurement")
    print("      could be made without decalibrating the book. Phase 7 is not "
          "run.")
    print("  L2  It scores against the real-data targets (clipped_window."
          "Targets), not against the")
    print("      simulator's MM-absent output. On levels/side those references "
          "differ a lot; %.2f" % REAL_LEVELS)
    print("      against %.2f; so this column reads differently from the "
          "probe's by construction." % ABSENT["levels_per_side"])
    l_floor = REAL_LEVELS * (1.0 - TOL_LEVELS)
    print("      The MM-absent simulator sits only %.1f%% above the levels "
          "band floor of %.2f before"
          % (100.0 * (ABSENT["levels_per_side"] / l_floor - 1.0), l_floor))
    print("      quote_size does anything, so a levels miss here would not be "
          "attributable to the maker.")
    print("  L3  Near-touch depth is NOT one of the four scored targets and is "
          "not scored. The probe")
    print("      reported it; clipped_window never scored it. Adding it as a "
          "fifth target would be a")
    print("      new gate, and new gates are not invented inside a run.")
    print("  L4  The MM is a large fraction of its own near-touch measurement "
          "bucket at these sizes")
    print("      (the probe measured 4.4% at 0.050 and 19.8% at 0.200). A "
          "size that passes the four")
    print("      targets can still be rebuilding the near-touch book rather "
          "than trading into it.")
    print("  L5  T=%.0fs per cell, so sd(q) is measured over one day while the "
          "damage A/B runs seven." % T)
    print("      Inventory half-lives here are 3-9h, so a day contains only a "
          "few, and sd(q) at 7 days")
    print("      could be larger. That would make this test conservative about "
          "reaching the threshold.")
    print("  L6  gamma is held at the binding viability cap. sd(q) moves only "
          "1.33x across five decades")
    print("      of gamma, so this is not a material free parameter; but it "
          "is held, not swept.")
    print("  L7  No committed default changed, probe_quotesize_lever.py was "
          "not modified, and the")
    print("      measurement loop is asserted identical to the probe's by "
          "T-mirror.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    main()
