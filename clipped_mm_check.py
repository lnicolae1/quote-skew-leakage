# clipped_mm_check.py: clipped placement, phase 6 gate: working point with the MM present

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world, TARGET
from clipped_depth_sweep import (LAM, REAL_MEDIAN_BP, REAL_MEAN_BP,
                                 REAL_LEVELS, REAL_NEAR_BTC, REAL_NEAR_LEVELS,
                                 REAL_TOUCH_AGE, TOL_WIDTH, TOL_LEVELS,
                                 TOL_RATE)
from clipped_window import verdict, margin_se, band, Z, REAL_SHAPE, TOL_SHAPE
from market_maker import MarketMaker, DEFAULT_K, DEFAULT_GAMMA

T = 86400.0
SEEDS = list(range(15))
LIFE = 720.0
DISP = 0.0055
P_MARKET = 0.0126
SAMPLE_EVERY = 60
MM_ID = "MM"
MEASURED_K = 0.06747

ABSENT = {"median_bp": 0.28879, "median_bp_se": 0.01276,
          "mean_bp": 0.7081, "levels_per_side": 551.23755,
          "levels_per_side_se": 2.53988, "orders_per_side": 623.3,
          "agg_per_s": 0.04464, "agg_per_s_se": 0.00073,
          "shape": 2.49631, "shape_se": 0.08231,
          "near_btc": 0.1244, "near_lvl": 3.1668, "touch_age": 395.93}


def _resid_best(book, full_best, want_max):
    """Best non-MM price on one side"""
    q = book.get(full_best)
    if q is not None and any(o.agent_id != MM_ID for o in q):
        return full_best
    cand = [p for p, qq in book.items()
            if p != full_best and any(o.agent_id != MM_ID for o in qq)]
    if not cand:
        return None
    return max(cand) if want_max else min(cand)


def run(seed, k):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=k)

    n_event = n_agg = n_fill = 0
    n_mm_fill = 0
    vol_total = vol_mm = 0.0
    full_bp, resid_bp = [], []
    lvl_ns, ord_ns, lvl_all, ord_all = [], [], [], []
    near_btc_ns, near_lvl_ns, near_btc_all = [], [], []
    mm_passive_share, mm_near_share = [], []
    n_resid_two_sided = 0

    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.fills:
                n_event += 1
                n_agg += len(set(f.price for f in r.fills))
                n_fill += len(r.fills)
                for f in r.fills:
                    vol_total += f.size
                    if f.counterparty_id == MM_ID:
                        n_mm_fill += 1
                        vol_mm += f.size
                mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, vf):
            fills = getattr(r, "fills", None)
            if fills:
                n_event += 1
                n_agg += len(set(f.price for f in fills))
                n_fill += len(fills)
                for f in fills:
                    vol_total += f.size
                    if f.counterparty_id == MM_ID:
                        n_mm_fill += 1
                        vol_mm += f.size
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mid = 0.5 * (bb + ba)
        full_bp.append(1e4 * (ba - bb) / mid)

        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is not None and ra is not None and ra > rb:
            n_resid_two_sided += 1
            resid_bp.append(1e4 * (ra - rb) / (0.5 * (rb + ra)))

        if t % SAMPLE_EVERY:
            continue

        n_ns = l_ns = n_all = l_all = 0
        sz_all = sz_mm = 0.0
        qb_ns = qa_ns = 0.0
        cb_ns = ca_ns = 0
        qb_all = qa_all = 0.0
        near_mm = near_all = 0.0
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        for book, is_bid in ((eng.bids, True), (eng.asks, False)):
            for p, q in book.items():
                live_ns = [o for o in q
                           if o.id not in seed_ids and o.agent_id != MM_ID]
                any_all = False
                for o in q:
                    if o.id in seed_ids:
                        continue
                    any_all = True
                    n_all += 1
                    sz_all += o.size
                    if o.agent_id == MM_ID:
                        sz_mm += o.size
                if any_all:
                    l_all += 1
                if live_ns:
                    n_ns += len(live_ns)
                    l_ns += 1
                near = (p >= lo) if is_bid else (p <= hi)
                if near:
                    s_ns = sum(o.size for o in live_ns)
                    s_all = sum(o.size for o in q if o.id not in seed_ids)
                    s_mm = sum(o.size for o in q if o.agent_id == MM_ID)
                    near_mm += s_mm
                    near_all += s_all
                    if is_bid:
                        qb_ns += s_ns; qb_all += s_all
                        if live_ns:
                            cb_ns += 1
                    else:
                        qa_ns += s_ns; qa_all += s_all
                        if live_ns:
                            ca_ns += 1
        ord_ns.append(0.5 * n_ns); lvl_ns.append(0.5 * l_ns)
        ord_all.append(0.5 * n_all); lvl_all.append(0.5 * l_all)
        near_btc_ns.append(0.5 * (qb_ns + qa_ns))
        near_lvl_ns.append(0.5 * (cb_ns + ca_ns))
        near_btc_all.append(0.5 * (qb_all + qa_all))
        if sz_all > 0:
            mm_passive_share.append(100.0 * sz_mm / sz_all)
        if near_all > 0:
            mm_near_share.append(100.0 * near_mm / near_all)

    fs, rs = sorted(full_bp), sorted(resid_bp)
    fmed = fs[len(fs) // 2] if fs else float("nan")
    rmed = rs[len(rs) // 2] if rs else float("nan")
    fmean = statistics.mean(fs) if fs else float("nan")
    rmean = statistics.mean(rs) if rs else float("nan")
    return {
        "full_median_bp": fmed, "full_mean_bp": fmean,
        "full_shape": fmean / fmed if fmed else float("nan"),
        "resid_median_bp": rmed, "resid_mean_bp": rmean,
        "resid_shape": rmean / rmed if rmed else float("nan"),
        "resid_two_sided_pct": 100.0 * n_resid_two_sided / T,
        "levels_ns": statistics.mean(lvl_ns),
        "orders_ns": statistics.mean(ord_ns),
        "levels_all": statistics.mean(lvl_all),
        "orders_all": statistics.mean(ord_all),
        "near_btc_ns": statistics.mean(near_btc_ns),
        "near_lvl_ns": statistics.mean(near_lvl_ns),
        "near_btc_all": statistics.mean(near_btc_all),
        "mm_passive_share": statistics.mean(mm_passive_share) if mm_passive_share else 0.0,
        "mm_near_share": statistics.mean(mm_near_share) if mm_near_share else 0.0,
        "agg_per_s": n_agg / T, "ev_per_s": n_event / T,
        "mm_fill_share": 100.0 * n_mm_fill / max(1, n_fill),
        "mm_vol_share": 100.0 * vol_mm / vol_total if vol_total else 0.0,
        "mm_fills": mm.n_buy_fills + mm.n_sell_fills,
        "mm_q": mm.q,
    }


def measure(k, seeds=SEEDS):
    per = None
    t0 = time.time()
    for s in seeds:
        x = run(s, k)
        if per is None:
            per = {key: [] for key in x}
        for key, v in x.items():
            per[key].append(v)
    n = len(seeds)
    out = {"k": k, "n": n, "secs": time.time() - t0}
    for key, v in per.items():
        out[key] = statistics.mean(v)
        out[key + "_se"] = (statistics.stdev(v) / (n ** 0.5)) if n > 1 else 0.0
    return out


def verdict_block(label, r, width_key, shape_key, levels_key):
    targets = [("median width bp", width_key, REAL_MEDIAN_BP, TOL_WIDTH),
               ("levels/side", levels_key, REAL_LEVELS, TOL_LEVELS),
               ("aggTrades/s", "agg_per_s", TARGET, TOL_RATE),
               ("mean/median shape", shape_key, REAL_SHAPE, TOL_SHAPE)]
    print("  %s" % label)
    met = 0
    margins = []
    for name, key, tgt, tol in targets:
        pt, se = r[key], r[key + "_se"]
        v = verdict(pt, se, tgt, tol)
        met += (v == "MET")
        lo_b, hi_b = band(tgt, tol)
        print("    %-20s %10.5f +/- %-9.5f  95%% [%9.5f, %9.5f]  band "
              "[%9.5f, %9.5f]  %s"
              % (name, pt, se, pt - Z * se, pt + Z * se, lo_b, hi_b, v))
        margins.append((name, margin_se(pt, se, tgt, tol)))
    print("    -> %d/4 MET   margins in SE: %s"
          % (met, ", ".join("%s %.2f" % (n.split()[0], m) for n, m in margins)))
    print("")
    return met, margins


def main():
    print("=" * 126)
    print("Phase 6 gate; the clipped working point with the market maker "
          "present")
    print("=" * 126)
    print("Working point: lam=%.4f  p_market=%.4f  mean_lifetime=%.0fs  "
          "disp=%.4f  JOIN clipping." % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x %.0fs. gamma=DEFAULT_GAMMA=%g (no gamma chosen here). "
          "DEFAULT_K never edited." % (len(SEEDS), T, DEFAULT_GAMMA))
    print("Two k values are run because DEFAULT_K=%.1f is flagged in "
          "market_maker.py as a placeholder" % DEFAULT_K)
    print("while measure_k.py measured %.5f; 2/k = $%.2f vs $%.2f, which are "
          "different books."
          % (MEASURED_K, 2 / DEFAULT_K, 2 / MEASURED_K))
    print("")

    results = {}
    for k, tag in ((DEFAULT_K, "k=DEFAULT_K=1.5 (committed default)"),
                   (MEASURED_K, "k=0.06747 (measure_k.py)")):
        print("=" * 126)
        print("%s" % tag)
        print("=" * 126)
        r = measure(k)
        results[k] = r

        print("  Touch width; full-book vs residual (non-MM), and the "
              "MM-absent baseline from 847a7fa")
        print("    %-26s %10s %10s %10s" % ("", "median bp", "mean bp", "shape"))
        print("    %-26s %10.4f %10.4f %10.2f"
              % ("full book (incl MM)", r["full_median_bp"],
                 r["full_mean_bp"], r["full_shape"]))
        print("    %-26s %10.4f %10.4f %10.2f"
              % ("residual (non-MM)", r["resid_median_bp"],
                 r["resid_mean_bp"], r["resid_shape"]))
        print("    %-26s %10.4f %10.4f %10.2f"
              % ("MM-ABSENT (847a7fa)", ABSENT["median_bp"],
                 ABSENT["mean_bp"], ABSENT["shape"]))
        print("    %-26s %10.4f" % ("real market (Phase 3)", REAL_MEDIAN_BP))
        print("    residual book two-sided %.2f%% of seconds"
              % r["resid_two_sided_pct"])
        print("")

        print("  depth; non-MM only vs including the MM")
        print("    %-26s %12s %12s" % ("", "levels/side", "orders/side"))
        print("    %-26s %12.1f %12.1f"
              % ("non-MM", r["levels_ns"], r["orders_ns"]))
        print("    %-26s %12.1f %12.1f"
              % ("including MM", r["levels_all"], r["orders_all"]))
        print("    %-26s %12.1f %12.1f"
              % ("MM-ABSENT (847a7fa)", ABSENT["levels_per_side"],
                 ABSENT["orders_per_side"]))
        print("    %-26s %12.1f" % ("real market (Phase 3)", REAL_LEVELS))
        print("")

        print("  near-touch (<=1bp of mid)")
        print("    %-26s %12s %12s" % ("", "BTC", "levels"))
        print("    %-26s %12.4f %12.2f"
              % ("non-MM", r["near_btc_ns"], r["near_lvl_ns"]))
        print("    %-26s %12.4f %12s" % ("including MM", r["near_btc_all"], "-"))
        print("    %-26s %12.4f %12.2f"
              % ("MM-ABSENT (847a7fa)", ABSENT["near_btc"], ABSENT["near_lvl"]))
        print("    %-26s %12.4f %12.2f"
              % ("real market (Phase 3)", REAL_NEAR_BTC, REAL_NEAR_LEVELS))
        print("")

        print("  MM footprint; what Phase 6 will depend on")
        print("    share of passive liquidity (all resting size) : %6.2f%% "
              "+/- %.2f" % (r["mm_passive_share"], r["mm_passive_share_se"]))
        print("    share of passive liquidity within 1bp         : %6.2f%% "
              "+/- %.2f" % (r["mm_near_share"], r["mm_near_share_se"]))
        print("    share of trade counterparty (fill count)      : %6.2f%% "
              "+/- %.2f" % (r["mm_fill_share"], r["mm_fill_share_se"]))
        print("    share of traded volume                        : %6.2f%% "
              "+/- %.2f" % (r["mm_vol_share"], r["mm_vol_share_se"]))
        print("    MM fills over the run: %.0f    terminal q: %+.5f BTC"
              % (r["mm_fills"], r["mm_q"]))
        print("    reference: ~99% of trades at k=1.5 and 63% of volume at "
              "the measured k,")
        print("    on a book with ~17.5 orders/side. This book has ~%.0f "
              "levels/side." % r["levels_ns"])
        print("")

        print("  rate: %.5f aggTrades/s +/- %.5f  (%.2fx the %.4f target)   "
              "[MM-absent %.5f]"
              % (r["agg_per_s"], r["agg_per_s_se"],
                 r["agg_per_s"] / TARGET, TARGET, ABSENT["agg_per_s"]))
        print("")

        print("  verdicts")
        verdict_block("against the FULL-BOOK touch (what a real venue shows):",
                      r, "full_median_bp", "full_shape", "levels_all")
        verdict_block("against the RESIDUAL touch (continuity with 847a7fa):",
                      r, "resid_median_bp", "resid_shape", "levels_ns")
        print("  wall clock %.0fs" % r["secs"])
        print("")


if __name__ == "__main__":
    main()
