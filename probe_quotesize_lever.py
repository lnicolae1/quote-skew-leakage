# probe_quotesize_lever.py: is quote_size a clean lever for MM participation?

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from clipped_mm_check import _resid_best, ABSENT, MM_ID

T = 86400.0
SEEDS = list(range(3))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
SAMPLE_EVERY = 60

GAMMA_CAP = 9.4e-6
C_PER_GAMMA = 1.458e6

QUOTE_SIZES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def run(seed, qsize):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA_CAP, quote_size=qsize)

    n_agg = n_fill = n_mm_fill = 0
    vol_total = vol_mm = 0.0
    resid_bp, mm_spreads = [], []
    lvl_ns, near_ns, near_mm_l, near_all_l = [], [], [], []

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
        q_ns = q_mm = q_all = 0.0
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
                    q_all += sum(o.size for o in q if o.id not in seed_ids)
        lvl_ns.append(0.5 * l_ns)
        near_ns.append(0.5 * q_ns)
        near_mm_l.append(0.5 * q_mm)
        near_all_l.append(0.5 * q_all)

    near_mm = statistics.mean(near_mm_l) if near_mm_l else 0.0
    near_all = statistics.mean(near_all_l) if near_all_l else 0.0
    return {
        "fill_share": 100.0 * n_mm_fill / n_fill if n_fill else 0.0,
        "vol_share": 100.0 * vol_mm / vol_total if vol_total else 0.0,
        "mm_fills": float(n_mm_fill),
        "mm_spread": statistics.median(mm_spreads) if mm_spreads else
        float("nan"),
        "width_bp": statistics.median(resid_bp) if resid_bp else float("nan"),
        "levels": statistics.mean(lvl_ns) if lvl_ns else float("nan"),
        "agg_per_s": n_agg / T,
        "near_ns": statistics.mean(near_ns) if near_ns else float("nan"),
        "near_mm": near_mm,
        "near_mm_share": (100.0 * near_mm / near_all) if near_all > 0 else 0.0,
    }


def measure(qsize):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, qsize)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    out = {"qsize": qsize, "secs": time.time() - t0}
    n = len(SEEDS)
    for k, v in per.items():
        out[k] = statistics.mean(v)
        out[k + "_se"] = (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0
    return out


def dev(val, base):
    return 100.0 * (val - base) / base if base else float("nan")


def main():
    print("=" * 126)
    print("Is quote_size a clean lever for MM participation?; probe only, "
          "no participation sweep run")
    print("=" * 126)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("k=%.5f. gamma=%.2e (the log-interpolated binding viability cap, "
          "C=%.2f); a diagnostic"
          % (K, GAMMA_CAP, GAMMA_CAP * C_PER_GAMMA))
    print("setting passed as a constructor argument, NOT a choice. "
          "DEFAULT_GAMMA=%g and DEFAULT_QUOTE_SIZE=%g"
          % (DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE))
    print("are untouched. %d seeds x %.0fs per size, book scanned every %ds."
          % (len(SEEDS), T, SAMPLE_EVERY))
    print("")

    rows = []
    for qs in QUOTE_SIZES:
        r = measure(qs)
        rows.append(r)
        print("  quote_size=%.3f BTC  measured in %3.0fs   fill share=%5.2f%%"
              "   vol share=%5.2f%%" % (qs, r["secs"], r["fill_share"],
                                        r["vol_share"]), flush=True)
    print("")

    print("=" * 126)
    print("Does the lever move participation?")
    print("=" * 126)
    print("  %11s %8s %14s %14s %13s %13s"
          % ("quote_size", "vs dflt", "fill share %", "vol share %",
             "MM fills/day", "MM spread $"))
    for r in rows:
        print("  %11.3f %7.1fx %8.2f+-%-4.2f %8.2f+-%-4.2f %13.1f %13.2f"
              % (r["qsize"], r["qsize"] / DEFAULT_QUOTE_SIZE,
                 r["fill_share"], r["fill_share_se"],
                 r["vol_share"], r["vol_share_se"],
                 r["mm_fills"], r["mm_spread"]))
    lo, hi = rows[0], rows[-1]
    print("")
    print("  Volume share spans %.2f%% -> %.2f%% over a %.0fx range in size "
          "(%.2fx in share)."
          % (lo["vol_share"], hi["vol_share"],
             hi["qsize"] / lo["qsize"],
             hi["vol_share"] / lo["vol_share"] if lo["vol_share"] else
             float("nan")))
    print("  A lever that cannot move participation over a wide range is no "
          "more usable than one")
    print("  that moves it by wrecking the book. Both failure modes matter.")

    print("")
    print("=" * 126)
    print("Does it disturb the book?; against the MM-ABSENT calibrated "
          "baselines (clipped_mm_check.ABSENT)")
    print("=" * 126)
    print("  %11s %11s %8s %11s %8s %11s %8s %11s %8s"
          % ("quote_size", "width bp", "dev", "levels/side", "dev",
             "aggTrade/s", "dev", "near BTC", "dev"))
    print("  %11s %11.5f %8s %11.2f %8s %11.5f %8s %11.4f %8s"
          % ("MM ABSENT", ABSENT["median_bp"], "--",
             ABSENT["levels_per_side"], "--", ABSENT["agg_per_s"], "--",
             ABSENT["near_btc"], "--"))
    for r in rows:
        print("  %11.3f %11.5f %7.1f%% %11.2f %7.1f%% %11.5f %7.1f%% "
              "%11.4f %7.1f%%"
              % (r["qsize"],
                 r["width_bp"], dev(r["width_bp"], ABSENT["median_bp"]),
                 r["levels"], dev(r["levels"], ABSENT["levels_per_side"]),
                 r["agg_per_s"], dev(r["agg_per_s"], ABSENT["agg_per_s"]),
                 r["near_ns"], dev(r["near_ns"], ABSENT["near_btc"])))
    print("")
    print("  All four are non-MM quantities (seed orders excluded, "
          "side-averaged), so any movement")
    print("  across rows is the MM perturbing the book indirectly; by "
          "absorbing sweeps that would")
    print("  otherwise have consumed resting orders. A clean lever leaves "
          "these flat.")

    print("")
    print("=" * 126)
    print("The near-touch ceiling; the MM sits ~0.91bp from mid, inside the "
          "1bp bucket it is measured in")
    print("=" * 126)
    print("  %11s %15s %17s %17s"
          % ("quote_size", "MM near BTC", "MM share of near", "non-MM near"))
    for r in rows:
        print("  %11.3f %15.4f %16.1f%% %17.4f"
              % (r["qsize"], r["near_mm"], r["near_mm_share"], r["near_ns"]))
    print("")
    print("  Calibrated non-MM near-touch depth is %.4f BTC/side. A quote_size "
          "that makes the MM a"
          % ABSENT["near_btc"])
    print("  large fraction of its own measurement bucket is not varying "
          "participation into a fixed")
    print("  book; it is rebuilding the near-touch book, which is a "
          "different experiment.")

    print("")
    print("=" * 126)
    print("What this probe does not do")
    print("=" * 126)
    print("  It does not run the participation sweep and it does not measure "
          "skew effectiveness. It")
    print("  only establishes whether quote_size is usable as the lever, and "
          "over what range. No")
    print("  committed default changed, no gamma chosen, step 7 not run.")


if __name__ == "__main__":
    main()
