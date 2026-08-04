# clipped_lam_sweep.py: clipped placement, phase 3: lam sweep to reach the real touch width

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world, TARGET

SIGMA, SPY = 0.3721, 365 * 24 * 3600
T = 86400.0
DISP = 0.0020
LIFE = 180.0
PRODUCT = 0.01804
REAL_UPDATE_RATE = 3.09

REAL_MEDIAN_BP = 0.29
REAL_LEVELS_PER_SIDE = 666.34
REAL_NEAR_BTC = 0.0452
REAL_NEAR_LEVELS = 2.15

ROWS = [(0.2255, 0.0800), (0.451, 0.0400), (0.902, 0.0200),
        (1.804, 0.0100), (2.255, 0.0080), (4.510, 0.0040)]


def run(seed, lam, p_market, near_touch=False):
    _p, vf, eng, noise, informed = make_world(seed, T, lam, p_market, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())

    n_event = n_agg = n_fill = 0
    n_limit = n_clipped_at_end = 0
    spreads_bp = []
    orders_side, levels_side = [], []
    near_btc, near_lvl = [], []

    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.order_type == "limit":
                n_limit += 1
            if r.fills:
                n_event += 1
                n_agg += len(set(f.price for f in r.fills))
                n_fill += len(r.fills)
        for r in informed.run_until(eng, t, vf):
            if getattr(r, "fills", None):
                n_event += 1
                n_agg += len(set(f.price for f in r.fills))
                n_fill += len(r.fills)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mid = 0.5 * (bb + ba)
        spreads_bp.append(1e4 * (ba - bb) / mid)

        nb = sum(len(q) for q in eng.bids.values())
        na = sum(len(q) for q in eng.asks.values())
        lb, la = len(eng.bids), len(eng.asks)
        for oid in seed_ids:
            o = eng.orders.get(oid)
            if o is None:
                continue
            if o.side == "buy":
                nb -= 1
                if len(eng.bids.get(o.price, ())) == 1:
                    lb -= 1
            else:
                na -= 1
                if len(eng.asks.get(o.price, ())) == 1:
                    la -= 1
        orders_side.append(0.5 * (nb + na))
        levels_side.append(0.5 * (lb + la))

        if near_touch:
            lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
            qb = sum(o.size for p, q in eng.bids.items() if p >= lo
                     for o in q if o.id not in seed_ids)
            qa = sum(o.size for p, q in eng.asks.items() if p <= hi
                     for o in q if o.id not in seed_ids)
            cb = sum(1 for p, q in eng.bids.items() if p >= lo
                     and any(o.id not in seed_ids for o in q))
            ca = sum(1 for p, q in eng.asks.items() if p <= hi
                     and any(o.id not in seed_ids for o in q))
            near_btc.append(0.5 * (qb + qa))
            near_lvl.append(0.5 * (cb + ca))

    n_clipped_at_end = getattr(noise, "n_clipped", 0)
    out = {
        "spreads_bp": spreads_bp,
        "orders_per_side": statistics.mean(orders_side),
        "levels_per_side": statistics.mean(levels_side),
        "ev_per_s": n_event / T, "agg_per_s": n_agg / T,
        "n_agg": n_agg, "n_fill": n_fill,
        "clipped_pct": 100.0 * n_clipped_at_end / max(1, n_limit),
    }
    if near_touch:
        out["near_btc"] = statistics.mean(near_btc)
        out["near_lvl"] = statistics.mean(near_lvl)
    return out


def measure(lam, p_market, seeds, near_touch=False):
    pooled = []
    keys = ["orders_per_side", "levels_per_side", "ev_per_s", "agg_per_s",
            "clipped_pct"] + (["near_btc", "near_lvl"] if near_touch else [])
    acc = {k: [] for k in keys}
    n_agg = n_fill = 0
    t0 = time.time()
    for s in seeds:
        x = run(s, lam, p_market, near_touch)
        pooled += x["spreads_bp"]
        for k in keys:
            acc[k].append(x[k])
        n_agg += x["n_agg"]; n_fill += x["n_fill"]
    pooled.sort()
    out = {k: statistics.mean(v) for k, v in acc.items()}
    out.update(median_bp=pooled[len(pooled) // 2],
               mean_bp=statistics.mean(pooled),
               n_agg=n_agg, n_fill=n_fill, secs=time.time() - t0)
    return out


def main():
    coh = SIGMA * math.sqrt(LIFE / SPY) / DISP
    print("=" * 118)
    print("Phase 3; lam/p_market tradeoff at fixed trade rate, JOIN clipping")
    print("=" * 118)
    print("Held: disp=%.4f, mean_lifetime=%.0fs (both committed values), "
          "lam*p_market=%.5f. MM absent." % (DISP, LIFE, PRODUCT))
    print("3 seeds x %.0fs. Target %.4f aggTrades/s. Real median touch %.2fbp, "
          "real levels/side %.0f (bid)."
          % (T, TARGET, REAL_MEDIAN_BP, REAL_LEVELS_PER_SIDE))
    print("Coherence ratio is constant at %.3f down the sweep (disp and life "
          "both held); not a result." % coh)
    print("")

    hdr = ("  %-9s %-8s %9s %9s %8s %8s %8s %9s %9s %8s %8s %7s"
           % ("lam", "p_mkt", "median bp", "mean bp", "ord/sd", "lvl/sd",
              "ev/s", "aggTr/s", "vs .0451", "agg/ev", "clip%", "coh"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    first_029 = first_666 = first_309 = None
    for lam, pm in ROWS:
        r = measure(lam, pm, [0, 1, 2])
        r["lam"], r["pm"] = lam, pm
        rows.append(r)
        k = r["agg_per_s"] / r["ev_per_s"] if r["ev_per_s"] else float("nan")
        flags = []
        if first_029 is None and r["median_bp"] <= REAL_MEDIAN_BP:
            first_029 = r; flags.append("<= 0.29bp")
        if first_666 is None and r["levels_per_side"] >= REAL_LEVELS_PER_SIDE:
            first_666 = r; flags.append(">= 666 lvl")
        if first_309 is None and lam >= REAL_UPDATE_RATE:
            first_309 = r; flags.append("lam > 3.09/s")
        print("  %-9.4f %-8.4f %9.4f %9.4f %8.1f %8.1f %8.4f %9.4f %8.2fx "
              "%8.4f %7.1f%% %7.3f%s"
              % (lam, pm, r["median_bp"], r["mean_bp"], r["orders_per_side"],
                 r["levels_per_side"], r["ev_per_s"], r["agg_per_s"],
                 r["agg_per_s"] / TARGET, k, r["clipped_pct"], coh,
                 ("   <== " + ", ".join(flags)) if flags else ""))

    print("")
    print("Flags")
    if first_029:
        print("  first median <= %.2fbp        : lam=%.4f pm=%.4f -> %.4f bp "
              "(mean %.4f), rate %.2fx target"
              % (REAL_MEDIAN_BP, first_029["lam"], first_029["pm"],
                 first_029["median_bp"], first_029["mean_bp"],
                 first_029["agg_per_s"] / TARGET))
    else:
        print("  median never reaches %.2fbp in this sweep." % REAL_MEDIAN_BP)
    if first_666:
        print("  first levels/side >= %.0f     : lam=%.4f pm=%.4f -> %.1f "
              "levels/side" % (REAL_LEVELS_PER_SIDE, first_666["lam"],
                               first_666["pm"], first_666["levels_per_side"]))
    else:
        print("  levels/side never reaches %.0f in this sweep (max %.1f)."
              % (REAL_LEVELS_PER_SIDE, max(r["levels_per_side"] for r in rows)))
    print("  lam crosses the real ~%.2f IDs/s book update rate at lam=%.3f "
          "(see caveat in header)."
          % (REAL_UPDATE_RATE, first_309["lam"] if first_309 else float("nan")))

    cand = None
    for r in rows:
        if r["median_bp"] <= REAL_MEDIAN_BP and r["agg_per_s"] <= TARGET:
            cand = r
            break
    print("")
    print("=" * 118)
    if cand is None:
        print("Follow-up skipped; no row reached %.2fbp with the rate on or "
              "under target." % REAL_MEDIAN_BP)
        print("=" * 118)
        return
    print("Follow-up; lam=%.4f pm=%.4f re-run at 5 seeds, with near-touch "
          "depth" % (cand["lam"], cand["pm"]))
    print("=" * 118)
    r5 = measure(cand["lam"], cand["pm"], [0, 1, 2, 3, 4], near_touch=True)
    print("  %-26s %12s %12s" % ("", "sim (5 seeds)", "real (Ph3)"))
    print("  %-26s %12.4f %12.4f" % ("median touch bp", r5["median_bp"], REAL_MEDIAN_BP))
    print("  %-26s %12.4f %12s" % ("mean touch bp", r5["mean_bp"], "0.5436"))
    print("  %-26s %12.1f %12.1f" % ("levels/side (total)", r5["levels_per_side"],
                                     REAL_LEVELS_PER_SIDE))
    print("  %-26s %12.1f %12s" % ("orders/side (total)", r5["orders_per_side"], "n/a"))
    print("  %-26s %12.4f %12.4f" % ("near-touch BTC (<=1bp)", r5["near_btc"],
                                     REAL_NEAR_BTC))
    print("  %-26s %12.2f %12.2f" % ("near-touch levels (<=1bp)", r5["near_lvl"],
                                     REAL_NEAR_LEVELS))
    print("  %-26s %12.4f %12.4f" % ("aggTrades/s", r5["agg_per_s"], TARGET))
    print("  %-26s %12.2fx %12s" % ("  vs target", r5["agg_per_s"] / TARGET, "1.00x"))
    print("")
    print("  ratios sim/real: width %.2fx  levels %.2fx  near-touch BTC %.2fx  "
          "near-touch levels %.2fx"
          % (r5["median_bp"] / REAL_MEDIAN_BP,
             r5["levels_per_side"] / REAL_LEVELS_PER_SIDE,
             r5["near_btc"] / REAL_NEAR_BTC,
             r5["near_lvl"] / REAL_NEAR_LEVELS))


if __name__ == "__main__":
    main()
