# clipped_depth_sweep.py: clipped placement, phase 4: (mean_lifetime, disp) sweep for near-touch depth

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
SEEDS = [0, 1, 2]
SAMPLE_EVERY = 60

LAM = 1.804
P_MARKET = 0.0100
LAM_CEILING = 3.09

LIVES = (180.0, 360.0, 720.0, 1440.0)
DISPS = (0.0020, 0.0040, 0.0080, 0.0120)

REAL_MEDIAN_BP = 0.2873
REAL_MEAN_BP = 0.5436
REAL_LEVELS = 666.34
REAL_NEAR_BTC = 0.0452
REAL_NEAR_LEVELS = 2.15
REAL_TOUCH_AGE = 247.0

TOL_WIDTH, TOL_LEVELS, TOL_RATE = 0.20, 0.20, 0.10


def run(seed, lam, p_market, life, disp):
    _p, vf, eng, noise, informed = make_world(seed, T, lam, p_market, life,
                                              disp, "join")
    seed_ids = set(eng.orders.keys())
    birth = {oid: 0.0 for oid in seed_ids}

    clock = [0.0]
    _orig_add = eng.add_limit_order

    def _add(side, price, size, agent_id):
        res = _orig_add(side, price, size, agent_id)
        if res.resting_size > 0:
            birth[res.order_id] = clock[0]
        return res
    eng.add_limit_order = _add

    n_event = n_agg = n_fill = n_limit = 0
    spreads_bp, touch_ages = [], []
    orders_side, levels_side, near_btc, near_lvl, all_ages = [], [], [], [], []
    n_two_sided = 0

    t = 0.0
    while t < T:
        t += 1.0
        clock[0] = t
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
        n_two_sided += 1
        mid = 0.5 * (bb + ba)
        spreads_bp.append(1e4 * (ba - bb) / mid)

        for book, px in ((eng.bids, bb), (eng.asks, ba)):
            q = book.get(px)
            if q:
                touch_ages.append(t - birth.get(q[0].id, t))

        if t % SAMPLE_EVERY:
            continue

        nb = na = lb = la = 0
        qb = qa = 0.0
        cb = ca = 0
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        ages = []
        for book, is_bid in ((eng.bids, True), (eng.asks, False)):
            for p, q in book.items():
                live = [o for o in q if o.id not in seed_ids]
                if not live:
                    continue
                n = len(live)
                if is_bid:
                    nb += n; lb += 1
                else:
                    na += n; la += 1
                for o in live:
                    ages.append(t - birth.get(o.id, t))
                if (is_bid and p >= lo) or ((not is_bid) and p <= hi):
                    s = sum(o.size for o in live)
                    if is_bid:
                        qb += s; cb += 1
                    else:
                        qa += s; ca += 1
        orders_side.append(0.5 * (nb + na))
        levels_side.append(0.5 * (lb + la))
        near_btc.append(0.5 * (qb + qa))
        near_lvl.append(0.5 * (cb + ca))
        if ages:
            ages.sort()
            all_ages.append(ages[len(ages) // 2])

    touch_ages.sort()
    return {
        "spreads_bp": spreads_bp,
        "orders_per_side": statistics.mean(orders_side),
        "levels_per_side": statistics.mean(levels_side),
        "near_btc": statistics.mean(near_btc),
        "near_lvl": statistics.mean(near_lvl),
        "touch_age": touch_ages[len(touch_ages) // 2] if touch_ages else float("nan"),
        "all_age": statistics.mean(all_ages) if all_ages else float("nan"),
        "ev_per_s": n_event / T, "agg_per_s": n_agg / T,
        "n_agg": n_agg, "n_fill": n_fill,
        "clipped_pct": 100.0 * getattr(noise, "n_clipped", 0) / max(1, n_limit),
        "two_sided_pct": 100.0 * n_two_sided / T,
    }


def measure(life, disp, seeds=SEEDS):
    pooled = []
    keys = ["orders_per_side", "levels_per_side", "near_btc", "near_lvl",
            "touch_age", "all_age", "ev_per_s", "agg_per_s", "clipped_pct",
            "two_sided_pct"]
    acc = {k: [] for k in keys}
    n_agg = n_fill = 0
    t0 = time.time()
    for s in seeds:
        x = run(s, LAM, P_MARKET, life, disp)
        pooled += x["spreads_bp"]
        for k in keys:
            acc[k].append(x[k])
        n_agg += x["n_agg"]; n_fill += x["n_fill"]
    pooled.sort()
    out = {k: statistics.mean(v) for k, v in acc.items()}
    out.update(median_bp=pooled[len(pooled) // 2],
               mean_bp=statistics.mean(pooled),
               n_agg=n_agg, n_fill=n_fill,
               life=life, disp=disp,
               coh=SIGMA * math.sqrt(life / SPY) / disp,
               secs=time.time() - t0)
    out["agg_over_ev"] = (out["agg_per_s"] / out["ev_per_s"]
                          if out["ev_per_s"] else float("nan"))
    return out


def hits(r):
    return (abs(r["median_bp"] - REAL_MEDIAN_BP) <= TOL_WIDTH * REAL_MEDIAN_BP,
            abs(r["levels_per_side"] - REAL_LEVELS) <= TOL_LEVELS * REAL_LEVELS,
            abs(r["agg_per_s"] - TARGET) <= TOL_RATE * TARGET)


HDR = ("  %-7s %-7s %8s %8s %8s %8s %9s %7s %7s %8s %8s %7s %7s %7s %6s  %s"
       % ("life", "disp", "med bp", "mean bp", "ord/sd", "lvl/sd", "nearBTC",
          "nearLv", "ev/s", "aggTr/s", "vs.0451", "agg/ev", "clip%",
          "tchAge", "coh", "flags"))


def show(r):
    w, l, a = hits(r)
    flags = ("W" if w else ".") + ("L" if l else ".") + ("R" if a else ".")
    return ("  %-7.0f %-7.4f %8.4f %8.4f %8.1f %8.1f %9.4f %7.2f %7.4f %8.4f "
            "%7.2fx %7.3f %6.1f%% %7.0f %6.3f  %s"
            % (r["life"], r["disp"], r["median_bp"], r["mean_bp"],
               r["orders_per_side"], r["levels_per_side"], r["near_btc"],
               r["near_lvl"], r["ev_per_s"], r["agg_per_s"],
               r["agg_per_s"] / TARGET, r["agg_over_ev"], r["clipped_pct"],
               r["touch_age"], r["coh"], flags))


def main():
    print("=" * 132)
    print("Phase 4; buy depth with mean_lifetime, re-widen with disp, at "
          "lam pinned below its physical ceiling")
    print("=" * 132)
    print("JOIN clipping, MM absent. lam=%.4f p_market=%.4f held (Phase 3's "
          "committed-rate point)." % (LAM, P_MARKET))
    print("lam is %.0f%% of the ~%.2f IDs/s real update-rate ceiling, which "
          "bounds submissions from above." % (100 * LAM / LAM_CEILING, LAM_CEILING))
    print("%d seeds x %.0fs. Full-book stats sampled every %ds; touch stats "
          "every 1s." % (len(SEEDS), T, SAMPLE_EVERY))
    print("Targets: median %.4fbp (+/-%.0f%%), levels/side %.1f (+/-%.0f%%), "
          "%.4f aggTr/s (+/-%.0f%%)."
          % (REAL_MEDIAN_BP, 100 * TOL_WIDTH, REAL_LEVELS, 100 * TOL_LEVELS,
             TARGET, 100 * TOL_RATE))
    print("Flags column: W=width on target, L=levels on target, R=rate on "
          "target; '...' = none.")
    print("")
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))

    rows = []
    for life in LIVES:
        for disp in DISPS:
            r = measure(life, disp)
            rows.append(r)
            print(show(r), flush=True)
        print("", flush=True)

    print("=" * 132)
    winners = [r for r in rows if all(hits(r))]
    if winners:
        print("Cells hitting all three targets:")
        for r in winners:
            print("  life=%.0f disp=%.4f -> median %.4fbp (%.2fx real), "
                  "levels/side %.1f (%.2fx), %.4f aggTr/s (%.2fx)"
                  % (r["life"], r["disp"], r["median_bp"],
                     r["median_bp"] / REAL_MEDIAN_BP, r["levels_per_side"],
                     r["levels_per_side"] / REAL_LEVELS, r["agg_per_s"],
                     r["agg_per_s"] / TARGET))
    else:
        print("no cell hit all three. Closest by each criterion:")
        for name, key, tgt in (("width", "median_bp", REAL_MEDIAN_BP),
                               ("levels", "levels_per_side", REAL_LEVELS),
                               ("rate", "agg_per_s", TARGET)):
            b = min(rows, key=lambda z: abs(z[key] - tgt))
            print("  best %-7s life=%.0f disp=%.4f -> %s=%.4f (%.2fx target)"
                  % (name, b["life"], b["disp"], key, b[key], b[key] / tgt))

    print("")
    print("age cost of a longer lifetime  (L=180 was chosen to pull the "
          "touch-setting order's median")
    print("age near the %.0fs seen in the MM-less world)" % REAL_TOUCH_AGE)
    print("  %-7s %12s %12s %14s" % ("life", "touch age", "all-order", "vs 247s"))
    for life in LIVES:
        sel = [r for r in rows if r["life"] == life]
        ta = statistics.mean(r["touch_age"] for r in sel)
        aa = statistics.mean(r["all_age"] for r in sel)
        print("  %-7.0f %11.0fs %11.0fs %13.2fx" % (life, ta, aa, ta / REAL_TOUCH_AGE))

    print("")
    print("coherence sanity check; is the condition really moot under "
          "clipping?")
    print("  If clipping works, nothing crosses: the clipped %% is the share "
          "that would have")
    print("  crossed, and the book should stay two-sided even at coherence "
          "ratios far below 1.")
    print("  %-7s %-7s %8s %9s %12s" % ("life", "disp", "coh", "clipped%", "two-sided%"))
    for r in rows:
        print("  %-7.0f %-7.4f %8.3f %8.1f%% %11.2f%%"
              % (r["life"], r["disp"], r["coh"], r["clipped_pct"],
                 r["two_sided_pct"]))

    print("")
    print("  total wall clock %.0fs" % sum(r["secs"] for r in rows))


if __name__ == "__main__":
    main()
