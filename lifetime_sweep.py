# lifetime_sweep.py: mean_lifetime sweep: tracking error vs book structure

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from market_maker import (MarketMaker, DEFAULT_QUOTE_SIZE, DEFAULT_SIGMA)
from damage_ab import (T, SEEDS, K, REF_MID, C_TARGET, GAMMA, mean_se,
                       SEC_PER_YEAR)
from clipped_placement import make_world
from clipped_mm_check import MM_ID
from clipped_depth_sweep import (REAL_MEDIAN_BP, REAL_LEVELS, TOL_WIDTH,
                                 TOL_LEVELS, TOL_RATE)

LAM, P_MARKET, DISP = 1.804, 0.0126, 0.0055
LIFETIMES = [60.0, 180.0, 360.0, 720.0, 1440.0, 2880.0]
BASELINE_LIFE = 720.0
TARGET_RATE = 0.0451
H_SHORT = 60
BOOK_EVERY = 600

SIGMA_PER_SQRT_S = DEFAULT_SIGMA * REF_MID / math.sqrt(SEC_PER_YEAR)


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def run(seed, life):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, life,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    n = int(T)
    mid = [0.0] * (n + 2)
    last = REF_MID
    fills = []
    stale_abs = []
    stale_all = []
    spread_pnl = 0.0
    n_agg = 0
    widths, lv_all, lv_1bp = [], [], []

    t = 0.0
    while t < T:
        t += 1.0
        ti = int(t)
        v = vf(t)
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fl = getattr(r, "fills", None)
                if not fl:
                    continue
                n_agg += len(set(f.price for f in fl))
                for f in fl:
                    if f.counterparty_id != MM_ID:
                        continue
                    ps = -1.0 if r.side == "buy" else 1.0
                    spread_pnl += -ps * (f.price - last) * f.size
                    fills.append((ti, ps, f.size, last, v - last))
                mm.on_fills(fl, r.side)
        mm.requote(eng, ti)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            last = 0.5 * (bb + ba)
        mid[ti] = last
        stale_all.append(v - last)
        stale_abs.append(abs(v - last))

        if ti % BOOK_EVERY or bb is None or ba is None:
            continue
        m = 0.5 * (bb + ba)
        nb, na = [], []
        for book, store in ((eng.bids, nb), (eng.asks, na)):
            for p, q in book.items():
                for o in q:
                    if o.id in seed_ids or o.agent_id == MM_ID:
                        continue
                    store.append(p)
                    break
        if not nb or not na:
            continue
        rb, ra = max(nb), min(na)
        if ra <= rb:
            continue
        widths.append(1e4 * (ra - rb) / (0.5 * (rb + ra)))
        lv_all.append(0.5 * (len(nb) + len(na)))
        lo, hi = m * (1 - 1e-4), m * (1 + 1e-4)
        lv_1bp.append(0.5 * (sum(1 for p in nb if p >= lo)
                             + sum(1 for p in na if p <= hi)))

    pnl = mm.mark_to_market(last)
    sd = statistics.stdev(stale_all) if len(stale_all) > 1 else float("nan")

    num = den = 0.0
    nwrong = 0
    for (ti, ps, sz, m0, st) in fills:
        if -ps * st > 0:
            nwrong += 1
        j = ti - 1 + H_SHORT
        if j > n:
            continue
        num += (-ps * (mid[j] - m0)) * sz
        den += sz
    thr = pct(sorted(abs(f[4]) for f in fills), 0.20) if fills else 0.0
    anum = aden = 0.0
    for (ti, ps, sz, m0, st) in fills:
        if abs(st) > thr:
            continue
        j = ti - 1 + H_SHORT
        if j > n:
            continue
        anum += (-ps * (mid[j] - m0)) * sz
        aden += sz

    return {
        "sd_stale": sd,
        "abs_stale": statistics.mean(stale_abs),
        "tau": (sd / SIGMA_PER_SQRT_S) ** 2,
        "pnl": pnl,
        "spread": spread_pnl,
        "inv": pnl - spread_pnl,
        "adv60": (num / den) if den > 0 else float("nan"),
        "agree60": (anum / aden) if aden > 0 else float("nan"),
        "wrong": (100.0 * nwrong / len(fills)) if fills else float("nan"),
        "n_fills": float(len(fills)),
        "width_bp": statistics.median(widths) if widths else float("nan"),
        "levels": statistics.mean(lv_all) if lv_all else float("nan"),
        "lv_1bp": statistics.mean(lv_1bp) if lv_1bp else float("nan"),
        "agg_per_s": n_agg / T,
    }


def measure(life):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, life)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    out = {"life": life, "secs": time.time() - t0}
    for k, v in per.items():
        m, se, _n = mean_se(v)
        out[k], out[k + "_se"] = m, se
    return out


def band(val, target, tol):
    return "in-band" if abs(val - target) <= tol * target else "OUT-OF-BAND"


def main():
    print("=" * 130)
    print("mean_lifetime sweep; is tracking error a free parameter, or "
          "welded to the calibrated book structure?")
    print("=" * 130)
    print("Control arm, no sniffer. %d seeds x %.0fs (%.0f days). k=%.5f, "
          "C=%.2f, gamma=%.4e."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA))
    print("lam=%.4f p_market=%.4f disp=%.4f, JOIN clipping. Only "
          "mean_lifetime varies." % (LAM, P_MARKET, DISP))
    print("sigma_$ per sqrt(s) = %.4f, so implied lag tau = (sd / %.4f)^2."
          % (SIGMA_PER_SQRT_S, SIGMA_PER_SQRT_S))
    print("Book structure measured non-MM and seed-excluded every %ds, the "
          "calibration's own convention." % BOOK_EVERY)
    print("")

    rows = []
    for L in LIFETIMES:
        r = measure(L)
        rows.append(r)
        print("  life=%6.0fs  %4.0fs   sd(V-mid)=%8.2f   tau=%8.0fs   "
              "PnL=%+9.2f   width=%.4fbp"
              % (L, r["secs"], r["sd_stale"], r["tau"], r["pnl"],
                 r["width_bp"]), flush=True)
    print("")

    print("=" * 130)
    print("A. Does tracking error scale with mean_lifetime?")
    print("=" * 130)
    print("  %10s %20s %20s %14s %12s"
          % ("life (s)", "sd(V-mid) $", "mean |V-mid| $", "implied tau s",
             "tau / life"))
    for r in rows:
        print("  %10.0f %9.2f +/- %-8.2f %9.2f +/- %-8.2f %14.0f %12.2f"
              % (r["life"], r["sd_stale"], r["sd_stale_se"], r["abs_stale"],
                 r["abs_stale_se"], r["tau"], r["tau"] / r["life"]))
    print("")
    print("  A roughly constant tau/life column means the lag is the order "
          "lifetime up to a constant,")
    print("  and mean_lifetime is the lever. A tau that barely moves means "
          "the lag comes from somewhere")
    print("  else and this parameter is not it.")

    print("")
    print("=" * 130)
    print("B. Does the maker become viable?")
    print("=" * 130)
    print("  %10s %20s %12s %12s %10s %14s %14s %10s"
          % ("life (s)", "MM PnL $", "spread", "INV", "INV %", "adverse h=60",
             "agree-bin", "wrong %"))
    for r in rows:
        tot = abs(r["spread"]) + abs(r["inv"])
        print("  %10.0f %9.2f +/- %-8.2f %12.2f %12.2f %9.1f%% %14.2f "
              "%14.2f %9.2f%%"
              % (r["life"], r["pnl"], r["pnl_se"], r["spread"], r["inv"],
                 (100.0 * abs(r["inv"]) / tot) if tot > 0 else float("nan"),
                 r["adv60"], r["agree60"], r["wrong"]))
    print("")
    print("  wrong %% is the share of fills taken on the wrong side of V; "
          "50%% is no directional")
    print("  selection. agree-bin is the residual adverse move where V and "
          "the mid agree; the ~28%%")
    print("  of the bleed that staleness does NOT explain (00b1773).")

    print("")
    print("=" * 130)
    print("C. What it costs the calibration; 847a7fa's own three targets")
    print("=" * 130)
    print("  targets: width %.4fbp +/-%.0f%%, levels/side %.2f +/-%.0f%%, "
          "aggTrade %.4f/s +/-%.0f%%"
          % (REAL_MEDIAN_BP, 100 * TOL_WIDTH, REAL_LEVELS, 100 * TOL_LEVELS,
             TARGET_RATE, 100 * TOL_RATE))
    print("")
    print("  %10s %12s %13s %12s %13s %12s %13s %12s"
          % ("life (s)", "width bp", "verdict", "levels/side", "verdict",
             "aggTrade/s", "verdict", "levels/1bp"))
    for r in rows:
        print("  %10.0f %12.4f %13s %12.1f %13s %12.5f %13s %12.2f"
              % (r["life"], r["width_bp"],
                 band(r["width_bp"], REAL_MEDIAN_BP, TOL_WIDTH),
                 r["levels"], band(r["levels"], REAL_LEVELS, TOL_LEVELS),
                 r["agg_per_s"], band(r["agg_per_s"], TARGET_RATE, TOL_RATE),
                 r["lv_1bp"]))

    print("")
    print("=" * 130)
    print("D. The verdict")
    print("=" * 130)
    base = [r for r in rows if abs(r["life"] - BASELINE_LIFE) < 1e-9][0]
    ok = [r for r in rows
          if band(r["width_bp"], REAL_MEDIAN_BP, TOL_WIDTH) == "in-band"
          and band(r["levels"], REAL_LEVELS, TOL_LEVELS) == "in-band"
          and band(r["agg_per_s"], TARGET_RATE, TOL_RATE) == "in-band"]
    print("  Levels passing all three calibration targets: %s"
          % (", ".join("%.0fs" % r["life"] for r in ok) if ok else "none"))
    if ok:
        best = min(ok, key=lambda r: r["sd_stale"])
        print("  Of those, the lowest tracking error is at life=%.0fs: "
              "sd(V-mid) = %.2f against %.2f at the"
              % (best["life"], best["sd_stale"], base["sd_stale"]))
        print("  current 720s; a %.0f%% reduction; with MM PnL %+.2f "
              "against %+.2f."
              % (100.0 * (1 - best["sd_stale"] / base["sd_stale"]),
                 best["pnl"], base["pnl"]))
        if best["sd_stale"] < 0.9 * base["sd_stale"]:
            print("")
            print("  outcome (a): tracking error is a free parameter. It "
                  "moves materially while all three")
            print("  calibrated quantities stay inside their committed "
                  "bands, so the calibration never")
            print("  pinned it and it can be changed without breaking "
                  "847a7fa.")
        else:
            print("")
            print("  Outcome (b): coupled. Every setting that keeps the "
                  "calibration also keeps essentially")
            print("  the same tracking error, so the two cannot be separated "
                  "with this parameter.")
    else:
        print("")
        print("  Outcome (b), strongest form: no lifetime in this grid keeps "
              "all three targets in band,")
        print("  including the committed 720s. That would itself be a "
              "finding about the calibration.")
    print("")
    print("  Whether the maker becomes viable is a separate question from "
          "whether tracking error moves;")
    print("  section B answers the first and section A the second. A lever "
          "that improves tracking")
    print("  without making PnL positive is still only a partial fix.")
    print("")
    print("  No committed default changed. Nothing retuned. No sniffer.")


if __name__ == "__main__":
    main()
