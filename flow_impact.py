# flow_impact.py: locate the adverse-fill mechanism in the flow model (no sweep, nothing retuned)
# context (d526241): MM fills adverse by $14.61-59.24/BTC = 69-114% of maker loss; toxicity,
# quoted width, AS spread ruled out (009fc18)
# 1. impact by size, all trades: aggressor_sign * (mid[t-1+h] - mid[t-1]), h = 1, 10, 60s;
#    decile bins + log-log exponent (real venues ~0.5)
# 2. impact by class (noise-market, noise-limit-mkt, informed-take, informed-post) and by
#    whether the passive side was informed; noise-limit-mkt should be ~0 under JOIN clipping
# 3. touch replenishment: price / price+depth recovery, censored at 600s
# 4. counterfactual: mid-move sd in windows with vs without a trade
# control arm: BASE MM, no sniffer, 8 seeds x 7 days, clipped working point

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from damage_ab import T, SEEDS, K, REF_MID, C_TARGET, GAMMA, mean_se
from toxicity_sweep import make_world_tox
from informed_traders import TOXICITY_PRESETS
from clipped_mm_check import MM_ID

IMPACT_H = [1, 10, 60]
CF_H = [1, 10, 60]
RECOVER_CAP = 600
N_BINS = 8
RATIO = TOXICITY_PRESETS["medium"]
CLASSES = ("noise-market", "noise-limit-mkt", "informed-take",
           "informed-post")


def loglog_fit(xs, ys):
    """OLS of log(y) on log(x); returns (exponent, intercept, R2, n_used). Drops non-positive x or y."""
    px = [(math.log(a), math.log(b)) for a, b in zip(xs, ys)
          if a > 0 and b > 0]
    n = len(px)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    mx = sum(p[0] for p in px) / n
    my = sum(p[1] for p in px) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in px)
    sxx = sum((p[0] - mx) ** 2 for p in px)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan"), n
    b = sxy / sxx
    a = my - b * mx
    syy = sum((p[1] - my) ** 2 for p in px)
    pred = [a + b * p[0] for p in px]
    ssr = sum((p[1] - q) ** 2 for p, q in zip(px, pred))
    r2 = 1.0 - ssr / syy if syy > 0 else float("nan")
    return b, a, r2, n


def run(seed):
    vf, eng, noise, informed = make_world_tox(seed, RATIO)
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    n = int(T)
    mid = [0.0] * (n + 2)
    bb_s = [0.0] * (n + 2)
    ba_s = [0.0] * (n + 2)
    dbid = [0.0] * (n + 2)
    dask = [0.0] * (n + 2)
    n_trades_sec = [0] * (n + 2)

    # events: (t, cls, side, size, inf_passive_frac)
    events = []
    last = REF_MID
    lbb = lba = REF_MID

    t = 0.0
    while t < T:
        t += 1.0
        ti = int(t)
        for src, recs in (("noise", noise.run_until(eng, t)),
                          ("inf", informed.run_until(eng, t, vf))):
            for r in recs:
                fl = getattr(r, "fills", None)
                if not fl:
                    continue
                sz = sum(f.size for f in fl)
                if sz <= 0:
                    continue
                infv = sum(f.size for f in fl
                           if isinstance(f.counterparty_id, str)
                           and f.counterparty_id.startswith("informed"))
                if src == "noise":
                    cls = ("noise-market" if r.order_type == "market"
                           else "noise-limit-mkt")
                else:
                    cls = ("informed-take" if r.action == "take"
                           else "informed-post")
                events.append((ti, cls, r.side, sz, infv / sz))
                n_trades_sec[ti] += 1
                mm.on_fills(fl, r.side)
        mm.requote(eng, ti)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None:
            lbb = bb
        if ba is not None:
            lba = ba
        if bb is not None and ba is not None:
            last = 0.5 * (bb + ba)
        mid[ti] = last
        bb_s[ti] = lbb
        ba_s[ti] = lba
        dbid[ti] = eng.depth_at(lbb) if lbb else 0.0
        dask[ti] = eng.depth_at(lba) if lba else 0.0

    out = {"n_events": float(len(events))}

    # ---- 1 & 2: impact -------------------------------------------------- #
    sizes = sorted(e[3] for e in events)
    if len(sizes) < N_BINS * 2:
        return out
    cuts = [sizes[int(len(sizes) * (i + 1) / N_BINS) - 1]
            for i in range(N_BINS)]

    for h in IMPACT_H:
        bin_sz = [[] for _ in range(N_BINS)]
        bin_im = [[] for _ in range(N_BINS)]
        cls_im = {c: [0.0, 0.0] for c in CLASSES}   # [sum impact*sz, sum sz]
        cls_n = {c: 0 for c in CLASSES}
        inf_hi = [0.0, 0.0]
        inf_lo = [0.0, 0.0]
        for (ti, cls, side, sz, ifr) in events:
            j = ti - 1 + h
            if j > n or ti < 1:
                continue
            sgn = 1.0 if side == "buy" else -1.0
            im = sgn * (mid[j] - mid[ti - 1])
            b = 0
            while b < N_BINS - 1 and sz > cuts[b]:
                b += 1
            bin_sz[b].append(sz)
            bin_im[b].append(im)
            cls_im[cls][0] += im * sz
            cls_im[cls][1] += sz
            cls_n[cls] += 1
            if ifr > 0.5:
                inf_hi[0] += im * sz
                inf_hi[1] += sz
            else:
                inf_lo[0] += im * sz
                inf_lo[1] += sz
        bs, bi = [], []
        for b in range(N_BINS):
            if bin_sz[b]:
                bs.append(statistics.mean(bin_sz[b]))
                bi.append(statistics.mean(bin_im[b]))
                out["bin%d_sz_%d" % (b, h)] = bs[-1]
                out["bin%d_im_%d" % (b, h)] = bi[-1]
        ex, _a, r2, nu = loglog_fit(bs, bi)
        out["expo_%d" % h] = ex
        out["expo_r2_%d" % h] = r2
        out["expo_n_%d" % h] = float(nu)
        for c in CLASSES:
            out["cls_%s_%d" % (c, h)] = (cls_im[c][0] / cls_im[c][1]
                                         if cls_im[c][1] > 0 else float("nan"))
            out["clsn_%s" % c] = float(cls_n[c])
            out["clsv_%s" % c] = cls_im[c][1]
        out["infpass_hi_%d" % h] = (inf_hi[0] / inf_hi[1] if inf_hi[1] > 0
                                    else float("nan"))
        out["infpass_lo_%d" % h] = (inf_lo[0] / inf_lo[1] if inf_lo[1] > 0
                                    else float("nan"))
        out["infpass_hivol"] = inf_hi[1]
        out["infpass_lovol"] = inf_lo[1]

    # ---- 3: replenishment ----------------------------------------------- #
    prec, drec = [], []
    pcens = dcens = 0
    for (ti, cls, side, sz, ifr) in events:
        if ti < 1 or ti + RECOVER_CAP > n:
            continue
        if side == "buy":                 # consumed the ask side
            p0, d0 = ba_s[ti - 1], dask[ti - 1]
            if ba_s[ti] <= p0 and dask[ti] >= d0:
                continue                  # touch not consumed
            gotp = gotd = None
            for u in range(ti, ti + RECOVER_CAP + 1):
                if gotp is None and ba_s[u] <= p0:
                    gotp = u - ti
                if gotd is None and ba_s[u] <= p0 and dask[u] >= d0:
                    gotd = u - ti
                if gotp is not None and gotd is not None:
                    break
        else:                             # consumed the bid side
            p0, d0 = bb_s[ti - 1], dbid[ti - 1]
            if bb_s[ti] >= p0 and dbid[ti] >= d0:
                continue
            gotp = gotd = None
            for u in range(ti, ti + RECOVER_CAP + 1):
                if gotp is None and bb_s[u] >= p0:
                    gotp = u - ti
                if gotd is None and bb_s[u] >= p0 and dbid[u] >= d0:
                    gotd = u - ti
                if gotp is not None and gotd is not None:
                    break
        if gotp is None:
            pcens += 1
        else:
            prec.append(gotp)
        if gotd is None:
            dcens += 1
        else:
            drec.append(gotd)
    tot = len(prec) + pcens
    out["rec_price_med"] = statistics.median(prec) if prec else float("nan")
    out["rec_price_cens"] = (100.0 * pcens / tot) if tot else float("nan")
    totd = len(drec) + dcens
    out["rec_depth_med"] = statistics.median(drec) if drec else float("nan")
    out["rec_depth_cens"] = (100.0 * dcens / totd) if totd else float("nan")
    out["rec_n"] = float(tot)

    # ---- 4: counterfactual ---------------------------------------------- #
    cum = [0] * (n + 2)
    run_c = 0
    for i in range(1, n + 1):
        run_c += n_trades_sec[i]
        cum[i] = run_c
    for h in CF_H:
        with_d, without_d = [], []
        for i in range(1, n + 1 - h, h):
            ntr = cum[i + h] - cum[i]
            d = mid[i + h] - mid[i]
            (with_d if ntr > 0 else without_d).append(d)
        out["cf_with_sd_%d" % h] = (statistics.stdev(with_d)
                                    if len(with_d) > 1 else float("nan"))
        out["cf_without_sd_%d" % h] = (statistics.stdev(without_d)
                                       if len(without_d) > 1 else float("nan"))
        out["cf_with_n_%d" % h] = float(len(with_d))
        out["cf_without_n_%d" % h] = float(len(without_d))
    return out


def main():
    print("=" * 128)
    print("Flow model diagnosis")
    print("=" * 128)
    print("Control arm, BASE MM, no sniffer. %d seeds x %.0fs (%.0f days). "
          "k=%.5f, C=%.2f, gamma=%.4e, toxicity=%.2f."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA, RATIO))
    print("")

    t0 = time.time()
    per = None
    for s in SEEDS:
        x = run(s)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per.setdefault(k, []).append(v)
    agg = {}
    for k, v in per.items():
        m, se, _n = mean_se(v)
        agg[k], agg[k + "_se"] = m, se
    print("  %d runs in %.0fs.  %.0f trade events per run."
          % (len(SEEDS), time.time() - t0, agg["n_events"]))

    print("")
    print("=" * 128)
    print("1. price impact by trade size")
    print("=" * 128)
    for h in IMPACT_H:
        print("")
        print("  h = %ds" % h)
        print("    %10s %16s %16s %14s"
              % ("size decile", "mean size (BTC)", "mean impact ($)",
                 "$ per BTC"))
        for b in range(N_BINS):
            kz, ki = "bin%d_sz_%d" % (b, h), "bin%d_im_%d" % (b, h)
            if kz not in agg:
                continue
            sz, im = agg[kz], agg[ki]
            print("    %10d %16.5f %16.4f %14.1f"
                  % (b + 1, sz, im, im / sz if sz > 0 else float("nan")))
        print("    log-log exponent = %.4f +/- %.4f   R2 = %.4f   (%d bins "
              "used)"
              % (agg["expo_%d" % h], agg["expo_%d_se" % h],
                 agg["expo_r2_%d" % h], agg["expo_n_%d" % h]))
    print("")
    print("  real venues ~0.5; near 1.0 = linear impact")

    print("")
    print("=" * 128)
    print("2. impact by source")
    print("=" * 128)
    print("  %-18s %10s %14s %14s %14s %14s"
          % ("class", "events", "volume BTC", "$/BTC h=1", "$/BTC h=10",
             "$/BTC h=60"))
    for c in CLASSES:
        print("  %-18s %10.0f %14.4f %14.1f %14.1f %14.1f"
              % (c, agg.get("clsn_%s" % c, float("nan")),
                 agg.get("clsv_%s" % c, float("nan")),
                 agg.get("cls_%s_1" % c, float("nan")),
                 agg.get("cls_%s_10" % c, float("nan")),
                 agg.get("cls_%s_60" % c, float("nan"))))
    print("")
    print("  noise-limit-mkt expected ~0 under JOIN clipping")
    print("")
    print("  by passive counterparty:")
    print("  %-34s %14s %14s %14s %14s"
          % ("passive counterparty", "volume BTC", "$/BTC h=1", "$/BTC h=10",
             "$/BTC h=60"))
    print("  %-34s %14.4f %14.1f %14.1f %14.1f"
          % ("mostly INFORMED resting", agg.get("infpass_hivol", float("nan")),
             agg.get("infpass_hi_1", float("nan")),
             agg.get("infpass_hi_10", float("nan")),
             agg.get("infpass_hi_60", float("nan"))))
    print("  %-34s %14.4f %14.1f %14.1f %14.1f"
          % ("mostly NOISE/MM resting", agg.get("infpass_lovol", float("nan")),
             agg.get("infpass_lo_1", float("nan")),
             agg.get("infpass_lo_10", float("nan")),
             agg.get("infpass_lo_60", float("nan"))))

    print("")
    print("=" * 128)
    print("3. touch replenishment")
    print("=" * 128)
    print("  %-30s %16s %16s"
          % ("measure", "median (s)", "never within %ds" % RECOVER_CAP))
    print("  %-30s %16.1f %15.1f%%"
          % ("best price returns", agg["rec_price_med"],
             agg["rec_price_cens"]))
    print("  %-30s %16.1f %15.1f%%"
          % ("price AND depth return", agg["rec_depth_med"],
             agg["rec_depth_cens"]))
    print("  (%.0f consuming trades per run)" % agg["rec_n"])
    print("")
    print("  median over completed recoveries; one trade every ~%.0fs"
          % (T / max(1.0, agg["n_events"])))

    print("")
    print("=" * 128)
    print("4. counterfactual: mid-move sd with vs without a trade")
    print("=" * 128)
    print("  %8s %16s %12s %16s %12s %12s"
          % ("H (s)", "sd with trade", "windows", "sd no trade", "windows",
             "ratio"))
    for h in CF_H:
        w = agg["cf_with_sd_%d" % h]
        wo = agg["cf_without_sd_%d" % h]
        print("  %8d %16.4f %12.0f %16.4f %12.0f %12.2f"
              % (h, w, agg["cf_with_n_%d" % h], wo,
                 agg["cf_without_n_%d" % h],
                 (w / wo) if wo and wo > 0 else float("nan")))
    print("")
    print("  non-overlapping windows")


if __name__ == "__main__":
    main()
