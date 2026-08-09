# staleness.py: is the maker the only participant not tracking V?
# - noise and informed orders are priced off V; the maker quotes off the observed mid
# - at each MM fill: s = V(t) - mid(t-1); pos_sign +1 bought, -1 sold; d = -pos_sign * s
#   (d > 0: maker took the side V says is wrong)
# - prediction: (1) adverse move rises monotonically with d; (2) adverse move ~0 where d ~ 0
#   (refuted if the bleed is as large there); (3) selection: fills over-represented in the
#   tails of |V - mid| relative to all seconds
# - cross-check vs 009fc18 (PnL -608.71 at toxicity 0.001, -466.49 at 0.30):
#   predicts sd(V - mid) rises as toxicity falls
# - control arm, no sniffer; 8 seeds x 7 days, C = 13.66, gamma from C

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

H_SHORT = 60
H_HOLD = 10800
N_BINS = 8
RATIO = TOXICITY_PRESETS["medium"]
TOX_LEVELS = [0.001, TOXICITY_PRESETS["medium"], 0.30]
# control-arm PnL at these levels (009fc18)
TOX_PNL = {0.001: -608.71, 0.10: -525.18, 0.30: -466.49}


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    i = int(p * (len(sorted_vals) - 1))
    return sorted_vals[i]


def run(seed, ratio):
    vf, eng, noise, informed = make_world_tox(seed, ratio)
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    n = int(T)
    mid = [0.0] * (n + 2)
    last = REF_MID
    fills = []                    # (t, pos_sign, size, m0, stale)
    all_stale = []

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
                for f in fl:
                    if f.counterparty_id != MM_ID:
                        continue
                    pos_sign = -1.0 if r.side == "buy" else 1.0
                    fills.append((ti, pos_sign, f.size, last, v - last))
                mm.on_fills(fl, r.side)
        mm.requote(eng, ti)
        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            last = 0.5 * (bb + ba)
        mid[ti] = last
        all_stale.append(v - last)

    out = {"n_fills": float(len(fills))}
    out["stale_sd"] = (statistics.stdev(all_stale) if len(all_stale) > 1
                       else float("nan"))
    out["stale_absmean"] = statistics.mean([abs(x) for x in all_stale])

    # distributions: all seconds vs fill moments
    a_abs = sorted(abs(x) for x in all_stale)
    f_abs = sorted(abs(e[4]) for e in fills)
    for p, tag in ((0.50, "p50"), (0.90, "p90"), (0.99, "p99")):
        out["all_%s" % tag] = pct(a_abs, p)
        out["fill_%s" % tag] = pct(f_abs, p)
    # share of fills above the all-seconds 80th/95th percentile
    for p, tag in ((0.80, "top20"), (0.95, "top5")):
        thr = pct(a_abs, p)
        out["fillshare_%s" % tag] = (100.0 * sum(1 for x in f_abs if x > thr)
                                     / len(f_abs)) if f_abs else float("nan")

    # directional staleness bins vs adverse move
    rows = []
    for (ti, ps, sz, m0, st) in fills:
        d = -ps * st
        a60 = a_hold = None
        j = ti - 1 + H_SHORT
        if j <= n:
            a60 = -ps * (mid[j] - m0)
        j = ti - 1 + H_HOLD
        if j <= n:
            a_hold = -ps * (mid[j] - m0)
        rows.append((d, sz, a60, a_hold))
    rows.sort(key=lambda r: r[0])
    m = len(rows)
    for b in range(N_BINS):
        lo, hi = int(m * b / N_BINS), int(m * (b + 1) / N_BINS)
        chunk = rows[lo:hi]
        if not chunk:
            continue
        out["bin%d_d" % b] = statistics.mean([c[0] for c in chunk])
        for tag, idx in (("a60", 2), ("ah", 3)):
            num = den = 0.0
            for c in chunk:
                if c[idx] is None:
                    continue
                num += c[idx] * c[1]
                den += c[1]
            out["bin%d_%s" % (b, tag)] = (num / den) if den > 0 else \
                float("nan")

    # V and mid agree: |staleness| in the bottom 20%
    thr = pct(sorted(abs(e[4]) for e in fills), 0.20)
    num = den = 0.0
    for (ti, ps, sz, m0, st) in fills:
        if abs(st) > thr:
            continue
        j = ti - 1 + H_SHORT
        if j > n:
            continue
        num += (-ps * (mid[j] - m0)) * sz
        den += sz
    out["agree_a60"] = (num / den) if den > 0 else float("nan")
    out["agree_thr"] = thr
    out["frac_wrong_side"] = 100.0 * sum(1 for r in rows if r[0] > 0) / m
    return out


def measure(ratio):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, ratio)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per.setdefault(k, []).append(v)
    out = {"ratio": ratio, "secs": time.time() - t0}
    for k, v in per.items():
        mm_, se, _n = mean_se(v)
        out[k], out[k + "_se"] = mm_, se
    return out


def main():
    print("=" * 126)
    print("staleness: is the maker the only participant not tracking V?")
    print("=" * 126)
    print("control arm, no sniffer. %d seeds x %.0fs (%.0f days). k=%.5f, "
          "C=%.2f, gamma=%.4e."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA))
    print("s = V(t) - mid(t-1); d = -(position sign) * s; d > 0: wrong side of V")
    print("prediction: adverse move rises with d, ~0 where d ~ 0.")
    print("")

    main_r = measure(RATIO)
    print("  toxicity %.2f measured in %.0fs -- %.0f MM fills per run"
          % (RATIO, main_r["secs"], main_r["n_fills"]), flush=True)
    print("")

    print("=" * 126)
    print("1. adverse move by directional staleness")
    print("=" * 126)
    print("  %8s %18s %20s %20s"
          % ("octile", "mean d ($)", "adverse h=60 ($/BTC)",
             "adverse h=10800 ($/BTC)"))
    for b in range(N_BINS):
        k = "bin%d_d" % b
        if k not in main_r:
            continue
        print("  %8d %18.4f %20.4f %20.4f"
              % (b + 1, main_r[k], main_r["bin%d_a60" % b],
                 main_r["bin%d_ah" % b]))
    print("")
    print("  fills where V and mid agree (|staleness| in bottom 20%%, "
          "below $%.4f):"
          % main_r["agree_thr"])
    print("    adverse move h=60 = %+.4f +/- %.4f $/BTC"
          % (main_r["agree_a60"], main_r["agree_a60_se"]))
    print("    ~0 confirms; comparable to overall $24.81/BTC refutes")
    print("")
    print("  fills on the wrong side of V (d > 0): %.2f%% +/- %.2f%%"
          % (main_r["frac_wrong_side"], main_r["frac_wrong_side_se"]))
    print("  50% = no directional selection")

    print("")
    print("=" * 126)
    print("2. exposure or selection: |V - mid| over all seconds vs at fills")
    print("=" * 126)
    print("  %-22s %14s %14s %14s"
          % ("distribution of |V-mid|", "p50 ($)", "p90 ($)", "p99 ($)"))
    print("  %-22s %14.4f %14.4f %14.4f"
          % ("all seconds", main_r["all_p50"], main_r["all_p90"],
             main_r["all_p99"]))
    print("  %-22s %14.4f %14.4f %14.4f"
          % ("at MM fills", main_r["fill_p50"], main_r["fill_p90"],
             main_r["fill_p99"]))
    print("")
    print("  share of MM fills above the all-seconds percentile:")
    print("    above the 80th: %6.2f%% +/- %.2f%%   (20%% = no selection)"
          % (main_r["fillshare_top20"], main_r["fillshare_top20_se"]))
    print("    above the 95th: %6.2f%% +/- %.2f%%   ( 5%% = no selection)"
          % (main_r["fillshare_top5"], main_r["fillshare_top5_se"]))
    print("")
    print("  above baseline: selected into stale moments, not merely exposed")

    print("")
    print("=" * 126)
    print("3. cross-check vs 009fc18 toxicity result")
    print("=" * 126)
    print("  009fc18: less informed flow, larger loss (-608.71 at toxicity "
          "0.001, -466.49 at 0.30)")
    print("  prediction: informed traders post at V, so sd(V - mid) rises as "
          "toxicity falls")
    print("")
    rows = [main_r]
    for tx in TOX_LEVELS:
        if abs(tx - RATIO) < 1e-12:
            continue
        r = measure(tx)
        rows.append(r)
        print("    toxicity %.3f measured in %.0fs" % (tx, r["secs"]),
              flush=True)
    rows.sort(key=lambda r: r["ratio"])
    print("")
    print("  %10s %20s %20s %18s"
          % ("toxicity", "sd(V - mid) ($)", "mean |V - mid| ($)",
             "PnL (009fc18)"))
    for r in rows:
        pnl = TOX_PNL.get(round(r["ratio"], 3), float("nan"))
        print("  %10.3f %10.4f +/- %-7.4f %10.4f +/- %-7.4f %18.2f"
              % (r["ratio"], r["stale_sd"], r["stale_sd_se"],
                 r["stale_absmean"], r["stale_absmean_se"], pnl))
    print("")
    lo, hi = rows[0], rows[-1]
    if lo["stale_sd"] > hi["stale_sd"]:
        print("  sd(V-mid) falls as toxicity rises (%.4f at %.3f -> %.4f at "
              "%.3f): predicted direction"
              % (lo["stale_sd"], lo["ratio"], hi["stale_sd"], hi["ratio"]))
        print("  consistent with informed posting pulling the book to V "
              "(009fc18 PnL ordering)")
    else:
        print("  sd(V-mid) does not fall as toxicity rises (%.4f at %.3f -> "
              "%.4f at %.3f)"
              % (lo["stale_sd"], lo["ratio"], hi["stale_sd"], hi["ratio"]))
        print("  does not explain 009fc18; take_fraction attribution stays "
              "unexplained")
    print("")
    print("  no defaults changed; nothing retuned; no sniffer")


if __name__ == "__main__":
    main()
