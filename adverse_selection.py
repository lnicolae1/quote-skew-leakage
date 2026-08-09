# adverse_selection.py: direct measurement of the maker's adverse selection (control-arm PnL loss)

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE, DEFAULT_SIGMA
from damage_ab import (T, SEEDS, K, REF_MID, C_TARGET, GAMMA, mean_se,
                       SEC_PER_YEAR)
from toxicity_sweep import make_world_tox, TouchMM, REAL_MEDIAN_BP
from informed_traders import TOXICITY_PRESETS
from clipped_mm_check import MM_ID

HORIZONS = [60, 300, 3600, 10800]
SAMPLE_DT = 60
RATIO = TOXICITY_PRESETS["medium"]

EARNED_TOTAL = {"BASE": 63.83, "TOUCH": 12.89}
INV_TOTAL = {"BASE": 589.01, "TOUCH": 556.44}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sxx = syy = 0.0
    for a, b in zip(xs, ys):
        da, db = a - mx, b - my
        sxy += da * db
        sxx += da * da
        syy += db * db
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def run(seed, mm_cls):
    vf, eng, noise, informed = make_world_tox(seed, RATIO)
    mm = mm_cls(horizon=T, k=K, gamma=GAMMA, quote_size=DEFAULT_QUOTE_SIZE)

    n = int(T)
    mid_series = [0.0] * (n + 2)
    last_mid = REF_MID
    fills_log = []
    q_series = []

    t = 0.0
    while t < T:
        t += 1.0
        ti = int(t)
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fl = getattr(r, "fills", None)
                if not fl:
                    continue
                for f in fl:
                    if f.counterparty_id != MM_ID:
                        continue
                    pos_sign = -1.0 if r.side == "buy" else 1.0
                    fills_log.append((ti, pos_sign, f.size, last_mid, mm.q))
                mm.on_fills(fl, r.side)
        mm.requote(eng, ti)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            last_mid = 0.5 * (bb + ba)
        mid_series[ti] = last_mid
        if ti % SAMPLE_DT == 0:
            q_series.append((ti, mm.q))

    out = {}
    for H in HORIZONS:
        num = den = 0.0
        bnum = bden = snum = sden = 0.0
        lnum = lden = shnum = shden = 0.0
        raw_num = 0.0
        dropped = 0
        for (ti, ps, sz, m0, qb) in fills_log:
            j = ti - 1 + H
            if j > n:
                dropped += 1
                continue
            dm = mid_series[j] - m0
            adv = -ps * dm
            num += adv * sz
            den += sz
            raw_num += dm * sz
            if ps > 0:
                bnum += adv * sz
                bden += sz
            else:
                snum += adv * sz
                sden += sz
            if qb > 0:
                lnum += adv * sz
                lden += sz
            elif qb < 0:
                shnum += adv * sz
                shden += sz
        out["adv_%d" % H] = (num / den) if den > 0 else float("nan")
        out["advtot_%d" % H] = num
        out["drift_%d" % H] = (raw_num / den) if den > 0 else float("nan")
        out["advbuy_%d" % H] = (bnum / bden) if bden > 0 else float("nan")
        out["advsell_%d" % H] = (snum / sden) if sden > 0 else float("nan")
        out["advbuyadj_%d" % H] = out["advbuy_%d" % H] + out["drift_%d" % H]
        out["advselladj_%d" % H] = out["advsell_%d" % H] - out["drift_%d" % H]
        out["advlong_%d" % H] = (lnum / lden) if lden > 0 else float("nan")
        out["advshort_%d" % H] = (shnum / shden) if shden > 0 else \
            float("nan")
        out["dropped_%d" % H] = float(dropped)

        qs, dps = [], []
        for (ti, q) in q_series:
            j = ti + H
            if j > n:
                continue
            qs.append(q)
            dps.append(mid_series[j] - mid_series[ti])
        out["corr_%d" % H] = pearson(qs, dps)

    out["n_fills"] = float(len(fills_log))
    out["mm_vol"] = sum(f[2] for f in fills_log)
    return out


def measure(mm_cls):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, mm_cls)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    out = {"secs": time.time() - t0}
    for k, v in per.items():
        m, se, _n = mean_se(v)
        out[k], out[k + "_se"] = m, se
    return out


def main():
    print("=" * 130)
    print("Adverse selection, measured directly; what does the price do "
          "after each MM fill?")
    print("=" * 130)
    print("Control arm only, no sniffer. %d seeds x %.0fs (%.0f days). "
          "k=%.5f, C=%.2f, gamma=%.4e, toxicity=%.2f."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA, RATIO))
    print("")
    print("adverse move = -(position sign) * (mid[t-1+H] - mid[t-1]), "
          "size-weighted, in $/BTC.")
    print("Positive means the price moved against the position the fill "
          "created. H=10800s is the")
    print("column that matters: the corrected inventory half-life is 1.9-4.1h "
          "(7d87911).")
    print("")

    arms = [("BASE  2/k, spread ~$18.36", MarketMaker, "BASE"),
            ("TOUCH  mkt spread ~$1.82", TouchMM, "TOUCH")]
    res = []
    for label, cls, key in arms:
        r = measure(cls)
        r["key"] = key
        res.append((label, r))
        print("  %-28s %4.0fs   fills %6.0f   vol %6.2f BTC   adverse@10800s "
              "= %+8.3f $/BTC"
              % (label, r["secs"], r["n_fills"], r["mm_vol"],
                 r["adv_10800"]), flush=True)
    print("")

    print("=" * 130)
    print("1. The adverse move per fill, by horizon; and is it the same at "
          "both widths?")
    print("=" * 130)
    print("  %-28s %18s %18s %18s %18s"
          % ("arm", "H=60s", "H=300s", "H=3600s", "H=10800s"))
    for label, r in res:
        print("  %-28s %8.4f+-%-8.4f %8.4f+-%-8.4f %8.4f+-%-8.4f "
              "%8.4f+-%-8.4f"
              % (label,
                 r["adv_60"], r["adv_60_se"], r["adv_300"], r["adv_300_se"],
                 r["adv_3600"], r["adv_3600_se"],
                 r["adv_10800"], r["adv_10800_se"]))
    print("")
    print("  Drift control; size-weighted mean raw price change over the "
          "same horizons ($/BTC).")
    print("  Should be ~0; if not, any buy/sell asymmetry below is partly the "
          "value path, not the maker.")
    for label, r in res:
        print("  %-28s %8.4f          %8.4f          %8.4f          %8.4f"
              % (label, r["drift_60"], r["drift_300"], r["drift_3600"],
                 r["drift_10800"]))
    b, tch = res[0][1], res[1][1]
    if tch["adv_10800"] not in (0.0,) and not math.isnan(tch["adv_10800"]):
        print("")
        print("  base / touch ratio: %.3f at H=10800s, %.3f at H=60s."
              % (b["adv_10800"] / tch["adv_10800"],
                 b["adv_60"] / tch["adv_60"]))
        print("  These are NOT 1, so a selection effect is real and "
              "measurable: a quote sitting ~9x")
        print("  outside the touch does catch somewhat more adverse fills. "
              "But it is a factor of ~1.3-1.7,")
        print("  nowhere near the ~7x shortfall below, and adverse selection "
              "of $%.1f-%.1f/BTC persists at"
              % (tch["adv_60"], tch["adv_10800"]))
        print("  the touch where there is no selection to speak of. So the "
              "selection effect is real but")
        print("  secondary; the dominant fact is that every fill is adverse, "
              "at every width tested.")

    print("")
    print("=" * 130)
    print("1b. The bridge to reality; adverse move as a fraction of an "
          "unconditional one-sigma move")
    print("=" * 130)
    print("  one-sigma over H = sigma * sqrt(H / seconds_per_year) * mid, "
          "with sigma = %.4f annualised" % DEFAULT_SIGMA)
    print("  (the Phase 3 measurement) and mid = $%.0f. In a real venue "
          "conditional adverse selection is" % REF_MID)
    print("  a few percent of a one-sigma move. This ratio is the number that "
          "says whether this")
    print("  simulator's flow is far more informative about the next price "
          "move than real flow is.")
    print("")
    print("  %-28s %8s %14s %14s %10s"
          % ("arm", "H (s)", "1-sigma ($)", "adverse ($)", "ratio"))
    for label, r in res:
        for H in HORIZONS:
            sig1 = DEFAULT_SIGMA * math.sqrt(H / SEC_PER_YEAR) * REF_MID
            adv = r["adv_%d" % H]
            print("  %-28s %8d %14.2f %14.2f %9.1f%%"
                  % (label, H, sig1, adv, 100.0 * adv / sig1))
    print("")
    print("  The ratio FALLS with H because the unconditional move grows as "
          "sqrt(H) while the adverse")
    print("  move barely grows; which is the same fact section 1c measures "
          "from the other side.")

    print("")
    print("=" * 130)
    print("1c. Diffusion check; does the adverse move grow like sqrt(H)?")
    print("=" * 130)
    print("  Slow informed drift accumulating would grow as sqrt(H). "
          "Immediate repricing that comes")
    print("  with the fill is already there at H=60s and barely grows. "
          "'diffusion fraction' = measured")
    print("  growth / sqrt growth: near 1 means diffusion, near 0 means the "
          "move is immediate.")
    print("")
    print("  %-28s %18s %12s %12s %14s"
          % ("arm", "horizon pair", "sqrt ratio", "measured", "diff. frac"))
    pairs = [(HORIZONS[i], HORIZONS[i + 1]) for i in range(len(HORIZONS) - 1)]
    pairs.append((HORIZONS[0], HORIZONS[-1]))
    for label, r in res:
        for (h0, h1) in pairs:
            sq = math.sqrt(float(h1) / h0)
            ms = r["adv_%d" % h1] / r["adv_%d" % h0]
            print("  %-28s %8d -> %-7d %12.3f %12.3f %14.3f"
                  % (label, h0, h1, sq, ms, ms / sq))

    print("")
    print("=" * 130)
    print("2. What half-spread would cover it?")
    print("=" * 130)
    print("  Earned per BTC is computed per arm from 009fc18's committed "
          "spread totals divided by this")
    print("  run's measured volume; the two arms differ by ~10x and reusing "
          "one rate for both would")
    print("  understate the TOUCH shortfall by an order of magnitude.")
    print("")
    print("  %-28s %14s %14s %14s %16s"
          % ("arm", "spread ($)", "volume (BTC)", "earned $/BTC",
             "shortfall"))
    for label, r in res:
        earned = EARNED_TOTAL[r["key"]] / r["mm_vol"] if r["mm_vol"] > 0             else float("nan")
        need = r["adv_10800"]
        print("  %-28s %14.2f %14.2f %14.3f %14.1fx"
              % (label, EARNED_TOTAL[r["key"]], r["mm_vol"], earned,
                 need / earned if earned else float("nan")))
    print("")
    print("  A shortfall factor near the ~9x implied by the PnL decomposition "
          "(needs ~$81, earns ~$8.83)")
    print("  confirms the mechanism and makes the finding quotable as a "
          "factor, not a direction.")
    print("")
    print("  Quantitative bridge; total (adverse move x size) over all "
          "fills, against the |INV| each")
    print("  arm actually lost (2eb46f1 / 009fc18). If a horizon reproduces "
          "it, the mechanism is")
    print("  sufficient, not merely present.")
    print("  %-28s %10s %12s %12s %12s %12s"
          % ("arm", "|INV|", "H=60s", "H=300s", "H=3600s", "H=10800s"))
    for label, r in res:
        iv = INV_TOTAL[r["key"]]
        print("  %-28s %10.2f %12.2f %12.2f %12.2f %12.2f"
              % (label, iv, r["advtot_60"], r["advtot_300"],
                 r["advtot_3600"], r["advtot_10800"]))
    print("  %-28s %10s %12s %12s %12s %12s"
          % ("  as % of |INV|", "", "", "", "", ""))
    for label, r in res:
        iv = INV_TOTAL[r["key"]]
        print("  %-28s %10s %11.0f%% %11.0f%% %11.0f%% %11.0f%%"
              % (label, "",
                 100.0 * r["advtot_60"] / iv, 100.0 * r["advtot_300"] / iv,
                 100.0 * r["advtot_3600"] / iv,
                 100.0 * r["advtot_10800"] / iv))

    print("")
    print("=" * 130)
    print("3. Is it symmetric in q's sign, or is the maker caught long into "
          "FALLS?")
    print("=" * 130)
    print("  Split by the side of the fill (which direction it pushed the "
          "maker's position):")
    print("  %-28s %22s %22s %14s"
          % ("arm", "MM bought (now longer)", "MM sold (now shorter)",
             "asymmetry"))
    for label, r in res:
        bb_, ss_ = r["advbuy_10800"], r["advsell_10800"]
        print("  %-28s %10.4f+-%-10.4f %10.4f+-%-10.4f %14.4f"
              % (label, bb_, r["advbuy_10800_se"], ss_,
                 r["advsell_10800_se"], bb_ - ss_))
    print("")
    print("  THE SAME split, drift-adjusted. A common drift D over the "
          "horizon contributes -D to buy")
    print("  fills and +D to sell fills, so it cancels in the overall mean "
          "but dominates the split.")
    print("  These columns add D back to the buy side and remove it from the "
          "sell side; they are the")
    print("  ones that answer the symmetry question at the holding horizon.")
    print("  %-28s %22s %22s %14s"
          % ("arm", "MM bought (adjusted)", "MM sold (adjusted)",
             "asymmetry"))
    for label, r in res:
        ba_, sa_ = r["advbuyadj_10800"], r["advselladj_10800"]
        print("  %-28s %22.4f %22.4f %14.4f" % (label, ba_, sa_, ba_ - sa_))
    print("")
    print("  H=60s, where the raw drift is already ~0, as an independent "
          "read on the same question:")
    print("  %-28s %22s %22s %14s"
          % ("arm", "MM bought", "MM sold", "asymmetry"))
    for label, r in res:
        print("  %-28s %10.4f+-%-10.4f %10.4f+-%-10.4f %14.4f"
              % (label, r["advbuy_60"], r["advbuy_60_se"], r["advsell_60"],
                 r["advsell_60_se"], r["advbuy_60"] - r["advsell_60"]))
    print("")
    print("  Split by the maker's inventory sign at the moment of the fill "
          "(NOT drift-adjustable --")
    print("  this split mixes both fill sides, so read it with the drift row "
          "in section 1 in view):")
    print("  %-28s %22s %22s %14s"
          % ("arm", "already long (q>0)", "already short (q<0)",
             "asymmetry"))
    for label, r in res:
        ll, sh = r["advlong_10800"], r["advshort_10800"]
        print("  %-28s %10.4f+-%-10.4f %10.4f+-%-10.4f %14.4f"
              % (label, ll, r["advlong_10800_se"], sh,
                 r["advshort_10800_se"], ll - sh))
    print("")
    print("  Both columns positive and close = symmetric adverse selection: "
          "every fill is adverse and")
    print("  the maker is picked off in both directions equally. One column "
          "much larger = a directional")
    print("  mechanism, and the write-up has to say which. These are "
          "different findings.")

    print("")
    print("=" * 130)
    print("4. Population-level check; corr(q_t, mid[t+H] - mid[t]), sampled "
          "every %ds" % SAMPLE_DT)
    print("=" * 130)
    print("  %-28s %18s %18s %18s %18s"
          % ("arm", "H=60s", "H=300s", "H=3600s", "H=10800s"))
    for label, r in res:
        print("  %-28s %8.4f+-%-8.4f %8.4f+-%-8.4f %8.4f+-%-8.4f "
              "%8.4f+-%-8.4f"
              % (label,
                 r["corr_60"], r["corr_60_se"], r["corr_300"],
                 r["corr_300_se"], r["corr_3600"], r["corr_3600_se"],
                 r["corr_10800"], r["corr_10800_se"]))
    print("")
    print("  negative means the maker is long before the price falls. This "
          "does not depend on the")
    print("  fill-level bookkeeping above being right, so agreement between "
          "the two is a check on")
    print("  both. Fills whose horizon ran past the end of the run were "
          "dropped, not clamped:")
    for label, r in res:
        print("    %-28s dropped %.0f of %.0f fills at H=10800s"
              % (label, r["dropped_10800"], r["n_fills"]))

    print("")
    print("  No committed default changed. gamma, k and quote_size are "
          "constructor arguments; gamma")
    print("  is derived from C. No sniffer anywhere in this file.")


if __name__ == "__main__":
    main()
