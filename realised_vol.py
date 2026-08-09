# realised_vol.py: in-sample unconditional price move, measured rather than computed from sigma

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
from toxicity_sweep import make_world_tox
from informed_traders import TOXICITY_PRESETS
from clipped_mm_check import MM_ID

HORIZONS = [60, 300, 3600, 10800]
RATIO = TOXICITY_PRESETS["medium"]

ADVERSE = {
    "BASE": {60: 24.8100, 300: 28.6379, 3600: 56.5071, 10800: 59.2398},
    "TOUCH": {60: 14.6080, 300: 16.7456, 3600: 38.2780, 10800: 45.2783},
}
OLD_RATIO = {
    "BASE": {60: 78.0, 300: 40.3, 3600: 22.9, 10800: 13.9},
    "TOUCH": {60: 45.9, 300: 23.5, 3600: 15.5, 10800: 10.6},
}


def run(seed):
    vf, eng, noise, informed = make_world_tox(seed, RATIO)
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    n = int(T)
    mid = [0.0] * (n + 2)
    last = REF_MID

    t = 0.0
    while t < T:
        t += 1.0
        ti = int(t)
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fl = getattr(r, "fills", None)
                if fl:
                    mm.on_fills(fl, r.side)
        mm.requote(eng, ti)
        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            last = 0.5 * (bb + ba)
        mid[ti] = last

    out = {"mean_mid": statistics.mean(mid[1:n + 1])}
    for H in HORIZONS:
        d = [mid[i + H] - mid[i] for i in range(1, n + 1 - H, H)]
        out["sd_%d" % H] = statistics.stdev(d) if len(d) > 1 else float("nan")
        out["nwin_%d" % H] = float(len(d))
    return out


def main():
    print("=" * 122)
    print("Realised mid volatility, measured in-sample; does the simulator "
          "reproduce the sigma it was given?")
    print("=" * 122)
    print("Control arm, BASE MM, no sniffer. %d seeds x %.0fs (%.0f days). "
          "k=%.5f, C=%.2f, gamma=%.4e, toxicity=%.2f."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA, RATIO))
    print("sigma input = %.4f annualised (sim_run.SIGMA = "
          "market_maker.DEFAULT_SIGMA, both feed the value path)."
          % DEFAULT_SIGMA)
    print("sd taken over non-overlapping windows of each horizon.")
    print("")

    t0 = time.time()
    per = None
    for s in SEEDS:
        x = run(s)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    agg = {}
    for k, v in per.items():
        m, se, _n = mean_se(v)
        agg[k], agg[k + "_se"] = m, se
    print("  %d runs in %.0fs.  realised mean mid = $%.2f (reference "
          "%0.0f)" % (len(SEEDS), time.time() - t0, agg["mean_mid"], REF_MID))
    print("")

    print("=" * 122)
    print("A. Realised vs sigma-implied")
    print("=" * 122)
    print("  %8s %9s %18s %16s %16s %10s %14s"
          % ("H (s)", "windows", "realised sd ($)", "implied @62000",
             "implied @mean", "ratio", "implied annual"))
    real = {}
    for H in HORIZONS:
        sd = agg["sd_%d" % H]
        real[H] = sd
        imp_ref = DEFAULT_SIGMA * math.sqrt(H / SEC_PER_YEAR) * REF_MID
        imp_mean = DEFAULT_SIGMA * math.sqrt(H / SEC_PER_YEAR) * agg["mean_mid"]
        ann = (sd / agg["mean_mid"]) * math.sqrt(SEC_PER_YEAR / H)
        print("  %8d %9.0f %8.2f+-%-8.2f %16.2f %16.2f %10.3f %13.4f"
              % (H, agg["nwin_%d" % H], sd, agg["sd_%d_se" % H], imp_ref,
                 imp_mean, sd / imp_ref, ann))
    print("")
    print("  implied annual = realised sd / mean mid * sqrt(year / H). "
          "Compare against the %.4f input."
          % DEFAULT_SIGMA)
    print("  A ratio near 1 means the mid inherits the value path's "
          "volatility; below 1 means the book")
    print("  is smoother than V (staleness), above 1 means microstructure "
          "noise adds to it.")

    print("")
    print("=" * 122)
    print("B. d526241 Section 1b, recomputed against the realised "
          "denominator")
    print("=" * 122)
    print("  Adverse moves are the committed figures from d526241 section 1; "
          "only the denominator changes.")
    print("")
    print("  %-8s %8s %14s %16s %14s %16s %12s"
          % ("arm", "H (s)", "adverse ($)", "old denom ($)", "old ratio",
             "realised denom", "new ratio"))
    for arm in ("BASE", "TOUCH"):
        for H in HORIZONS:
            adv = ADVERSE[arm][H]
            old_d = DEFAULT_SIGMA * math.sqrt(H / SEC_PER_YEAR) * REF_MID
            new_d = real[H]
            print("  %-8s %8d %14.2f %16.2f %13.1f%% %16.2f %11.1f%%"
                  % (arm, H, adv, old_d, 100.0 * adv / old_d, new_d,
                     100.0 * adv / new_d))
    print("")
    print("  If the new ratios differ materially from the old, the new ones "
          "are the headline and")
    print("  d526241's section 1b is superseded by this file. If they agree, "
          "78%/46% stands as")
    print("  measured rather than as an artefact of using the input sigma.")
    print("")
    print("  Either way the comparison to reality is unchanged in kind: in a "
          "real venue conditional")
    print("  adverse selection is a few percent of an unconditional move over "
          "the same horizon.")
    print("")
    print("  No committed default changed. No sniffer. gamma derived from C.")


if __name__ == "__main__":
    main()
