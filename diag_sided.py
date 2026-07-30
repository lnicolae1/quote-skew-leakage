# diag_sided.py: one-sided book frequency vs side-independent price draw

import math
import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T / dt), seed=31415)


class HalfNormalNoise(NoiseTraderFlow):
    """Proposed C2b, tested here only: noise_traders.py is NOT modified"""

    def limit_price(self, engine, side):
        eta = abs(self.rng.gauss(0.0, self.disp))
        ref = self.reference()
        return self._round_to_tick(ref * math.exp(-eta if side == "buy"
                                                  else eta))


def run(cls, lam_mult=1, disp=0.0002, measure_marketable=False):
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000 - (i + 1) * 0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000 + (i + 1) * 0.5, 2), 0.02, "seed")

    noise = cls(lam=LAM_NOISE * lam_mult, p_market=0.5, value_fn=value_fn,
                disp=disp, seed=1001, reference_price=62000.0)
    informed = InformedTraderFlow(seed=2002)

    devs, two_sided, spreads, n_rest = [], 0, [], []
    n_limit = n_marketable = n_wrong_side = 0
    n_samples = 0
    t = 0.0
    while t < T:
        t += 1.0
        v = value_fn(t)
        for r in noise.run_until(eng, t):
            if r.order_type == "limit":
                n_limit += 1
                if r.fills:
                    n_marketable += 1
                if (r.side == "buy" and r.price > v) or \
                   (r.side == "sell" and r.price < v):
                    n_wrong_side += 1
        informed.run_until(eng, t, value_fn)
        bb, ba, m = eng.best_bid(), eng.best_ask(), eng.mid()
        n_samples += 1
        n_rest.append(len(eng.orders))
        if m is not None:
            devs.append(abs(m - v))
            two_sided += 1
            spreads.append(ba - bb)
    return {
        "mean_dev": statistics.mean(devs) if devs else float("nan"),
        "median_dev": statistics.median(devs) if devs else float("nan"),
        "pct_two_sided": 100.0 * two_sided / n_samples,
        "mean_resting": statistics.mean(n_rest),
        "median_spread": statistics.median(spreads) if spreads else float("nan"),
        "n_limit": n_limit, "n_marketable": n_marketable,
        "n_wrong_side": n_wrong_side,
    }


print("Test 1; how much 'liquidity supply' is actually liquidity demand?")
r = run(NoiseTraderFlow)
print("  current C2 (symmetric normal draw):")
print("    noise limit orders submitted        : %d" % r["n_limit"])
print("    ...priced on the wrong side of V    : %d (%.1f%%)"
      % (r["n_wrong_side"], 100.0 * r["n_wrong_side"] / r["n_limit"]))
print("    ...marketable on arrival (got fills): %d (%.1f%%)"
      % (r["n_marketable"], 100.0 * r["n_marketable"] / r["n_limit"]))
print("")

print("Test 2; proposed C2b (half-normal, side-dependent) vs current C2")
print("%-28s %11s %11s %12s %13s %12s"
      % ("variant", "mean|mid-V|", "med|mid-V|", "%two-sided",
         "mean resting", "med spread"))
for name, cls in (("C2  symmetric (current)", NoiseTraderFlow),
                  ("C2b half-normal (proposed)", HalfNormalNoise)):
    x = run(cls)
    print("%-28s %11.2f %11.2f %12.1f %13.1f %12.2f"
          % (name, x["mean_dev"], x["median_dev"], x["pct_two_sided"],
             x["mean_resting"], x["median_spread"]))

print("")
print("Test 3; C2b across arrival rates and disp")
print("%-9s %-8s %11s %11s %12s %13s %12s"
      % ("lam_mult", "disp", "mean|mid-V|", "med|mid-V|", "%two-sided",
         "mean resting", "med spread"))
for mult, disp in ((1, 0.0002), (5, 0.0002), (20, 0.0002), (5, 0.00005)):
    x = run(HalfNormalNoise, lam_mult=mult, disp=disp)
    print("%-9d %-8.5f %11.2f %11.2f %12.1f %13.1f %12.2f"
          % (mult, disp, x["mean_dev"], x["median_dev"], x["pct_two_sided"],
             x["mean_resting"], x["median_spread"]))
