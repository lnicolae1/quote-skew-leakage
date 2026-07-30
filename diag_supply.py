# diag_supply.py: liquidity supply vs demand at p_market = 0.5

import math
import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T / dt), seed=31415)


class HalfNormalNoise(NoiseTraderFlow):
    """C2b variant, tested here only; noise_traders.py is NOT modified"""
    def limit_price(self, engine, side):
        eta = abs(self.rng.gauss(0.0, self.disp))
        ref = self.reference()
        return self._round_to_tick(ref * math.exp(-eta if side == "buy"
                                                  else eta))


def run(cls, lam_mult, p_market, disp=0.0002):
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000 - (i + 1) * 0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000 + (i + 1) * 0.5, 2), 0.02, "seed")

    noise = cls(lam=LAM_NOISE * lam_mult, p_market=p_market, value_fn=value_fn,
                disp=disp, seed=1001, reference_price=62000.0)
    informed = InformedTraderFlow(seed=2002)

    devs, two_sided, spreads, n_rest = [], 0, [], []
    n_trades = 0
    n_samples = 0
    t = 0.0
    while t < T:
        t += 1.0
        v = value_fn(t)
        for r in noise.run_until(eng, t):
            if r.fills:
                n_trades += 1
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
        "trade_rate": n_trades / T,
    }


print("Holding the emergent trade rate at the Phase 3 value (mult*p_market~1)")
print("Phase 3 target trade rate: %.4f /s" % LAM_NOISE)
print("")
print("%-14s %-9s %-9s %11s %11s %12s %13s %11s %11s"
      % ("variant", "lam_mult", "p_market", "mean|mid-V|", "med|mid-V|",
         "%two-sided", "mean resting", "med spread", "trade/s"))
for name, cls in (("C2 symmetric", NoiseTraderFlow),
                  ("C2b half-norm", HalfNormalNoise)):
    for mult, pm in ((1, 0.5), (5, 0.2), (20, 0.05), (50, 0.02),
                     (200, 0.005)):
        x = run(cls, mult, pm)
        print("%-14s %-9d %-9.3f %11.2f %11.2f %12.1f %13.1f %11.2f %11.4f"
              % (name, mult, pm, x["mean_dev"], x["median_dev"],
                 x["pct_two_sided"], x["mean_resting"], x["median_spread"],
                 x["trade_rate"]))
