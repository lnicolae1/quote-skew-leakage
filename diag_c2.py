# diag_c2.py: C2 failures: order starvation hypothesis

import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T / dt), seed=31415)


def run(lam_mult, disp=0.0002):
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000 - (i + 1) * 0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000 + (i + 1) * 0.5, 2), 0.02, "seed")

    noise = NoiseTraderFlow(lam=LAM_NOISE * lam_mult, p_market=0.5,
                            value_fn=value_fn, disp=disp, seed=1001,
                            reference_price=62000.0)
    informed = InformedTraderFlow(seed=2002)

    devs, two_sided, n_rest, spreads = [], 0, [], []
    noise_vol = informed_vol = 0.0
    n_samples = 0
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            noise_vol += sum(f.size for f in r.fills)
        for r in informed.run_until(eng, t, value_fn):
            informed_vol += sum(f.size for f in r.fills)
        bb, ba, m = eng.best_bid(), eng.best_ask(), eng.mid()
        n_samples += 1
        n_rest.append(len(eng.orders))
        if m is not None:
            devs.append(abs(m - value_fn(t)))
            two_sided += 1
            spreads.append(ba - bb)
    return {
        "lam_mult": lam_mult, "disp": disp,
        "mean_dev": statistics.mean(devs) if devs else float("nan"),
        "median_dev": statistics.median(devs) if devs else float("nan"),
        "pct_two_sided": 100.0 * two_sided / n_samples,
        "mean_resting": statistics.mean(n_rest),
        "median_spread": statistics.median(spreads) if spreads else float("nan"),
        "noise_arr": noise.n_arrivals, "inf_arr": informed.n_arrivals,
        "noise_vol": noise_vol, "informed_vol": informed_vol,
    }


print("Sensitivity: is the book starved? (disp fixed at 0.0002)")
print("%-9s %-11s %11s %11s %13s %12s %12s"
      % ("lam_mult", "lam(/s)", "mean|mid-V|", "med|mid-V|", "%two-sided",
         "mean resting", "med spread"))
results = []
for mult in (1, 5, 20, 50):
    r = run(mult)
    results.append(r)
    print("%-9d %-11.4f %11.2f %11.2f %13.1f %12.1f %12.2f"
          % (mult, LAM_NOISE * mult, r["mean_dev"], r["median_dev"],
             r["pct_two_sided"], r["mean_resting"], r["median_spread"]))

print("")
print("informed volume share; reconciling the ~2%% vs ~4%% wobble")
base = results[0]
print("  measured executed volume this run (lam_mult=1):")
print("    noise executed    : %.5f BTC" % base["noise_vol"])
print("    informed executed : %.5f BTC" % base["informed_vol"])
tot = base["noise_vol"] + base["informed_vol"]
print("    informed share of executed volume: %.2f%%"
      % (100.0 * base["informed_vol"] / tot))
print("")
lam_i = 0.10 * LAM_NOISE
theory = (lam_i * 0.00076) / (lam_i * 0.00076 + LAM_NOISE * 0.00347)
print("  theoretical share of submitted volume:")
print("    informed rate x size = %.6f x %.5f = %.8f BTC/s"
      % (lam_i, 0.00076, lam_i * 0.00076))
print("    noise    rate x size = %.6f x %.5f = %.8f BTC/s"
      % (LAM_NOISE, 0.00347, LAM_NOISE * 0.00347))
print("    => informed share of submitted volume: %.2f%%" % (100.0 * theory))
print("")
print("  arrivals: noise=%d informed=%d (informed %.1f%% by count)"
      % (base["noise_arr"], base["inf_arr"],
         100.0 * base["inf_arr"] / (base["noise_arr"] + base["inf_arr"])))
