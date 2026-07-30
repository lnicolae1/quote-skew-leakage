# run_gamma_sweep.py: gamma sweep driver over sim_run.run_sim; writes gamma_sweep_results.txt

import os
import statistics
import time

import sim_run

_HERE = os.path.dirname(os.path.abspath(__file__))

HORIZON = 8640000.0
DURATION = 259200.0
SEEDS = [0, 1, 2, 3, 4]
GRID = [1e-9, 3e-9, 5e-9, 1e-8, 3e-8, 6e-8, 1e-7, 2e-7, 4e-7, 1e-6, 3e-6]

K_USED = float(os.environ.get("SWEEP_K", "0")) or None
_tag = ("_k%.5f" % K_USED) if K_USED else ""
OUT = os.path.join(_HERE, "gamma_sweep_results%s.txt" % _tag)
PKL = os.path.join(_HERE, "sweep_raw%s.pkl" % _tag)

FIELDS = [
    "k", "mean_q", "mm_passive_volume_btc", "mm_aggressive_volume_btc",
    "C_at_t0", "C_at_end", "mean_C",
    "ar1_phi_raw", "ar1_phi_raw_stderr", "implied_half_life_raw_s",
    "ar1_phi_demeaned", "ar1_phi_demeaned_stderr", "implied_half_life_demeaned_s",
    "sd_q", "max_abs_q", "terminal_q", "n_sign_changes",
    "frac_time_abs_q_below_0.1",
    "skew_mean", "skew_sd", "skew_mean_abs", "skew_p95_abs",
    "skew_sd_over_median_mm_spread",
    "DIAGNOSTIC_ols_q_on_skew_r2", "DIAGNOSTIC_ols_q_on_skew_slope",
    "DIAGNOSTIC_ols_q_on_tickrounded_skew_r2",
    "mm_fills_total", "mm_fills_per_hour", "mean_inter_fill_time_s",
    "mm_share_of_volume", "mm_volume_btc", "total_executed_volume_btc",
    "terminal_mark_to_market_pnl", "pnl_per_fill",
    "median_mm_quoted_spread", "median_residual_touch_spread",
    "n_resid_spreads", "median_residual_touch_spread_excluding_seed",
    "n_resid_spreads_ns", "mm_spread_over_residual_spread",
    "mean_abs_mid_minus_V", "book_two_sided_fraction",
    "emergent_trade_rate_per_s", "median_touch_order_age_s",
    "birth_lookup_hit_rate", "frac_residual_bid_is_seed",
    "frac_residual_ask_is_seed",
]


def agg(vals):
    """(mean, across-seed sd, n_present)"""
    good = [v for v in vals if v is not None]
    if not good:
        return None, None, 0
    m = statistics.mean(good)
    s = statistics.stdev(good) if len(good) > 1 else 0.0
    return m, s, len(good)


def fmt(v, w=10, p=4):
    if v is None:
        return "None".rjust(w)
    if isinstance(v, int):
        return str(v).rjust(w)
    if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-4):
        return ("%.*e" % (2, v)).rjust(w)
    return ("%.*f" % (p, v)).rjust(w)


def main():
    t_start = time.time()
    results = {}
    fh = open(OUT, "w")

    fh.write("Gamma sweep; raw per-run output\n")
    fh.write("=" * 100 + "\n")
    fh.write("horizon_s = %.1f (100 days, so tau decays ~3%% over the run and C is\n"
             "            effectively constant within each run)\n" % HORIZON)
    fh.write("duration_s = %.1f (3 days)   seeds = %s\n" % (DURATION, SEEDS))
    fh.write("k = %s  (constructor arg; DEFAULT_K never edited)\n"
             % (K_USED if K_USED else "1.5; default placeholder"))
    fh.write("Locked config elsewhere: lam=0.2255, p_market=0.02, mean_lifetime=180.0,\n"
             "            disp=0.0020, sigma=0.3721.\n")
    fh.write("DEFAULT_LAMBDA=0.0451 is a Phase-3 measurement, untouched.\n")
    fh.write("None in ar1_*/ols_* at high gamma = the MM never traded, so there is no\n"
             "            inventory process to measure. That is an answer, not missing data.\n")
    fh.write("=" * 100 + "\n\n")

    for g in GRID:
        runs = []
        for s in SEEDS:
            t0 = time.time()
            d = sim_run.run_sim(gamma=g, horizon_s=HORIZON,
                                duration_s=DURATION, seed=s, k=K_USED)
            el = time.time() - t0
            runs.append(d)
            print("  gamma=%-9.0e seed=%d  C=%8.3f  fills=%6d  share=%s  phi_raw=%s  (%.1fs)"
                  % (g, s, d["C_at_t0"], d["mm_fills_total"],
                     fmt(d["mm_share_of_volume"], 6, 3),
                     fmt(d["ar1_phi_raw"], 8, 5), el))
            fh.write("-" * 100 + "\n")
            fh.write("run gamma=%.3e  C_at_t0=%.5f  seed=%d\n" % (g, d["C_at_t0"], s))
            fh.write("-" * 100 + "\n")
            for k in sorted(d.keys()):
                if k == "notes":
                    continue
                fh.write("  %-46s %s\n" % (k, d[k]))
            fh.write("\n")
        results[g] = runs
        fh.flush()

    fh.write("\n" + "=" * 100 + "\n")
    fh.write("per-gamma aggregates (mean +/- across-seed sd, n = seeds producing a number)\n")
    fh.write("=" * 100 + "\n")
    for g in GRID:
        runs = results[g]
        fh.write("\ngamma = %.3e   C_at_t0 = %.4f\n" % (g, runs[0]["C_at_t0"]))
        for f in FIELDS:
            m, s, n = agg([r[f] for r in runs])
            if n == 0:
                fh.write("  %-46s None (all %d seeds)\n" % (f, len(runs)))
            else:
                fh.write("  %-46s %s  +/- %s   [n=%d]\n"
                         % (f, fmt(m, 14, 5), fmt(s, 12, 5), n))
    fh.write("\nnotes: %s\n" % runs[0]["notes"])
    fh.close()

    print("\nTOTAL wall clock: %.1fs" % (time.time() - t_start))

    import pickle
    with open(PKL, "wb") as p:
        pickle.dump(results, p)
    print("raw results pickled for table rendering")


if __name__ == "__main__":
    main()
