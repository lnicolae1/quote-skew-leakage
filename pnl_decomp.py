# pnl_decomp.py: exact PnL decomposition into spread and inventory components

import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker
import sim_run

HORIZON = 8640000.0
DURATION = 259200.0


def decomp(gamma, seed, k=None):
    n = int(DURATION)
    sv, sn, si = (sim_run.SEED_V_BASE + 1000 * seed,
                  sim_run.SEED_NOISE_BASE + 1000 * seed,
                  sim_run.SEED_INF_BASE + 1000 * seed)
    path = generate_value_path(sim_run.SIGMA, 62000.0, 1.0, n, seed=sv)
    vf = path_value(path, 1.0)
    eng = MatchingEngine()
    sim_run._seed_book(eng)
    noise = NoiseTraderFlow(value_fn=vf, reference_price=62000.0, seed=sn)
    informed = InformedTraderFlow(seed=si)
    kw = {} if k is None else {"k": k}
    mm = MarketMaker(gamma=gamma, horizon=HORIZON, **kw)

    t = 0.0
    while t < DURATION:
        t += 1.0
        for r in noise.run_until(eng, t):
            if getattr(r, "fills", None):
                mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, vf):
            if getattr(r, "fills", None):
                mm.on_fills(r.fills, r.side)
        mm.requote(eng, t)

    rows = mm.log.private_view()
    qs = [r.true_inventory_q for r in rows]
    ss = [r.observed_mid for r in rows]
    inv = sum(qs[i] * (ss[i + 1] - ss[i]) for i in range(len(qs) - 1))
    pnl = mm.mark_to_market(ss[-1])
    return {
        "gamma": gamma, "seed": seed, "k": mm.k,
        "pnl": pnl, "inventory_pnl": inv, "spread_pnl": pnl - inv,
        "mean_q": statistics.mean(qs), "terminal_q": qs[-1],
        "fills": mm.n_buy_fills + mm.n_sell_fills,
        "S_move": ss[-1] - ss[0],
    }


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=float, default=None,
                    help="MM fill-decay coefficient; omit to use DEFAULT_K")
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[1e-8, 1e-7, 2e-7, 1e-6],
                    help="gamma grid to decompose")
    args = ap.parse_args()

    K = args.k
    if K is None and os.environ.get("SWEEP_K"):
        K = float(os.environ["SWEEP_K"])

    print("Exact PnL decomposition   PnL = INV + spread,  INV = sum q_t dS_t")
    print("k = %s, horizon=%.0f, 3 days"
          % ("None -> DEFAULT_K (1.5, the original placeholder)" if K is None
             else "%.5f (passed in; DEFAULT_K unedited)" % K, HORIZON))
    print("=" * 104)
    print("%9s %6s | %10s %12s %12s | %9s %8s %8s"
          % ("C", "seed", "PnL", "INV(advsel)", "spread", "mean_q", "dS", "fills"))
    print("-" * 104)
    for gamma in args.gammas:
        C = gamma * 1.45817e8
        rows = [decomp(gamma, s, k=K) for s in range(5)]
        for r in rows:
            print("%9.3f %6d | %10.2f %12.2f %12.2f | %+9.4f %8.0f %8d"
                  % (C, r["seed"], r["pnl"], r["inventory_pnl"],
                     r["spread_pnl"], r["mean_q"], r["S_move"], r["fills"]))
        print("%9.3f %6s | %10.2f %12.2f %12.2f | %+9.4f %8s %8.0f   <== mean"
              % (C, "mean", statistics.mean(x["pnl"] for x in rows),
                 statistics.mean(x["inventory_pnl"] for x in rows),
                 statistics.mean(x["spread_pnl"] for x in rows),
                 statistics.mean(x["mean_q"] for x in rows), "",
                 statistics.mean(x["fills"] for x in rows)))
        print("-" * 104)
