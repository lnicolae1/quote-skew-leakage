# clipped_placement.py: noise flow with clipped placement
# - a limit order that would cross the opposite touch is moved to rest instead of executing
#   JOIN: rest at the near touch; TOUCH: rest one tick inside the opposite touch
# - subclass of NoiseTraderFlow overriding limit_price(); super() is called first so the RNG stream is unchanged

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sim_run
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from informed_traders import InformedTraderFlow, path_value
from noise_traders import (NoiseTraderFlow, DEFAULT_ORDER_RATE,
                           DEFAULT_P_MARKET)

SIGMA, SPY = 0.3721, 365 * 24 * 3600
T = 86400.0
SEEDS = [0, 1, 2]
LAM = DEFAULT_ORDER_RATE
P_MARKET = DEFAULT_P_MARKET
TARGET = 0.0451

POINTS = [
    ("(i)   disp=0.0009 life=180", 0.0009, 180.0),
    ("(ii)  disp=0.0013 life=433", 0.0013, 433.0),
    ("(iii) disp=0.0020 life=180", 0.0020, 180.0),
]

# unclipped reference results (bookwidth_results.txt)
UNCLIPPED = {
    "(i)   disp=0.0009 life=180": dict(med=1.5825, mean=2.2511, f=0.1427,
                                       rest=15.2, ev=0.0397, agg=0.0611),
    "(ii)  disp=0.0013 life=433": dict(med=1.2297, mean=1.9721, f=0.1578,
                                       rest=34.6, ev=0.0433, agg=0.0698),
    "(iii) disp=0.0020 life=180": dict(med=2.7210, mean=3.5656, f=0.0652,
                                       rest=17.5, ev=0.0222, agg=0.0347),
}


class ClippedNoiseTraderFlow(NoiseTraderFlow):
    """NoiseTraderFlow whose would-be crossing limit orders are clipped to rest."""

    def __init__(self, *a, **kw):
        self.clip_mode = kw.pop("clip_mode", "join")
        if self.clip_mode not in ("join", "touch"):
            raise ValueError("clip_mode must be 'join' or 'touch'")
        self.n_clipped = 0
        super().__init__(*a, **kw)

    def limit_price(self, engine, side):
        price = super().limit_price(engine, side)

        if side == "buy":
            opp = engine.best_ask()
            # buys match while best_ask <= price: crossing is >=
            if opp is None or price < opp:
                return price
            self.n_clipped += 1
            near = engine.best_bid()
            if self.clip_mode == "join" and near is not None:
                return near
            return self._round_to_tick(opp - self.tick)
        else:
            opp = engine.best_bid()
            # sells match while best_bid >= price: crossing is <=
            if opp is None or price > opp:
                return price
            self.n_clipped += 1
            near = engine.best_ask()
            if self.clip_mode == "join" and near is not None:
                return near
            return self._round_to_tick(opp + self.tick)


def make_world(seed, T, lam, p_market, life, disp, clip_mode):
    """Same construction and seeds as diag_bookwidth._make_world, with the flow under test."""
    n = int(T)
    path = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE, 1.0, n,
                               seed=sim_run.SEED_V_BASE + 1000 * seed)
    vf = path_value(path, 1.0)
    eng = MatchingEngine()
    sim_run._seed_book(eng)
    kw = dict(lam=lam, p_market=p_market, mean_lifetime=life, disp=disp,
              value_fn=vf, reference_price=sim_run.REFERENCE_PRICE,
              seed=sim_run.SEED_NOISE_BASE + 1000 * seed)
    if clip_mode is None:
        noise = NoiseTraderFlow(**kw)
    else:
        noise = ClippedNoiseTraderFlow(clip_mode=clip_mode, **kw)
    informed = InformedTraderFlow(seed=sim_run.SEED_INF_BASE + 1000 * seed)
    return path, vf, eng, noise, informed


def run(seed, disp, life, clip_mode, lam=LAM, p_market=P_MARKET):
    _p, vf, eng, noise, informed = make_world(seed, T, lam, p_market, life,
                                              disp, clip_mode)
    n_event = n_agg = n_fill = 0
    n_limit = n_limit_marketable = 0
    spreads_bp = []
    orders_side, levels_side = [], []

    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.order_type == "limit":
                n_limit += 1
                if r.fills:
                    n_limit_marketable += 1
            if r.fills:
                n_event += 1
                n_agg += len(set(f.price for f in r.fills))
                n_fill += len(r.fills)
        for r in informed.run_until(eng, t, vf):
            if getattr(r, "fills", None):
                n_event += 1
                n_agg += len(set(f.price for f in r.fills))
                n_fill += len(r.fills)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            spreads_bp.append(1e4 * (ba - bb) / (0.5 * (bb + ba)))

        # orders and distinct price levels per side (non-seed)
        nb = na = 0
        pb, pa = set(), set()
        for o in eng.orders.values():
            if str(o.agent_id) == "seed":
                continue
            if o.side == "buy":
                nb += 1; pb.add(o.price)
            else:
                na += 1; pa.add(o.price)
        orders_side.append(0.5 * (nb + na))
        levels_side.append(0.5 * (len(pb) + len(pa)))

    return {
        "spreads_bp": spreads_bp,
        "orders_per_side": statistics.mean(orders_side),
        "levels_per_side": statistics.mean(levels_side),
        "ev_per_s": n_event / T, "agg_per_s": n_agg / T,
        "n_agg": n_agg, "n_fill": n_fill,
        "f": n_limit_marketable / max(1, n_limit),
        "n_clipped": getattr(noise, "n_clipped", 0),
        "n_limit": n_limit,
        "two_sided_pct": 100.0 * len(spreads_bp) / T,
    }


def measure(disp, life, clip_mode, lam=LAM, p_market=P_MARKET):
    pooled = []
    acc = {k: [] for k in ("orders_per_side", "levels_per_side", "ev_per_s",
                           "agg_per_s", "f", "two_sided_pct")}
    n_agg = n_fill = n_clip = n_lim = 0
    t0 = time.time()
    for s in SEEDS:
        x = run(s, disp, life, clip_mode, lam, p_market)
        pooled += x["spreads_bp"]
        for k in acc:
            acc[k].append(x[k])
        n_agg += x["n_agg"]; n_fill += x["n_fill"]
        n_clip += x["n_clipped"]; n_lim += x["n_limit"]
    pooled.sort()
    out = {k: statistics.mean(v) for k, v in acc.items()}
    out.update(median_bp=pooled[len(pooled) // 2],
               mean_bp=statistics.mean(pooled),
               n_agg=n_agg, n_fill=n_fill, n_clipped=n_clip, n_limit=n_lim,
               secs=time.time() - t0)
    return out


def main():
    print("=" * 108)
    print("Clipped placement: does refusing to cross decouple width "
          "from marketability?")
    print("=" * 108)
    print("MM ABSENT, informed present. lam=%.4f p_market=%.3f. %d seeds x "
          "%.0fs. Target %.4f aggTrades/s."
          % (LAM, P_MARKET, len(SEEDS), T, TARGET))
    print("JOIN  = would-be crossing order rests at the near touch (joins the queue).")
    print("TOUCH = would-be crossing order rests one tick inside the opposite touch.")
    print("")

    hdr = ("%-28s %-7s %9s %9s %7s %8s %8s %8s %9s %9s"
           % ("point", "mode", "median bp", "mean bp", "f", "ord/sd",
              "lvl/sd", "ev/s", "aggTr/s", "vs .0451"))
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for tag, disp, life in POINTS:
        u = UNCLIPPED[tag]
        print("%-28s %-7s %9.4f %9.4f %6.1f%% %8.1f %8s %8.4f %9.4f %8.2fx"
              % (tag, "none", u["med"], u["mean"], 100 * u["f"], u["rest"],
                 "~=ord", u["ev"], u["agg"], u["agg"] / TARGET))
        for mode in ("join", "touch"):
            r = measure(disp, life, mode)
            results[(tag, mode)] = r
            print("%-28s %-7s %9.4f %9.4f %6.1f%% %8.1f %8.1f %8.4f %9.4f %8.2fx"
                  % ("", mode.upper(), r["median_bp"], r["mean_bp"],
                     100 * r["f"], r["orders_per_side"], r["levels_per_side"],
                     r["ev_per_s"], r["agg_per_s"], r["agg_per_s"] / TARGET))
        print("")

    print("fill vs AGG: do orders still never share a price?")
    print("  (unclipped: 2982 fills vs 2980 aggTrades, MM absent)")
    print("  %-28s %-7s %10s %10s %10s %12s"
          % ("point", "mode", "fills", "aggTrades", "fill/agg", "clipped"))
    for tag, disp, life in POINTS:
        for mode in ("join", "touch"):
            r = results[(tag, mode)]
            print("  %-28s %-7s %10d %10d %10.4f %11.1f%%"
                  % (tag, mode.upper(), r["n_fill"], r["n_agg"],
                     r["n_fill"] / max(1, r["n_agg"]),
                     100.0 * r["n_clipped"] / max(1, r["n_limit"])))

    print("")
    print("f = 0 by construction under clipping (no noise limit")
    print("order executes on arrival). Column 'clipped' = share of limit")
    print("orders that would have crossed.")


if __name__ == "__main__":
    main()
