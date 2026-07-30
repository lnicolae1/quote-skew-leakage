# diag_coherence.py: why C2b limit orders are ~40% marketable

import math, statistics
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

SPY = 365*24*3600
T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T/dt), seed=31415)


def resid(engine):
    bo = ao = None
    for o in engine.orders.values():
        if str(o.agent_id) == "MM": continue
        if o.side == "buy":
            if bo is None or o.price > bo.price: bo = o
        else:
            if ao is None or o.price < ao.price: ao = o
    return bo, ao


def run(mult, pm, life, disp):
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")
    noise = NoiseTraderFlow(lam=LAM_NOISE*mult, p_market=pm, seed=1001,
                            value_fn=value_fn, reference_price=62000.0,
                            mean_lifetime=life, disp=disp)
    informed = InformedTraderFlow(seed=2002)
    mm = MarketMaker(horizon=T)
    unc, con, ages = [], [], []
    r2 = tw = n_trades = n_lim = n_lim_mkt = 0
    birth = {oid: 0.0 for oid in eng.orders}
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.order_type == "limit":
                n_lim += 1
                if r.fills: n_lim_mkt += 1
            if r.order_id is not None: birth.setdefault(r.order_id, t)
            if r.fills:
                n_trades += 1; mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, value_fn):
            if getattr(r, "order_id", None) is not None:
                birth.setdefault(r.order_id, t)
            if r.fills:
                n_trades += 1; mm.on_fills(r.fills, r.side)
        mm.requote(eng, t)
        bo, ao = resid(eng)
        v, m = value_fn(t), eng.mid()
        if eng.best_bid() is not None and eng.best_ask() is not None: tw += 1
        if bo: ages.append(t-birth.get(bo.id, t))
        if ao: ages.append(t-birth.get(ao.id, t))
        if m is not None:
            unc.append(abs(m-v))
            if bo and ao: r2 += 1; con.append(abs(m-v))
    ages.sort()
    return dict(trate=n_trades/T, uncond=statistics.mean(unc),
                cond=statistics.mean(con) if con else float('nan'),
                age=ages[len(ages)//2] if ages else float('nan'),
                r2=100.0*r2/86400, tw=100.0*tw/86400,
                lim_mkt=100.0*n_lim_mkt/max(1, n_lim),
                rest=sum(1 for o in eng.orders.values() if str(o.agent_id) != "MM"))


if __name__ == "__main__":
    print("Coherence ratio = sigma*sqrt(L/yr) / disp   (V-drift per lifetime, "
          "in units of the book's own width)")
    print("%-6s %-8s %7s %8s %9s %8s %10s %9s %8s %9s %7s"
          % ("life", "disp", "ratio", "trade/s", "lim mkt%", "uncond|e|",
             "cond|e|", "resid2sd", "age med", "full2sd", "resting"))
    print("-" * 112)
    for life, disp in ((180.0, 0.0002), (90.0, 0.0002), (30.0, 0.0002),
                       (10.0, 0.0002), (180.0, 0.0009), (180.0, 0.0020)):
        ratio = 0.3721*math.sqrt(life/SPY)/disp
        x = run(20, 0.05, life, disp)
        print("%-6.0f %-8.5f %7.2f %8.4f %8.1f%% %9.2f %9.2f %8.1f%% %7.0fs %8.2f%% %7d"
              % (life, disp, ratio, x["trate"], x["lim_mkt"], x["uncond"],
                 x["cond"], x["r2"], x["age"], x["tw"], x["rest"]))
