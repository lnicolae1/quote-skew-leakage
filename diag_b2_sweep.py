# diag_b2_sweep.py: B2 at mean_lifetime=180s: anchor freshness vs two-sidedness

import statistics
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T/dt), seed=31415)


def resid(engine):
    bo = ao = None
    for o in engine.orders.values():
        if str(o.agent_id) == "MM":
            continue
        if o.side == "buy":
            if bo is None or o.price > bo.price: bo = o
        else:
            if ao is None or o.price < ao.price: ao = o
    return bo, ao


def run(life, mult, pm):
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")
    noise = NoiseTraderFlow(lam=LAM_NOISE*mult, p_market=pm, seed=1001,
                            value_fn=value_fn, reference_price=62000.0,
                            mean_lifetime=life)
    informed = InformedTraderFlow(seed=2002)
    mm = MarketMaker(horizon=T)
    birth = {oid: 0.0 for oid in eng.orders}
    unc, con, ages, r2, tw, ntr = [], [], [], 0, 0, 0
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.order_id is not None: birth.setdefault(r.order_id, t)
            if r.fills:
                ntr += 1
                mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, value_fn):
            if getattr(r, "order_id", None) is not None:
                birth.setdefault(r.order_id, t)
            if r.fills:
                ntr += 1
                mm.on_fills(r.fills, r.side)
        mm.requote(eng, t)
        bo, ao = resid(eng)
        v, m = value_fn(t), eng.mid()
        if eng.best_bid() is not None and eng.best_ask() is not None: tw += 1
        if bo: ages.append(t - birth.get(bo.id, t))
        if ao: ages.append(t - birth.get(ao.id, t))
        if m is not None:
            unc.append(abs(m - v))
            if bo and ao:
                r2 += 1
                con.append(abs(m - v))
    ages.sort()
    return dict(uncond=statistics.mean(unc), cond=statistics.mean(con) if con else float('nan'),
                age=ages[len(ages)//2] if ages else float('nan'),
                r2=100.0*r2/86400, tw=100.0*tw/86400,
                rest=sum(1 for o in eng.orders.values() if str(o.agent_id) != "MM"),
                trate=ntr/T, q=mm.q)


print("%-7s %-6s %-7s %10s %10s %9s %11s %10s %7s %9s"
      % ("life", "mult", "p_mkt", "uncond|e|", "cond|e|", "age med",
         "resid 2-sided", "full 2-sd", "resting", "trade/s"))
print("-" * 100)
for mult, pm in ((1, 0.5), (10, 0.05)):
    for life in (180.0, 600.0, 1800.0):
        x = run(life, mult, pm)
        print("%-7.0f %-6d %-7.3f %10.2f %10.2f %9.0f %11.1f%% %9.2f%% %7d %9.4f"
              % (life, mult, pm, x["uncond"], x["cond"], x["age"], x["r2"],
                 x["tw"], x["rest"], x["trate"]))
print("")
print("Phase 3 target trade rate: %.4f/s.  Pre-B2 with-MM: uncond $282.85, "
      "cond $197.33, age 1634s, resid 79.8%%, resting 56." % LAM_NOISE)
