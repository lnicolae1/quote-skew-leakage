# diag_c2b.py: C2b supply sweep

import statistics, sys
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

dt = 1.0


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


def run(mult, pm, life=180.0, T=86400.0):
    n_steps = int(T/dt)
    path = generate_value_path(0.3721, 62000.0, dt, n_steps, seed=31415)
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
    unc, con, ages, rows = [], [], [], []
    r2 = tw = 0
    n_trades = n_lim = n_lim_mkt = n_buy = n_sell = 0
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.side == "buy": n_buy += 1
            else: n_sell += 1
            if r.order_type == "limit":
                n_lim += 1
                if r.fills: n_lim_mkt += 1
            if r.order_id is not None: birth.setdefault(r.order_id, t)
            if r.fills:
                n_trades += 1
                mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, value_fn):
            if getattr(r, "order_id", None) is not None:
                birth.setdefault(r.order_id, t)
            if r.fills:
                n_trades += 1
                mm.on_fills(r.fills, r.side)
        mm.requote(eng, t)
        bo, ao = resid(eng)
        v, m = value_fn(t), eng.mid()
        if eng.best_bid() is not None and eng.best_ask() is not None: tw += 1
        if bo: ages.append(t - birth.get(bo.id, t))
        if ao: ages.append(t - birth.get(ao.id, t))
        if m is not None:
            unc.append(abs(m-v))
            if bo and ao:
                r2 += 1; con.append(abs(m-v))
        rows.append((m, v))
    ages.sort()
    return dict(trate=n_trades/T, uncond=statistics.mean(unc),
                cond=statistics.mean(con) if con else float('nan'),
                age=ages[len(ages)//2] if ages else float('nan'),
                r2=100.0*r2/n_steps, tw=100.0*tw/n_steps,
                rest=sum(1 for o in eng.orders.values() if str(o.agent_id) != "MM"),
                lim_mkt=100.0*n_lim_mkt/max(1, n_lim),
                buyfrac=100.0*n_buy/max(1, n_buy+n_sell), q=mm.q, rows=rows,
                canc=noise.n_cancelled, gone=noise.n_already_gone)


if len(sys.argv) > 1 and sys.argv[1] == "3day":
    mult, pm = float(sys.argv[2]), float(sys.argv[3])
    x = run(mult, pm, T=3*86400.0)
    print("3-day with-MM, C2b, mult=%g p_market=%g  "
          "[pre-B2 by day: $235.92/$247.25/$296.71; "
          "post-B2 C2: $263.96/$207.91/$289.27]" % (mult, pm))
    for d in range(3):
        seg = x["rows"][d*86400:(d+1)*86400]
        u = [m-v for (m, v) in seg if m is not None]
        print("  day %d: mean|mid-V| $%7.2f   signed $%+8.2f"
              % (d+1, statistics.mean(map(abs, u)), statistics.mean(u)))
    allu = [m-v for (m, v) in x["rows"] if m is not None]
    print("  whole 3 days: mean|mid-V| $%.2f  signed $%+.2f"
          % (statistics.mean(map(abs, allu)), statistics.mean(allu)))
    print("  trade rate %.4f/s   two-sided %.2f%%   resid 2-sided %.1f%%   "
          "age med %.0fs" % (x["trate"], x["tw"], x["r2"], x["age"]))
    print("  MM q at 3 days: %+.5f BTC" % x["q"])
    sys.exit()

print("C2b supply sweep (mean_lifetime=180s, mult*p_market held = 1)")
print("Phase 3 trade-rate target: %.4f/s" % LAM_NOISE)
print("%-6s %-7s %8s %10s %9s %8s %9s %9s %7s %8s %7s"
      % ("mult", "p_mkt", "trade/s", "uncond|e|", "cond|e|", "age med",
         "resid2sd", "full2sd", "resting", "lim mkt%", "buy%"))
print("-" * 108)
for mult, pm in ((1, 0.5), (2, 0.5), (5, 0.2), (10, 0.1), (20, 0.05)):
    x = run(mult, pm)
    print("%-6d %-7.3f %8.4f %10.2f %9.2f %8.0f %8.1f%% %8.2f%% %7d %7.1f%% %6.1f%%"
          % (mult, pm, x["trate"], x["uncond"], x["cond"], x["age"], x["r2"],
             x["tw"], x["rest"], x["lim_mkt"], x["buyfrac"]))
