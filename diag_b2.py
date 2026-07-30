# diag_b2.py: B2 evaluation, same harness as diag_lag_stale.py

import statistics, sys
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import (NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE,
                           DEFAULT_MEAN_LIFETIME)
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

dt = 1.0
SEED_V, SEED_N, SEED_I = 31415, 1001, 2002


def residual_touch_orders(engine):
    bo = ao = None
    for o in engine.orders.values():
        if str(o.agent_id) == "MM":
            continue
        if o.side == "buy":
            if bo is None or o.price > bo.price:
                bo = o
        else:
            if ao is None or o.price < ao.price:
                ao = o
    return bo, ao


def run(with_mm, T, life=DEFAULT_MEAN_LIFETIME):
    n_steps = int(T / dt)
    path = generate_value_path(0.3721, 62000.0, dt, n_steps, seed=SEED_V)
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")
    noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=SEED_N,
                            value_fn=value_fn, reference_price=62000.0,
                            mean_lifetime=life)
    informed = InformedTraderFlow(seed=SEED_I)
    mm = MarketMaker(horizon=T) if with_mm else None

    birth = {oid: 0.0 for oid in eng.orders}
    rows = []
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.order_id is not None:
                birth.setdefault(r.order_id, t)
            if r.fills and mm is not None:
                mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, value_fn):
            if getattr(r, "order_id", None) is not None:
                birth.setdefault(r.order_id, t)
            if r.fills and mm is not None:
                mm.on_fills(r.fills, r.side)
        if mm is not None:
            mm.requote(eng, t)
        bo, ao = residual_touch_orders(eng)
        rmid = (bo.price + ao.price)/2.0 if (bo and ao) else None
        rows.append((t, eng.best_bid(), eng.best_ask(), eng.mid(), rmid,
                     value_fn(t),
                     t - birth.get(bo.id, t) if bo else None,
                     t - birth.get(ao.id, t) if ao else None))
    return path, eng, rows, mm, noise


def stats(rows, eng, mm, noise, tag):
    n = len(rows)
    unc = [r[3]-r[5] for r in rows if r[3] is not None]
    con = [r[3]-r[5] for r in rows if r[3] is not None and r[4] is not None]
    two = sum(1 for r in rows if r[1] is not None and r[2] is not None)
    ages = sorted(a for r in rows for a in (r[6], r[7]) if a is not None)
    rest = sum(1 for o in eng.orders.values() if str(o.agent_id) != "MM")
    resid2 = sum(1 for r in rows if r[4] is not None)
    print("  %s" % tag)
    print("    mean|mid-V| unconditional      : $%7.2f  (n=%d)"
          % (statistics.mean(map(abs, unc)), len(unc)))
    print("    mean|mid-V| conditioned        : $%7.2f  (n=%d)"
          % (statistics.mean(map(abs, con)) if con else float('nan'), len(con)))
    print("    signed mean (uncond)           : $%+7.2f" % statistics.mean(unc))
    print("    touch-order age  median %6.0fs   mean %6.0fs   p90 %6.0fs"
          % (ages[len(ages)//2], statistics.mean(ages), ages[int(0.9*len(ages))]))
    print("    non-MM orders resting at end   : %d" % rest)
    print("    residual(non-MM) book two-sided: %d/%d (%.1f%%)"
          % (resid2, n, 100.0*resid2/n))
    print("    full book two-sided            : %d/%d (%.2f%%)"
          % (two, n, 100.0*two/n))
    print("    B2 cancels: scheduled=%d  cancelled=%d  already filled=%d"
          % (noise.n_scheduled, noise.n_cancelled, noise.n_already_gone))
    if mm is not None:
        sp = sorted(r[2]-r[1] for r in rows if r[1] is not None and r[2] is not None)
        print("    MM: fills=%d  q_final=%+.5f  median touch spread $%.2f"
              % (mm.n_buy_fills+mm.n_sell_fills, mm.q, sp[len(sp)//2]))
    print("")


LIFE = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MEAN_LIFETIME
T1 = 86400.0
print("=" * 74)
print("B2 active; mean_lifetime = %.0f s" % LIFE)
print("=" * 74)
p0, e0, rows0, _, n0 = run(False, T1, LIFE)
stats(rows0, e0, None, n0,
      "NO-MM   [pre-B2: uncond $61.88, age med 247s, resting 9]")
p1, e1, rows1, mm1, n1 = run(True, T1, LIFE)
stats(rows1, e1, mm1, n1,
      "WITH-MM [pre-B2: uncond $282.85, cond $197.33, age med 1634s, "
      "resting 56, 99.99% two-sided]")

print("=" * 74)
print("3-day with-MM run  [pre-B2 by day: $235.92 / $247.25 / $296.71]")
print("=" * 74)
p3, e3, rows3, mm3, n3 = run(True, 3*86400.0, LIFE)
for d in range(3):
    seg = rows3[d*86400:(d+1)*86400]
    u = [r[3]-r[5] for r in seg if r[3] is not None]
    tw = sum(1 for r in seg if r[1] is not None and r[2] is not None)
    print("  day %d: mean|mid-V| $%7.2f   signed $%+8.2f   two-sided %.2f%%   "
          "V %.0f -> %.0f" % (d+1, statistics.mean(map(abs, u)),
                              statistics.mean(u), 100.0*tw/len(seg),
                              seg[0][5], seg[-1][5]))
allu = [r[3]-r[5] for r in rows3 if r[3] is not None]
print("  whole 3 days: mean|mid-V| $%.2f   signed $%+.2f"
      % (statistics.mean(map(abs, allu)), statistics.mean(allu)))
print("  MM q at 3 days: %+.5f BTC   [pre-B2: -1.85951]" % mm3.q)
