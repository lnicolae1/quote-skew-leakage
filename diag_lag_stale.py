# diag_lag_stale.py: no-MM vs with-MM mid error against V.
# - control: also scored only where the residual (non-MM) book is two-sided
# - Q2: touch-order age; Q3: lag vs bias; Q3b: 3-day run, error by day

import statistics, math
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

dt = 1.0
SEED_V, SEED_N, SEED_I = 31415, 1001, 2002


def residual_touch_orders(engine):
    """(best_bid_order, best_ask_order), MM quotes excluded."""
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


def run(with_mm, T):
    n_steps = int(T/dt)
    path = generate_value_path(0.3721, 62000.0, dt, n_steps, seed=SEED_V)
    value_fn = path_value(path, dt)
    eng = MatchingEngine()
    for i in range(5):
        eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")
    noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=SEED_N,
                            value_fn=value_fn, reference_price=62000.0)
    informed = InformedTraderFlow(seed=SEED_I)
    mm = MarketMaker(horizon=T) if with_mm else None

    birth = {}                    # order_id -> submission time
    for oid in eng.orders:
        birth[oid] = 0.0
    rows = []                     # (t, full_mid, resid_mid, V, bid_age, ask_age)
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if getattr(r, "order_id", None) is not None:
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
        rows.append((t, eng.mid(), rmid, value_fn(t),
                     t - birth.get(bo.id, t) if bo else None,
                     t - birth.get(ao.id, t) if ao else None))
    return path, eng, rows, mm


def err(rows, which, cond_resid=False):
    out = []
    for (_t, fm, rm, v, _ba, _aa) in rows:
        m = fm if which == "full" else rm
        if m is None:
            continue
        if cond_resid and rm is None:
            continue
        out.append(m - v)
    return out


T1 = 86400.0
print("=" * 72)
print("CONTROL : same-condition scoring")
print("=" * 72)
p0, e0, rows0, _ = run(False, T1)
p1, e1, rows1, mm1 = run(True, T1)

for tag, rows, which in (("no-MM   mid", rows0, "full"),
                         ("with-MM mid", rows1, "full")):
    u = err(rows, which)
    c = err(rows, which, cond_resid=True)
    print("  %s" % tag)
    print("     unconditional: n=%6d  mean|e| $%7.2f   signed mean $%+8.2f"
          % (len(u), statistics.mean(map(abs, u)), statistics.mean(u)))
    print("     residual (non-MM) book two-sided:")
    print("                     n=%6d  mean|e| $%7.2f   signed mean $%+8.2f"
          % (len(c), statistics.mean(map(abs, c)), statistics.mean(c)))
print("")

print("=" * 72)
print("Q2: age of orders at the non-MM touch")
print("=" * 72)
for tag, rows, eng in (("no-MM  ", rows0, e0), ("with-MM", rows1, e1)):
    ages = [a for r in rows for a in (r[4], r[5]) if a is not None]
    ages.sort()
    print("  %s  touch-order age: median %6.0f s   mean %7.0f s   "
          "p90 %7.0f s   max %7.0f s"
          % (tag, ages[len(ages)//2], statistics.mean(ages),
             ages[int(0.9*len(ages))], ages[-1]))
    rest = sum(1 for o in eng.orders.values() if str(o.agent_id) != "MM")
    print("           non-MM orders resting at end: %d" % rest)
    rm = [r[2] for r in rows if r[2] is not None]
    print("           residual(non-MM) book two-sided: %d/%d (%.1f%%)"
          % (len(rm), len(rows), 100.0*len(rm)/len(rows)))
print("")

print("=" * 72)
print("Q3: lag or bias; best shift of mid against V")
print("=" * 72)
V = [r[3] for r in rows1]
M = [r[1] for r in rows1]
best = None
for L in list(range(0, 1801, 60)) + list(range(1800, 7201, 600)):
    vals = [abs(M[i+L]-V[i]) for i in range(len(V)-L) if M[i+L] is not None]
    e = statistics.mean(vals)
    if best is None or e < best[1]:
        best = (L, e)
    if L % 300 == 0 or L in (60, 120):
        print("   lag %5ds : mean|mid(t+L) - V(t)| = $%.2f" % (L, e))
print("   -> best lag %ds at $%.2f  (lag 0 was $%.2f)"
      % (best[0], best[1], statistics.mean(
          [abs(M[i]-V[i]) for i in range(len(V)) if M[i] is not None])))
print("")

print("=" * 72)
print("Q3b: transient or persistent; 3-day with-MM run, error by day")
print("=" * 72)
p3, e3, rows3, mm3 = run(True, 3*86400.0)
for d in range(3):
    seg = rows3[d*86400:(d+1)*86400]
    u = [r[1]-r[3] for r in seg if r[1] is not None]
    print("   day %d: mean|mid-V| $%7.2f   signed mean $%+8.2f   "
          "V %.0f -> %.0f" % (d+1, statistics.mean(map(abs, u)),
                              statistics.mean(u), seg[0][3], seg[-1][3]))
allu = [r[1]-r[3] for r in rows3 if r[1] is not None]
print("   whole 3 days: mean|mid-V| $%.2f   signed mean $%+.2f"
      % (statistics.mean(map(abs, allu)), statistics.mean(allu)))
print("   MM q at end: %+.5f BTC" % mm3.q)
