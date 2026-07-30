# verify_defaults.py: locked-config check on bare module defaults
import statistics
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
import noise_traders as NT
from noise_traders import NoiseTraderFlow
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

print("module defaults: lam=%.4f/s (=%.4f x %.1f)  p_market=%.2f  "
      "mean_lifetime=%.0fs  disp=%.4f"
      % (NT.DEFAULT_ORDER_RATE, NT.DEFAULT_LAMBDA, NT.DEFAULT_SUPPLY_MULT,
         NT.DEFAULT_P_MARKET, NT.DEFAULT_MEAN_LIFETIME, NT.DEFAULT_DISP))
print("")

T, dt = 86400.0, 1.0
path = generate_value_path(0.3721, 62000.0, dt, int(T/dt), seed=31415)
value_fn = path_value(path, dt)
eng = MatchingEngine()
for i in range(5):
    eng.add_limit_order("buy", round(62000-(i+1)*0.5, 2), 0.02, "seed")
    eng.add_limit_order("sell", round(62000+(i+1)*0.5, 2), 0.02, "seed")

noise = NoiseTraderFlow(value_fn=value_fn, reference_price=62000.0, seed=1001)
informed = InformedTraderFlow(seed=2002)
mm = MarketMaker(horizon=T)

birth = {oid: 0.0 for oid in eng.orders}
devs, ages = [], []
n_tr = r2 = tw = n_lim = n_lim_mkt = n_buy = 0
n_all = 0
t = 0.0
while t < T:
    t += 1.0
    for r in noise.run_until(eng, t):
        n_all += 1
        if r.side == "buy": n_buy += 1
        if r.order_type == "limit":
            n_lim += 1
            if r.fills: n_lim_mkt += 1
        if r.order_id is not None: birth.setdefault(r.order_id, t)
        if r.fills:
            n_tr += 1; mm.on_fills(r.fills, r.side)
    for r in informed.run_until(eng, t, value_fn):
        if getattr(r, "order_id", None) is not None:
            birth.setdefault(r.order_id, t)
        if r.fills:
            n_tr += 1; mm.on_fills(r.fills, r.side)
    mm.requote(eng, t)
    bo = ao = None
    for o in eng.orders.values():
        if str(o.agent_id) == "MM": continue
        if o.side == "buy":
            if bo is None or o.price > bo.price: bo = o
        else:
            if ao is None or o.price < ao.price: ao = o
    if bo: ages.append(t-birth.get(bo.id, t))
    if ao: ages.append(t-birth.get(ao.id, t))
    if bo and ao: r2 += 1
    if eng.best_bid() is not None and eng.best_ask() is not None: tw += 1
    m = eng.mid()
    if m is not None: devs.append(abs(m-value_fn(t)))
ages.sort()
print("One-day with-MM run, bare defaults  [target]")
print("  emergent trade rate   : %.4f /s          [0.0416, Phase 3 0.0451]"
      % (n_tr/T))
print("  mean |mid - V|        : $%.2f            [~$57]" % statistics.mean(devs))
print("  full book two-sided   : %.2f%%           [100.00%%]" % (100.0*tw/86400))
print("  residual non-MM 2-sd  : %.1f%%            [99.5%%]" % (100.0*r2/86400))
print("  touch-order age median: %.0fs             [124s]" % ages[len(ages)//2])
print("  marketable limit ord. : %.1f%% of %d      [14.9%%]"
      % (100.0*n_lim_mkt/n_lim, n_lim))
print("  noise buy fraction    : %.2f%%           [49.68%%]" % (100.0*n_buy/n_all))
print("  MM fills=%d  q=%+.5f BTC" % (mm.n_buy_fills+mm.n_sell_fills, mm.q))
