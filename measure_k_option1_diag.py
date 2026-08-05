# measure_k_option1_diag.py: two checks on measure_k_option1's k: bid/ask asymmetry and fit stability

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from informed_traders import InformedTraderFlow, path_value
from clipped_placement import ClippedNoiseTraderFlow
from market_maker import DEFAULT_QUOTE_SIZE
import sim_run

T = 86400.0
SEEDS = list(range(10))
PROBE_ID = "PROBE"
TICK = 0.01
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
TEST_DELTAS = [0.25, 2.20, 8.13]
RESULTS_FILE = os.path.join(_HERE, "measure_k_option1_results.txt")


def build_world(seed, dedrift):
    n = int(T)
    path = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE, 1.0, n,
                               seed=sim_run.SEED_V_BASE + 1000 * seed)
    log_drift = math.log(path[-1] / path[0])
    if dedrift:
        alpha = log_drift / (len(path) - 1)
        path = [p * math.exp(-alpha * i) for i, p in enumerate(path)]
    vf = path_value(path, 1.0)
    eng = MatchingEngine()
    sim_run._seed_book(eng)
    noise = ClippedNoiseTraderFlow(
        clip_mode="join", lam=LAM, p_market=P_MARKET, mean_lifetime=LIFE,
        disp=DISP, value_fn=vf, reference_price=sim_run.REFERENCE_PRICE,
        seed=sim_run.SEED_NOISE_BASE + 1000 * seed)
    informed = InformedTraderFlow(seed=sim_run.SEED_INF_BASE + 1000 * seed)
    return path, vf, eng, noise, informed, log_drift


def probe(seed, delta, dedrift):
    """measure_k_option1.run_one, unchanged except for the world builder"""
    _p, vf, eng, noise, informed, log_drift = build_world(seed, dedrift)
    n_bid = n_ask = 0
    exposure = 0.0
    live = {}
    t = 0.0
    while t < T:
        t += 1.0
        for side, (oid, size0) in list(live.items()):
            o = eng.orders.get(oid)
            if o is None:
                if side == "buy":
                    n_bid += 1
                else:
                    n_ask += 1
            else:
                if o.size < size0 - 1e-15:
                    if side == "buy":
                        n_bid += 1
                    else:
                        n_ask += 1
                eng.cancel_order(oid)
        live.clear()
        mid = eng.mid()
        if mid is not None:
            for side, px in (("buy", round(round((mid - delta) / TICK) * TICK, 8)),
                             ("sell", round(round((mid + delta) / TICK) * TICK, 8))):
                res = eng.add_limit_order(side, px, DEFAULT_QUOTE_SIZE, PROBE_ID)
                if res.fills:
                    if side == "buy":
                        n_bid += 1
                    else:
                        n_ask += 1
                    exposure += 1.0
                elif res.resting_size > 0:
                    live[side] = (res.order_id, res.resting_size)
                    exposure += 1.0
        for r in noise.run_until(eng, t):
            pass
        for r in informed.run_until(eng, t, vf):
            pass
    for side, (oid, size0) in live.items():
        o = eng.orders.get(oid)
        if o is not None and o.size < size0 - 1e-15:
            if side == "buy":
                n_bid += 1
            else:
                n_ask += 1
    return n_bid, n_ask, exposure, log_drift


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan")
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    return r, t


def wls(X, y, w):
    """Weighted least squares for a design matrix X (list of rows, each row a"""
    p = len(X[0])
    A = [[sum(w[i] * X[i][a] * X[i][b] for i in range(len(X)))
          for b in range(p)] for a in range(p)]
    rhs = [sum(w[i] * X[i][a] * y[i] for i in range(len(X))) for a in range(p)]
    M = [A[r][:] + [rhs[r]] for r in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(p):
            if r == c or M[c][c] == 0:
                continue
            f = M[r][c] / M[c][c]
            for cc in range(c, p + 1):
                M[r][cc] -= f * M[c][cc]
    beta = [M[r][p] / M[r][r] if M[r][r] else 0.0 for r in range(p)]
    W = sum(w)
    my = sum(w[i] * y[i] for i in range(len(y))) / W
    ss_res = sum(w[i] * (y[i] - sum(beta[a] * X[i][a] for a in range(p))) ** 2
                 for i in range(len(y)))
    ss_tot = sum(w[i] * (y[i] - my) ** 2 for i in range(len(y)))
    return beta, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def read_curve():
    """Parse the 12-point curve out of measure_k_option1_results.txt"""
    rows = []
    with open(RESULTS_FILE) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 6 and "/" in p[3]:
                try:
                    d, nf, lam = float(p[0]), int(p[2]), float(p[4])
                except ValueError:
                    continue
                if lam > 0:
                    rows.append((d, math.log(lam), nf))
    return rows


def main():
    t0 = time.time()
    print("=" * 112)
    print("Check 1; is the bid/ASK asymmetry explained by value-path drift?")
    print("=" * 112)
    print("Per-seed drift over one day, and that seed's own ask/bid fill "
          "ratio at %d deltas." % len(TEST_DELTAS))
    print("generate_value_path is driftless in log space, so the "
          "deterministic drift is only")
    print("+0.019%/day. Any real asymmetry must come from realized per-seed "
          "drift (daily sd ~$1,200).")
    print("")

    base = {}
    print("  %-6s %12s %12s %10s %10s %10s %10s"
          % ("seed", "drift $", "drift %", "bid fills", "ask fills",
             "ask/bid", "ln(a/b)"))
    print("  " + "-" * 76)
    drifts, ratios = [], []
    for s in SEEDS:
        tb = ta = 0
        ld = None
        for d in TEST_DELTAS:
            b, a, e, ld = probe(s, d, dedrift=False)
            tb += b; ta += a
        base[s] = (tb, ta)
        dollars = sim_run.REFERENCE_PRICE * (math.exp(ld) - 1.0)
        ratio = ta / tb if tb else float("nan")
        drifts.append(ld)
        ratios.append(math.log(ratio))
        print("  %-6d %12.2f %12.4f %10d %10d %10.4f %10.4f"
              % (s, dollars, 100.0 * (math.exp(ld) - 1.0), tb, ta, ratio,
                 math.log(ratio)), flush=True)

    mean_drift = statistics.mean(drifts)
    print("")
    print("  mean log-drift over the %d seeds: %+.6f  (= %+.3f%% = $%+.2f)"
          % (len(SEEDS), mean_drift, 100.0 * (math.exp(mean_drift) - 1),
             sim_run.REFERENCE_PRICE * (math.exp(mean_drift) - 1)))
    print("  sd of log-drift across seeds     : %.6f" % statistics.stdev(drifts))
    tb_all = sum(base[s][0] for s in SEEDS)
    ta_all = sum(base[s][1] for s in SEEDS)
    print("  pooled bid %d / ask %d -> ask/bid = %.4f"
          % (tb_all, ta_all, ta_all / tb_all))
    r, tstat = pearson(drifts, ratios)
    print("")
    print("  correlation seed log-drift vs seed ln(ask/bid): r = %+.4f  "
          "(t = %+.2f, n = %d)" % (r, tstat, len(SEEDS)))
    print("  Expected under the drift explanation: strongly positive.")

    print("")
    print("  De-drift control; same seeds, net drift removed, volatility "
          "untouched")
    print("  %-6s %10s %10s %10s   %10s %10s %10s"
          % ("delta", "bid(drift)", "ask(drift)", "a/b", "bid(dd)", "ask(dd)",
             "a/b"))
    print("  " + "-" * 76)
    for d in TEST_DELTAS:
        b0 = a0 = b1 = a1 = 0
        for s in SEEDS:
            b, a, _e, _ld = probe(s, d, dedrift=False)
            b0 += b; a0 += a
            b, a, _e, _ld = probe(s, d, dedrift=True)
            b1 += b; a1 += a
        print("  %-6.2f %10d %10d %10.4f   %10d %10d %10.4f"
              % (d, b0, a0, a0 / b0 if b0 else float("nan"),
                 b1, a1, a1 / b1 if b1 else float("nan")), flush=True)
    print("")
    print("  If drift is the cause, the right-hand a/b column should sit at "
          "~1.00.")

    print("")
    print("=" * 112)
    print("Check 2; what shape is the fill-intensity curve?")
    print("=" * 112)
    rows = read_curve()
    print("  %d points read from measure_k_option1_results.txt" % len(rows))
    if len(rows) < 4:
        print("  not enough points parsed; aborting check 2")
        return
    ds = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    ws = [float(r[2]) for r in rows]
    print("  All three forms predict ln lambda with the same fill-count "
          "weights, so R2 is comparable.")
    print("")

    b_exp, r2_exp = wls([[1.0, d] for d in ds], ys, ws)
    b_qua, r2_qua = wls([[1.0, d, d * d] for d in ds], ys, ws)
    b_pow, r2_pow = wls([[1.0, math.log(d)] for d in ds], ys, ws)

    print("  %-38s %10s %12s" % ("form", "R2", "params"))
    print("  " + "-" * 66)
    print("  %-38s %10.5f   ln A=%.4f  k=%.5f"
          % ("exponential  lam = A*exp(-k*d)  [as]", r2_exp, b_exp[0],
             -b_exp[1]))
    print("  %-38s %10.5f   a=%.4f b=%.5f c=%.6f"
          % ("quadratic    ln lam = a+b*d+c*d^2", r2_qua, b_qua[0], b_qua[1],
             b_qua[2]))
    print("  %-38s %10.5f   ln A=%.4f  b=%.5f"
          % ("power law    lam = A*d^-b", r2_pow, b_pow[0], -b_pow[1]))
    print("")
    print("  residual variance left unexplained (1-R2):  exponential %.5f   "
          "quadratic %.5f   power %.5f"
          % (1 - r2_exp, 1 - r2_qua, 1 - r2_pow))
    if b_qua[2] > 0:
        print("  quadratic c = %+.6f > 0 -> log-hazard is convex: decay slows "
              "with distance," % b_qua[2])
        print("  i.e. the tail is fatter than a single exponential. The local "
              "decay rate is")
        print("  -(b + 2c*d): %.5f at $0.25, %.5f at $5, %.5f at $30."
              % (-(b_qua[1] + 2 * b_qua[2] * 0.25),
                 -(b_qua[1] + 2 * b_qua[2] * 5.0),
                 -(b_qua[1] + 2 * b_qua[2] * 30.0)))
    print("")
    print("  Note on the power law: lambda = A*d^-b diverges as d -> 0, so it "
          "cannot be right")
    print("  at the touch however well it fits the swept range. Reported for "
          "comparison only.")
    print("")
    print("  total wall clock %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
