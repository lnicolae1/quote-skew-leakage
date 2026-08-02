# diag_bookwidth.py: why the residual (non-MM) book is ~4.11 bp wide vs 0.29 bp real (measurement only)

import math
import os
import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import (NoiseTraderFlow, DEFAULT_ORDER_RATE,
                           DEFAULT_P_MARKET, DEFAULT_MEAN_LIFETIME,
                           DEFAULT_DISP)
from informed_traders import InformedTraderFlow, path_value
import sim_run

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "bookwidth_results.txt")

BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0),
           (5.0, 10.0), (10.0, float("inf"))]
BUCKET_NAMES = ["0-0.5bp", "0.5-1bp", "1-2bp", "2-5bp", "5-10bp", "10+bp"]

HIST_BINW = 0.05
HIST_MAX = 400.0

T_PROFILE = 86400.0
T_SENS = 21600.0
PROFILE_SEEDS = [0, 1, 2]

_lines = []


def emit(s=""):
    print(s)
    _lines.append(s)


def bucket_of(bp):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= bp < hi:
            return i
    return len(BUCKETS) - 1


def hist_pct(hist, p):
    """Percentile from a {bin_index: count} histogram"""
    total = sum(hist.values())
    if total == 0:
        return float("nan")
    target = p * total
    c = 0
    for b in sorted(hist):
        c += hist[b]
        if c >= target:
            return (b + 0.5) * HIST_BINW
    return float("nan")


def _make_world(seed, T, lam, p_market, life, disp):
    """Value path, engine with the standard seeded book, noise and informed"""
    n = int(T)
    path = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE, 1.0, n,
                               seed=sim_run.SEED_V_BASE + 1000 * seed)
    vf = path_value(path, 1.0)
    eng = MatchingEngine()
    sim_run._seed_book(eng)
    noise = NoiseTraderFlow(lam=lam, p_market=p_market, mean_lifetime=life,
                            disp=disp, value_fn=vf,
                            reference_price=sim_run.REFERENCE_PRICE,
                            seed=sim_run.SEED_NOISE_BASE + 1000 * seed)
    informed = InformedTraderFlow(seed=sim_run.SEED_INF_BASE + 1000 * seed)
    return path, vf, eng, noise, informed


def run_profile(seed, T=T_PROFILE, lam=DEFAULT_ORDER_RATE,
                p_market=DEFAULT_P_MARKET, life=DEFAULT_MEAN_LIFETIME,
                disp=DEFAULT_DISP):
    path, vf, eng, noise, informed = _make_world(seed, T, lam, p_market,
                                                 life, disp)

    clock = [0.0]

    cancelled = set()
    _orig_cancel = eng.cancel_order

    def _cancel(oid):
        ok = _orig_cancel(oid)
        if ok:
            cancelled.add(oid)
        return ok
    eng.cancel_order = _cancel

    place_bp = {"noise": [], "informed": []}
    place_bp_v = {"noise": [], "informed": []}
    n_no_mid = [0]
    fate = {i: {"immediate": 0, "fill": 0, "cancel": 0, "resting": 0}
            for i in range(len(BUCKETS))}
    fate_aggressive = {"immediate": 0, "fill": 0, "cancel": 0, "resting": 0}
    tracked = {}

    _orig_add = eng.add_limit_order

    def _add(side, price, size, agent_id):
        mid_before = eng.mid()
        res = _orig_add(side, price, size, agent_id)
        who = ("noise" if str(agent_id).startswith("noise")
               else "informed" if str(agent_id).startswith("informed")
               else None)
        if who is None:
            return res
        v = vf(clock[0])
        if mid_before is None:
            key = None
            n_no_mid[0] += 1
        else:
            d = (mid_before - price) if side == "buy" else (price - mid_before)
            bp = 1e4 * d / mid_before
            place_bp[who].append(bp)
            key = bucket_of(bp) if bp >= 0 else "aggressive"
        dv = (v - price) if side == "buy" else (price - v)
        place_bp_v[who].append(1e4 * dv / v)

        if res.fills:
            if key == "aggressive":
                fate_aggressive["immediate"] += 1
            elif key is not None:
                fate[key]["immediate"] += 1
        elif res.resting_size > 0 and key is not None:
            tracked[res.order_id] = [key, res.resting_size]
        return res
    eng.add_limit_order = _add

    n_samples = 0
    spreads_bp = []
    qty = {"buy": [0.0] * len(BUCKETS), "sell": [0.0] * len(BUCKETS)}
    cnt = {"buy": [0] * len(BUCKETS), "sell": [0] * len(BUCKETS)}
    lvl = {"buy": [0] * len(BUCKETS), "sell": [0] * len(BUCKETS)}
    seed_qty = {"buy": [0.0] * len(BUCKETS), "sell": [0.0] * len(BUCKETS)}
    rest_hist = {}
    n_resting_side = {"buy": 0, "sell": 0}
    n_one_sided = 0

    t = 0.0
    while t < T:
        t += 1.0
        clock[0] = t
        noise.run_until(eng, t)
        informed.run_until(eng, t, vf)

        for oid in list(tracked.keys()):
            o = eng.orders.get(oid)
            key = tracked[oid][0]
            slot = fate_aggressive if key == "aggressive" else fate[key]
            if o is None:
                slot["cancel" if oid in cancelled else "fill"] += 1
                del tracked[oid]
            elif o.size < tracked[oid][1] - 1e-15:
                slot["fill"] += 1
                del tracked[oid]

        mid = eng.mid()
        if mid is None:
            n_one_sided += 1
            continue
        n_samples += 1
        bb, ba = eng.best_bid(), eng.best_ask()
        spreads_bp.append(1e4 * (ba - bb) / mid)

        prices_seen = {"buy": [set() for _ in BUCKETS],
                       "sell": [set() for _ in BUCKETS]}
        for o in eng.orders.values():
            d = (mid - o.price) if o.side == "buy" else (o.price - mid)
            bp = 1e4 * d / mid
            b = bucket_of(bp if bp >= 0 else 0.0)
            if str(o.agent_id) == "seed":
                seed_qty[o.side][b] += o.size
                continue
            qty[o.side][b] += o.size
            cnt[o.side][b] += 1
            prices_seen[o.side][b].add(o.price)
            n_resting_side[o.side] += 1
            hb = int(min(max(bp, 0.0), HIST_MAX) / HIST_BINW)
            rest_hist[hb] = rest_hist.get(hb, 0) + 1
        for s in ("buy", "sell"):
            for b in range(len(BUCKETS)):
                lvl[s][b] += len(prices_seen[s][b])

    for oid, (key, _sz) in tracked.items():
        slot = fate_aggressive if key == "aggressive" else fate[key]
        slot["resting"] += 1

    return {
        "seed": seed, "n_samples": n_samples, "n_one_sided": n_one_sided,
        "spreads_bp": spreads_bp, "place_bp": place_bp,
        "place_bp_v": place_bp_v, "fate": fate,
        "fate_aggressive": fate_aggressive, "qty": qty, "cnt": cnt,
        "lvl": lvl, "seed_qty": seed_qty, "rest_hist": rest_hist,
        "n_resting_side": n_resting_side, "n_no_mid": n_no_mid[0],
        "n_cancelled": noise.n_cancelled, "n_scheduled": noise.n_scheduled,
        "V_end": vf(T - 1.0),
    }


def run_width(seed, T, lam, p_market, life, disp):
    """Touch width and resting count only: no per-order bucketing, so this"""
    path, vf, eng, noise, informed = _make_world(seed, T, lam, p_market,
                                                 life, disp)
    n_trades = 0
    spreads_bp = []
    n_rest = []
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.fills:
                n_trades += 1
        for r in informed.run_until(eng, t, vf):
            if getattr(r, "fills", None):
                n_trades += 1
        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            m = 0.5 * (bb + ba)
            spreads_bp.append(1e4 * (ba - bb) / m)
        n_rest.append(sum(1 for o in eng.orders.values()
                          if str(o.agent_id) != "seed"))
    spreads_bp.sort()
    return {
        "median_bp": spreads_bp[len(spreads_bp) // 2] if spreads_bp else float("nan"),
        "mean_bp": statistics.mean(spreads_bp) if spreads_bp else float("nan"),
        "two_sided_pct": 100.0 * len(spreads_bp) / T,
        "mean_resting": statistics.mean(n_rest),
        "trade_rate": n_trades / T,
    }


def loglog_slope(xs, ys):
    """Elasticity d ln y / d ln x by least squares"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx if sxx > 0 else float("nan")


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    import time
    t0 = time.time()

    emit("=" * 100)
    emit("Book-width diagnosis; why is the residual book ~4bp wide?")
    emit("=" * 100)
    emit("MM ABSENT. Informed traders present (same setup measure_k.py used).")
    emit("Locked config: lam=%.4f  p_market=%.3f  mean_lifetime=%.0fs  disp=%.4f"
         % (DEFAULT_ORDER_RATE, DEFAULT_P_MARKET, DEFAULT_MEAN_LIFETIME,
            DEFAULT_DISP))
    emit("Profile: %d seeds x %.0fs. Sensitivity: 1 seed x %.0fs per cell."
         % (len(PROFILE_SEEDS), T_PROFILE, T_SENS))
    emit("")

    runs = []
    for s in PROFILE_SEEDS:
        r = run_profile(s)
        runs.append(r)
        sp = sorted(r["spreads_bp"])
        emit("  seed %d: median spread %.4fbp  two-sided %.2f%%  "
             "mean resting %.1f/side  (%.0fs)"
             % (s, sp[len(sp) // 2],
                100.0 * r["n_samples"] / (r["n_samples"] + r["n_one_sided"]),
                0.5 * (r["n_resting_side"]["buy"] + r["n_resting_side"]["sell"])
                / max(1, r["n_samples"]), time.time() - t0))
    emit("")

    all_sp = sorted(x for r in runs for x in r["spreads_bp"])
    N = sum(r["n_samples"] for r in runs)
    n_side = 0.5 * sum(r["n_resting_side"]["buy"] + r["n_resting_side"]["sell"]
                       for r in runs) / N

    emit("=" * 100)
    emit("Section 0; touch width (pooled, %d two-sided samples)" % N)
    emit("=" * 100)
    emit("  median %.4f bp    mean %.4f bp" % (all_sp[len(all_sp) // 2],
                                               statistics.mean(all_sp)))
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        emit("    p%-3d %8.4f bp" % (100 * p, all_sp[int(p * len(all_sp))]))
    emit("  mean resting non-seed orders per side: %.1f" % n_side)
    pred = 2.0 * 1.2533 * DEFAULT_DISP / n_side * 1e4
    emit("  analytic prediction 2*1.2533*disp/N_per_side = %.4f bp" % pred)
    emit("    measured / predicted = %.2fx  (excess above 1.0 = staleness)"
         % (all_sp[len(all_sp) // 2] / pred))
    emit("")

    emit("=" * 100)
    emit("Section 1; placement vs resting distance")
    emit("=" * 100)
    emit("  Signed distance: positive = passive (priced behind the mid),")
    emit("  negative = aggressive (priced through the mid, i.e. marketable).")
    emit("")
    for who in ("noise", "informed"):
        v = [x for r in runs for x in r["place_bp"][who]]
        vv = [x for r in runs for x in r["place_bp_v"][who]]
        if not v:
            emit("  %-9s no placements" % who)
            continue
        neg = sum(1 for x in v if x < 0)
        emit("  %s limit orders placed: %d" % (who.upper(), len(v)))
        emit("    vs mid at submission:  median %+8.4f bp   mean %+8.4f bp"
             % (pct(v, 0.5), statistics.mean(v)))
        emit("      p05 %+8.4f  p25 %+8.4f  p75 %+8.4f  p95 %+8.4f"
             % (pct(v, 0.05), pct(v, 0.25), pct(v, 0.75), pct(v, 0.95)))
        emit("      aggressive (negative, marketable on arrival): %d (%.2f%%)"
             % (neg, 100.0 * neg / len(v)))
        emit("    vs V at submission:    median %+8.4f bp   mean %+8.4f bp"
             % (pct(vv, 0.5), statistics.mean(vv)))
    emit("")
    hist = {}
    for r in runs:
        for k, c in r["rest_hist"].items():
            hist[k] = hist.get(k, 0) + c
    emit("  resting distance from mid, time-averaged (order-seconds weighted,")
    emit("  non-seed orders only, %d order-seconds):" % sum(hist.values()))
    emit("    median %8.4f bp" % hist_pct(hist, 0.5))
    for p in (0.05, 0.25, 0.75, 0.95):
        emit("      p%-3d %8.4f bp" % (100 * p, hist_pct(hist, p)))
    emit("")

    emit("=" * 100)
    emit("Section 2; time-averaged depth profile by distance from mid")
    emit("=" * 100)
    emit("  Reminder: no order can sit closer to the mid than half the spread,")
    emit("  so empty near buckets are implied by the spread, not independent.")
    emit("")
    emit("  %-10s | %12s %10s %9s | %12s %10s %9s"
         % ("bucket", "bid qty BTC", "bid orders", "bid lvls",
            "ask qty BTC", "ask orders", "ask lvls"))
    emit("  " + "-" * 88)
    for b, name in enumerate(BUCKET_NAMES):
        bq = sum(r["qty"]["buy"][b] for r in runs) / N
        aq = sum(r["qty"]["sell"][b] for r in runs) / N
        bc = sum(r["cnt"]["buy"][b] for r in runs) / N
        ac = sum(r["cnt"]["sell"][b] for r in runs) / N
        bl = sum(r["lvl"]["buy"][b] for r in runs) / N
        al = sum(r["lvl"]["sell"][b] for r in runs) / N
        emit("  %-10s | %12.5f %10.2f %9.2f | %12.5f %10.2f %9.2f"
             % (name, bq, bc, bl, aq, ac, al))
    tb = sum(sum(r["qty"]["buy"]) for r in runs) / N
    ta = sum(sum(r["qty"]["sell"]) for r in runs) / N
    emit("  " + "-" * 88)
    emit("  %-10s | %12.5f %10.2f %9s | %12.5f %10.2f %9s"
         % ("total", tb, sum(sum(r["cnt"]["buy"]) for r in runs) / N, "",
            ta, sum(sum(r["cnt"]["sell"]) for r in runs) / N, ""))
    sb = sum(sum(r["seed_qty"]["buy"]) for r in runs) / N
    sa = sum(sum(r["seed_qty"]["sell"]) for r in runs) / N
    emit("  seeded-book orders excluded above: %.5f BTC bid / %.5f BTC ask"
         % (sb, sa))
    emit("  (the 10 _seed_book orders never expire; they have no B2 lifetime")
    emit("   -- so they are reported separately rather than mixed in.)")
    emit("")

    emit("=" * 100)
    emit("Section 3; fate by placement distance")
    emit("=" * 100)
    emit("  'immediate' = traded on arrival (never rested).")
    emit("  'fill'/'cancel' = rested, then first-passage fill / B2 expiry.")
    emit("")
    emit("  %-12s %8s | %10s %10s %10s %10s"
         % ("placed at", "n", "immediate", "filled", "cancelled", "resting"))
    emit("  " + "-" * 74)
    agg = {"immediate": 0, "fill": 0, "cancel": 0, "resting": 0}
    for r in runs:
        for k in agg:
            agg[k] += r["fate_aggressive"][k]
    tot = sum(agg.values())
    if tot:
        emit("  %-12s %8d | %9.1f%% %9.1f%% %9.1f%% %9.1f%%"
             % ("aggressive", tot, 100.0 * agg["immediate"] / tot,
                100.0 * agg["fill"] / tot, 100.0 * agg["cancel"] / tot,
                100.0 * agg["resting"] / tot))
    for b, name in enumerate(BUCKET_NAMES):
        a = {k: sum(r["fate"][b][k] for r in runs)
             for k in ("immediate", "fill", "cancel", "resting")}
        tot = sum(a.values())
        if tot == 0:
            emit("  %-12s %8d | %s" % (name, 0, "no orders ever placed here"))
            continue
        emit("  %-12s %8d | %9.1f%% %9.1f%% %9.1f%% %9.1f%%"
             % (name, tot, 100.0 * a["immediate"] / tot, 100.0 * a["fill"] / tot,
                100.0 * a["cancel"] / tot, 100.0 * a["resting"] / tot))
    emit("")

    emit("=" * 100)
    emit("Section 4; one-at-A-time sensitivity of the touch width")
    emit("=" * 100)
    emit("  Each row varies one parameter; the others stay at the locked config.")
    emit("  trade/s is shown because a config that fixes the width by raising")
    emit("  supply may break the Phase 3 trade-rate calibration (target 0.0451).")
    emit("")
    grids = [
        ("disp", [0.0005, 0.0010, 0.0020, 0.0040, 0.0080]),
        ("mean_lifetime", [45.0, 90.0, 180.0, 360.0, 720.0]),
        ("lam", [0.25 * DEFAULT_ORDER_RATE, 0.5 * DEFAULT_ORDER_RATE,
                 DEFAULT_ORDER_RATE, 2 * DEFAULT_ORDER_RATE,
                 4 * DEFAULT_ORDER_RATE]),
        ("p_market", [0.005, 0.01, 0.02, 0.04, 0.08]),
    ]
    KWNAME = {"disp": "disp", "mean_lifetime": "life", "lam": "lam",
              "p_market": "p_market"}
    LOCKED = {"disp": DEFAULT_DISP, "mean_lifetime": DEFAULT_MEAN_LIFETIME,
              "lam": DEFAULT_ORDER_RATE, "p_market": DEFAULT_P_MARKET}
    elasticities = {}
    for pname, vals in grids:
        emit("  %s:" % pname)
        emit("    %-12s %11s %11s %10s %12s %10s"
             % ("value", "median bp", "mean bp", "2-sided%", "resting/side",
                "trade/s"))
        widths = []
        for v in vals:
            kw = dict(lam=DEFAULT_ORDER_RATE, p_market=DEFAULT_P_MARKET,
                      life=DEFAULT_MEAN_LIFETIME, disp=DEFAULT_DISP)
            kw[KWNAME[pname]] = v
            x = run_width(0, T_SENS, **kw)
            widths.append(x["median_bp"])
            marker = ("  <== locked" if abs(v - LOCKED[pname]) < 1e-12 else "")
            emit("    %-12.5f %11.4f %11.4f %9.2f%% %12.1f %10.4f%s"
                 % (v, x["median_bp"], x["mean_bp"], x["two_sided_pct"],
                    x["mean_resting"] / 2.0, x["trade_rate"], marker))
        e = loglog_slope(vals, widths)
        elasticities[pname] = e
        emit("    elasticity d ln(width) / d ln(%s) = %+.3f" % (pname, e))
        emit("")

    emit("  ranked by |elasticity|; which single knob moves the width most:")
    for pname, e in sorted(elasticities.items(),
                           key=lambda kv: -abs(kv[1])):
        emit("    %-14s %+.3f" % (pname, e))
    emit("")
    emit("  total wall clock %.0fs" % (time.time() - t0))

    with open(OUT, "w") as f:
        f.write("\n".join(_lines) + "\n")
    print("\nwritten to %s" % OUT)


if __name__ == "__main__":
    main()
