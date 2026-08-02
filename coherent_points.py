# coherent_points.py: touch width measured at three (disp, mean_lifetime) points

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import diag_bookwidth as D

SIGMA = 0.3721
SPY = 365 * 24 * 3600
T = 86400.0
SEEDS = [0, 1, 2]
LAM = D.DEFAULT_ORDER_RATE
P_MARKET = D.DEFAULT_P_MARKET
TARGET_AGGTRADE_RATE = 0.0451

POINTS = [
    ("(i)   coherent @ locked life", 0.0009, 180.0),
    ("(ii)  log-log nearest",        0.0013, 433.0),
    ("(iii) LOCKED control",         0.0020, 180.0),
]

CONTROL_MEDIAN_BP = 2.7210
CONTROL_MEAN_BP = 3.5656


def coherence_ratio(life, disp):
    """diag_coherence.py's condition: V's drift over one order lifetime"""
    return SIGMA * math.sqrt(life / SPY) / disp


def run(seed, disp, life):
    """One MM-absent day"""
    _path, value_fn, eng, noise, informed = D._make_world(
        seed, T, LAM, P_MARKET, life, disp)

    n_event = n_agg = 0
    n_limit = n_limit_marketable = 0
    spreads_bp, n_resting = [], []

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
        for r in informed.run_until(eng, t, value_fn):
            if getattr(r, "fills", None):
                n_event += 1
                n_agg += len(set(f.price for f in r.fills))

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            spreads_bp.append(1e4 * (ba - bb) / (0.5 * (bb + ba)))
        n_resting.append(sum(1 for o in eng.orders.values()
                             if str(o.agent_id) != "seed"))

    return {
        "spreads_bp": spreads_bp,
        "resting_per_side": statistics.mean(n_resting) / 2.0,
        "ev_per_s": n_event / T,
        "agg_per_s": n_agg / T,
        "f": n_limit_marketable / max(1, n_limit),
        "n_limit": n_limit,
        "two_sided_pct": 100.0 * len(spreads_bp) / T,
    }


def measure(disp, life):
    """Pool seeds days at one (disp, life) point"""
    pooled, rest, ev, agg, f, two_sided = [], [], [], [], [], []
    n_limit = 0
    t0 = time.time()
    for s in SEEDS:
        x = run(s, disp, life)
        pooled += x["spreads_bp"]
        rest.append(x["resting_per_side"])
        ev.append(x["ev_per_s"])
        agg.append(x["agg_per_s"])
        f.append(x["f"])
        two_sided.append(x["two_sided_pct"])
        n_limit += x["n_limit"]
    pooled.sort()
    return {
        "disp": disp, "life": life,
        "ratio": coherence_ratio(life, disp),
        "median_bp": pooled[len(pooled) // 2],
        "mean_bp": statistics.mean(pooled),
        "resting_per_side": statistics.mean(rest),
        "ev_per_s": statistics.mean(ev),
        "agg_per_s": statistics.mean(agg),
        "f": statistics.mean(f),
        "two_sided_pct": statistics.mean(two_sided),
        "n_limit": n_limit,
        "secs": time.time() - t0,
    }


def main():
    print("=" * 104)
    print("Coherent-point measurement; does satisfying the coherence "
          "condition close the book-width gap?")
    print("=" * 104)
    print("MM ABSENT, informed traders present. lam=%.4f p_market=%.3f "
          "(read from noise_traders, not overridden)." % (LAM, P_MARKET))
    print("%d seeds x %.0fs per point. Phase 3 real-market median touch: "
          "0.29bp. Phase 3 trade rate: %.4f aggTrades/s."
          % (len(SEEDS), T, TARGET_AGGTRADE_RATE))
    print("")

    hdr = ("%-30s %7s %9s %9s %8s %8s %9s %9s %7s"
           % ("point", "coh.rat", "median bp", "mean bp", "rest/sd",
              "ev/s", "aggTr/s", "vs .0451", "f"))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for tag, disp, life in POINTS:
        r = measure(disp, life)
        r["tag"] = tag
        rows.append(r)
        print("%-30s %7.3f %9.4f %9.4f %8.1f %8.4f %9.4f %8.2fx %6.1f%%"
              % (tag, r["ratio"], r["median_bp"], r["mean_bp"],
                 r["resting_per_side"], r["ev_per_s"], r["agg_per_s"],
                 r["agg_per_s"] / TARGET_AGGTRADE_RATE, 100.0 * r["f"]))

    print("")
    print("  (disp, life): " + "   ".join(
        "%s=(%.4f, %.0f)" % (r["tag"].split()[0], r["disp"], r["life"])
        for r in rows))
    print("  two-sided%:    " + "   ".join(
        "%s=%.2f" % (r["tag"].split()[0], r["two_sided_pct"]) for r in rows))
    print("  noise limit orders: " + "   ".join(
        "%s=%d" % (r["tag"].split()[0], r["n_limit"]) for r in rows))
    print("  wall clock:   " + "   ".join(
        "%s=%.0fs" % (r["tag"].split()[0], r["secs"]) for r in rows))

    ctl = rows[-1]
    print("")
    print("control check; does the locked point reproduce diag_bookwidth "
          "Section 0?")
    print("  expected median %.4f / mean %.4f bp   measured %.4f / %.4f bp   %s"
          % (CONTROL_MEDIAN_BP, CONTROL_MEAN_BP, ctl["median_bp"],
             ctl["mean_bp"],
             "match" if abs(ctl["median_bp"] - CONTROL_MEDIAN_BP) < 0.005
             else "*** mismatch ***"))

    print("")
    print("extreme-order-statistic model  width = 2*1.2533*disp/N_per_side")
    print("  The best bid is set by the smallest |eta| among resting bids, so "
          "the width is an")
    print("  extreme order statistic, not a mean. Excess above 1.00x is "
          "staleness the model omits.")
    print("  %-30s %9s %11s %11s %10s"
          % ("point", "coh.rat", "predicted", "measured", "meas/pred"))
    for r in sorted(rows, key=lambda z: z["ratio"]):
        pred = 1e4 * 2 * 1.2533 * r["disp"] / r["resting_per_side"]
        print("  %-30s %9.3f %10.4fbp %10.4fbp %9.2fx"
              % (r["tag"], r["ratio"], pred, r["median_bp"],
                 r["median_bp"] / pred))

    print("")
    print("diag_frontier.py supply ceiling; reported in both conventions")
    print("  As written:  f*lam*(1-p_market) <= %.4f/s. The left side counts "
          "order events;" % TARGET_AGGTRADE_RATE)
    print("  %.4f was measured in aggTrades. Multiplying by the measured "
          "agg/ev puts both" % TARGET_AGGTRADE_RATE)
    print("  sides in the same units. diag_frontier.py is NOT edited; this "
          "is a measurement.")
    print("  %-30s %7s %13s %10s %12s %10s"
           % ("point", "agg/ev", "as written", "verdict", "in aggTrades",
              "verdict"))
    for r in rows:
        k = r["agg_per_s"] / r["ev_per_s"]
        c_ev = r["f"] * LAM * (1.0 - P_MARKET)
        c_agg = c_ev * k
        print("  %-30s %7.3f %13.4f %9.2fx %12.4f %9.2fx %s"
              % (r["tag"], k, c_ev, c_ev / TARGET_AGGTRADE_RATE,
                 c_agg, c_agg / TARGET_AGGTRADE_RATE,
                 "OK" if c_agg <= TARGET_AGGTRADE_RATE else "breached"))

    print("")
    print("accounting check; lam*p_market + f*lam*(1-p_market) should "
          "reconcile to ev/s")
    print("  (the residual is informed-trader flow, which the identity omits)")
    for r in rows:
        pred_ev = LAM * P_MARKET + r["f"] * LAM * (1.0 - P_MARKET)
        print("  %-30s %.5f + %.4f = %.4f  vs measured %.4f  (informed %.4f)"
              % (r["tag"], LAM * P_MARKET, r["f"] * LAM * (1.0 - P_MARKET),
                 pred_ev, r["ev_per_s"], r["ev_per_s"] - pred_ev))

    print("")
    print("Reference: diag_frontier.py recorded f ~ 0.126 at disp=0.0020 and "
          "~0.218 at disp=0.0009,")
    print("but those were MM-present at mult=20 / p_market=0.05; a different "
          "regime, not a contradiction.")


if __name__ == "__main__":
    main()
