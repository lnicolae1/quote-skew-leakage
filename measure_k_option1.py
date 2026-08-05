# measure_k_option1.py: k from the maker's own quotes and fills (internally calibrated), clipped working point
# - controlled quoter: one order per side at exactly delta from mid, cancelled and reposted every second
# - no AS formula, no skew, no gamma; delta set on a log grid $0.25-$30
# - fill = first passage (order gone or reduced); fill rate lambda(delta) = fills / exposure
# - fit: ln lambda = ln A - k * delta, weighted least squares (measure_k.py's wls_fit)

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import DEFAULT_QUOTE_SIZE, DEFAULT_SIGMA
import sim_run

T = 86400.0
SEEDS = list(range(10))
PROBE_ID = "PROBE"
TICK = 0.01

# clipped working point
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055

# log-spaced $0.25-$30
N_DELTA = 12
DELTAS = [round(0.25 * (30.0 / 0.25) ** (i / (N_DELTA - 1.0)), 2)
          for i in range(N_DELTA)]

OLD_K = 0.06747
CLIPPED_K_LO, CLIPPED_K_HI = 0.03997, 0.09026
TOUCH_BP, TOUCH_USD = 0.3110, 0.3110 * 62000.0 / 1e4
MID_1S_SD = DEFAULT_SIGMA / math.sqrt(365 * 24 * 3600) * 62000.0


def run_one(seed, delta):
    """One day of the controlled quoter at +/-delta. Returns (fills, exposure, by side)."""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    n_fill_bid = n_fill_ask = 0
    exposure = 0.0
    live = {}

    t = 0.0
    while t < T:
        t += 1.0

        # 1) score last second's quotes: gone or reduced = filled (first passage)
        for side, (oid, size0) in list(live.items()):
            o = eng.orders.get(oid)
            if o is None:
                if side == "buy":
                    n_fill_bid += 1
                else:
                    n_fill_ask += 1
            else:
                if o.size < size0 - 1e-15:
                    if side == "buy":
                        n_fill_bid += 1
                    else:
                        n_fill_ask += 1
                eng.cancel_order(oid)
        live.clear()

        # 2) mid with own quotes off the book
        mid = eng.mid()

        # 3) one order per side at delta from mid
        if mid is not None:
            bid_px = round(round((mid - delta) / TICK) * TICK, 8)
            ask_px = round(round((mid + delta) / TICK) * TICK, 8)
            for side, px in (("buy", bid_px), ("sell", ask_px)):
                res = eng.add_limit_order(side, px, DEFAULT_QUOTE_SIZE,
                                          PROBE_ID)
                if res.fills:
                    # crossed on arrival: counted as a fill at this distance
                    if side == "buy":
                        n_fill_bid += 1
                    else:
                        n_fill_ask += 1
                    exposure += 1.0
                elif res.resting_size > 0:
                    live[side] = (res.order_id, res.resting_size)
                    exposure += 1.0

        # 4) advance one second
        for r in noise.run_until(eng, t):
            pass
        for r in informed.run_until(eng, t, vf):
            pass

    # score what is still live
    for side, (oid, size0) in live.items():
        o = eng.orders.get(oid)
        if o is not None and o.size < size0 - 1e-15:
            if side == "buy":
                n_fill_bid += 1
            else:
                n_fill_ask += 1

    return n_fill_bid, n_fill_ask, exposure


def wls_fit(xs, ys, ws):
    """measure_k.py's weighted least squares, unchanged"""
    W = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / W
    my = sum(w * y for w, y in zip(ws, ys)) / W
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    syy = sum(w * (y - my) ** 2 for w, y in zip(ws, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = len(xs) - 2
    se_b = math.sqrt((resid / dof) / sxx) if dof > 0 else float("nan")
    r2 = 1.0 - resid / syy if syy > 0 else float("nan")
    return a, b, se_b, r2


def main():
    t0 = time.time()
    print("=" * 100)
    print("Measure k: Option 1 (the MM's own quotes, its own fills)")
    print("=" * 100)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping." % (LAM, P_MARKET, LIFE, DISP))
    print("Controlled quoter: one order per side at delta from mid, "
          "size %.2f BTC," % DEFAULT_QUOTE_SIZE)
    print("cancelled and reposted every second (market_maker.py "
          "cadence). No AS formula,")
    print("no skew, no gamma; delta set by the script.")
    print("%d seeds x %.0fs per delta.  k is internally calibrated." % (len(SEEDS), T))
    print("")
    print("Scale check: mid's 1-second sd = $%.2f; this book's touch = $%.2f."
          % (MID_1S_SD, TOUCH_USD))
    print("Quote distance and 1 s price move are comparable near the "
          "touch; check R2 first.")
    print("")

    print("%10s %14s %10s %10s %14s %12s"
          % ("delta ($)", "exposure(s)", "fills", "bid/ask", "lambda(/s)",
             "ln lambda"))
    print("-" * 100)

    rows = []
    for d in DELTAS:
        fb = fa = 0
        expo = 0.0
        for s in SEEDS:
            b, a, e = run_one(s, d)
            fb += b; fa += a; expo += e
        nf = fb + fa
        lam = nf / expo if expo > 0 else 0.0
        star = "  (n<20, excluded from fit)" if nf < 20 else ""
        print("%10.2f %14.0f %10d %10s %14.4e %12s%s"
              % (d, expo, nf, "%d/%d" % (fb, fa), lam,
                 ("%.4f" % math.log(lam)) if lam > 0 else "  -inf", star),
              flush=True)
        if nf >= 20 and lam > 0:
            rows.append((d, math.log(lam), nf))

    print("")
    print("=" * 100)
    print("Exponential fit  ln lambda = ln A - k * delta   (weighted by fill "
          "count)")
    print("=" * 100)
    windows = [("full swept range $0.25-30", lambda r: True),
               ("near-touch $0.25-5", lambda r: r[0] <= 5.0),
               ("near-touch $0.25-10", lambda r: r[0] <= 10.0),
               ("far field $5-30", lambda r: r[0] >= 5.0)]
    fits = {}
    for label, sel in windows:
        sub = [r for r in rows if sel(r)]
        if len(sub) < 3:
            print("%-30s too few points (n=%d)" % (label, len(sub)))
            continue
        a, b, se, r2 = wls_fit([r[0] for r in sub], [r[1] for r in sub],
                               [r[2] for r in sub])
        k = -b
        fits[label] = (k, se, r2)
        print("%-30s  k = %.5f +/- %.5f   R2 = %.4f   npoints=%d  nfills=%d"
              % (label, k, se, r2, len(sub), sum(r[2] for r in sub)))
        if k > 0:
            print("%-30s  1/k = $%.2f     2/k = $%.2f  = %.4f bp"
                  % ("", 1.0 / k, 2.0 / k, 1e4 * (2.0 / k) / 62000.0))
        else:
            print("%-30s  k <= 0: no decay detected on this window"
                  % "")

    print("")
    print("=" * 100)
    print("Comparison")
    print("=" * 100)
    print("  %-42s %10s %12s %10s" % ("estimator / window", "k", "2/k ($)",
                                      "x touch"))
    print("  %-42s %10.5f %12.2f %9.1fx"
          % ("measure_k.py (old book, MM-absent, 0-40)", OLD_K, 2 / OLD_K,
             (2 / OLD_K) / TOUCH_USD))
    for nm, kk in (("measure_k_clipped.py bracket, low (full)", CLIPPED_K_LO),
                   ("measure_k_clipped.py bracket, high (0-20)", CLIPPED_K_HI)):
        print("  %-42s %10.5f %12.2f %9.1fx"
              % (nm, kk, 2 / kk, (2 / kk) / TOUCH_USD))
    for label, (k, se, r2) in fits.items():
        if k > 0:
            print("  %-42s %10.5f %12.2f %9.1fx   R2=%.4f"
                  % ("Option 1: " + label, k, 2 / k, (2 / k) / TOUCH_USD, r2))
        else:
            print("  %-42s %10.5f %12s %9s   R2=%.4f"
                  % ("Option 1: " + label, k, "n/a", "n/a", r2))
    print("")
    print("  this book's touch: %.4f bp = $%.2f" % (TOUCH_BP, TOUCH_USD))
    print("\ntotal wall clock %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
