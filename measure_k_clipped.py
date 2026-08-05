# measure_k_clipped.py: re-measures k at the clipped working point

import math
import statistics
import time

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from informed_traders import InformedTraderFlow, path_value
from clipped_placement import ClippedNoiseTraderFlow
import sim_run

DURATION = 259200.0
SEEDS = [0, 1, 2]
BUCKET_W = 1.0

LAM = 1.804
P_MARKET = 0.0126
LIFE = 720.0
DISP = 0.0055

OLD_K = 0.06747
THIS_BOOK_TOUCH_BP = 0.3110
THIS_BOOK_TOUCH_USD = 0.3110 * 62000.0 / 1e4


def run_one(seed):
    n = int(DURATION)
    sv = sim_run.SEED_V_BASE + 1000 * seed
    sn = sim_run.SEED_NOISE_BASE + 1000 * seed
    si = sim_run.SEED_INF_BASE + 1000 * seed

    path = generate_value_path(sim_run.SIGMA, 62000.0, 1.0, n, seed=sv)
    vf = path_value(path, 1.0)
    eng = MatchingEngine()
    sim_run._seed_book(eng)

    cancelled = set()
    _orig_cancel = eng.cancel_order

    def _cancel(oid):
        ok = _orig_cancel(oid)
        if ok:
            cancelled.add(oid)
        return ok
    eng.cancel_order = _cancel

    noise = ClippedNoiseTraderFlow(clip_mode="join", lam=LAM,
                                   p_market=P_MARKET, mean_lifetime=LIFE,
                                   disp=DISP, value_fn=vf,
                                   reference_price=62000.0, seed=sn)
    informed = InformedTraderFlow(seed=si)

    exposure, fills, dsum = {}, {}, {}
    tracked = {}
    n_censored_cancel = 0
    resting_counts = []

    t = 0.0
    while t < DURATION:
        t += 1.0
        mid = eng.mid()

        bucket_at_start = {}
        if mid is not None:
            orders = eng.orders
            for oid, rec in tracked.items():
                o = orders.get(oid)
                if o is None:
                    continue
                d = (mid - o.price) if o.side == "buy" else (o.price - mid)
                b = int(math.floor(d / BUCKET_W))
                bucket_at_start[oid] = b
                exposure[b] = exposure.get(b, 0.0) + 1.0
                dsum[b] = dsum.get(b, 0.0) + d
            resting_counts.append(len(tracked))

        for r in noise.run_until(eng, t):
            pass
        for r in informed.run_until(eng, t, vf):
            pass

        for oid in list(tracked.keys()):
            o = eng.orders.get(oid)
            if o is None:
                if oid in cancelled:
                    n_censored_cancel += 1
                else:
                    b = bucket_at_start.get(oid)
                    if b is not None:
                        fills[b] = fills.get(b, 0) + 1
                del tracked[oid]
            elif o.size < tracked[oid][1] - 1e-15:
                b = bucket_at_start.get(oid)
                if b is not None:
                    fills[b] = fills.get(b, 0) + 1
                del tracked[oid]

        for oid, o in eng.orders.items():
            if oid not in tracked and not str(o.agent_id).startswith("seed"):
                tracked[oid] = [o.side, o.size]

    diag = {"censored_cancel": n_censored_cancel,
            "censored_end": len(tracked),
            "mean_resting": statistics.mean(resting_counts) if resting_counts else 0,
            "total_fills": sum(fills.values()),
            "total_exposure": sum(exposure.values())}
    return exposure, fills, dsum, diag


def wls_fit(xs, ys, ws):
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
    print("measure k on the clipped working point")
    print("lam=%.4f  p_market=%.4f  mean_lifetime=%.0fs  disp=%.4f  JOIN "
          "clipping.  MM ABSENT." % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x %.0fs, $%.0f buckets; measure_k.py's method unchanged."
          % (len(SEEDS), DURATION, BUCKET_W))
    print("")

    EXP, FIL, DSM = {}, {}, {}
    for s in SEEDS:
        e, f, d, diag = run_one(s)
        print("  seed %d: fills=%d  exposure=%.0f order-sec  mean_resting=%.1f  "
              "censored(cancel/end)=%d/%d  (%.0fs)"
              % (s, diag["total_fills"], diag["total_exposure"],
                 diag["mean_resting"], diag["censored_cancel"],
                 diag["censored_end"], time.time() - t0), flush=True)
        for k_, v in e.items():
            EXP[k_] = EXP.get(k_, 0.0) + v
        for k_, v in f.items():
            FIL[k_] = FIL.get(k_, 0) + v
        for k_, v in d.items():
            DSM[k_] = DSM.get(k_, 0.0) + v

    print("\n" + "=" * 96)
    print("Empirical fill-intensity curve (pooled over %d seeds, MM ABSENT)"
          % len(SEEDS))
    print("=" * 96)
    print("%10s %12s %10s %14s %12s %10s"
          % ("delta ($)", "exposure(s)", "fills", "hazard(/s)", "ln hazard",
             "mean d"))
    print("-" * 96)

    rows = []
    shown = 0
    for b in sorted(EXP.keys()):
        expo, nf = EXP[b], FIL.get(b, 0)
        if expo <= 0:
            continue
        dbar = DSM[b] / expo
        haz = nf / expo
        lo = b * BUCKET_W
        star = "  (n<20, excluded from fit)" if nf < 20 else ""
        if shown < 45 or nf >= 20:
            print("%10s %12.0f %10d %14.3e %12s %10.2f%s"
                  % ("%.0f-%.0f" % (lo, lo + BUCKET_W), expo, nf, haz,
                     ("%.4f" % math.log(haz)) if haz > 0 else "  -inf",
                     dbar, star))
            shown += 1
        if nf >= 20 and haz > 0 and dbar >= 0:
            rows.append((dbar, math.log(haz), nf, lo))

    print("\n" + "=" * 96)
    print("Exponential fit  ln lambda = ln A - k * delta   (weighted by fill "
          "count)")
    print("=" * 96)
    print("measure_k.py's three windows:")
    out = {}
    for label, sel in (("full positive-delta support", lambda r: True),
                       ("near-touch 0-20 dollars", lambda r: r[3] < 20),
                       ("near-touch 0-40 dollars", lambda r: r[3] < 40)):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 3:
            print("  %-32s too few buckets" % label)
            continue
        a, b, se, r2 = wls_fit([r[0] for r in sub], [r[1] for r in sub],
                               [r[2] for r in sub])
        k = -b
        out[label] = k
        print("  %-32s  k = %.5f +/- %.5f   R2 = %.4f   nbuckets=%d  nfills=%d"
              % (label, k, se, r2, len(sub), sum(r[2] for r in sub)))
        if k > 0:
            print("  %-32s  1/k = $%.2f     2/k = $%.2f  = %.4f bp  <- implied "
                  "baseline spread"
                  % ("", 1.0 / k, 2.0 / k, 1e4 * (2.0 / k) / 62000.0))

    print("")
    print("supplementary windows, sized for this book rather than the old one")
    print("(the 0-20/0-40 windows were chosen when the touch was ~$30 wide; "
          "here it is $%.2f):" % THIS_BOOK_TOUCH_USD)
    for label, sel in (("near-touch 0-5 dollars", lambda r: r[3] < 5),
                       ("near-touch 0-10 dollars", lambda r: r[3] < 10)):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 3:
            print("  %-32s too few buckets (n=%d)" % (label, len(sub)))
            continue
        a, b, se, r2 = wls_fit([r[0] for r in sub], [r[1] for r in sub],
                               [r[2] for r in sub])
        k = -b
        out[label] = k
        print("  %-32s  k = %.5f +/- %.5f   R2 = %.4f   nbuckets=%d  nfills=%d"
              % (label, k, se, r2, len(sub), sum(r[2] for r in sub)))
        if k > 0:
            print("  %-32s  1/k = $%.2f     2/k = $%.2f  = %.4f bp"
                  % ("", 1.0 / k, 2.0 / k, 1e4 * (2.0 / k) / 62000.0))

    print("\n" + "=" * 96)
    print("Comparison; implied MM baseline width vs this book's actual touch")
    print("=" * 96)
    print("  this book's touch: %.4f bp = $%.2f  (clipped_mm_check.py, "
          "MM present, measured k)" % (THIS_BOOK_TOUCH_BP, THIS_BOOK_TOUCH_USD))
    print("  old book's k = %.5f -> 2/k = $%.2f = %.4f bp  (%.0fx this book's "
          "touch)"
          % (OLD_K, 2.0 / OLD_K, 1e4 * (2.0 / OLD_K) / 62000.0,
             (2.0 / OLD_K) / THIS_BOOK_TOUCH_USD))
    print("  %-32s %10s %12s %10s" % ("window", "k", "2/k ($)", "x touch"))
    for label, k in out.items():
        if k > 0:
            print("  %-32s %10.5f %12.2f %9.1fx"
                  % (label, k, 2.0 / k, (2.0 / k) / THIS_BOOK_TOUCH_USD))
    print("")
    print("  A 2/k far above the touch means the MM quotes outside the book "
          "and cannot compete;")
    print("  a 2/k at or below it means the MM sets the touch. Neither is "
          "chosen here; reported only.")
    print("\ntotal wall clock %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
