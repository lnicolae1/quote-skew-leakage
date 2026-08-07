# sniffer_saturation.py: sniffer saturation diagnostic and level vs velocity decomposition

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA
from sniffer_fill_only import PublicPrint, ols, TRAIN_FRAC, WINDOWS
from clipped_mm_check import _resid_best
from sniffer_skew import (SkewView, build_features, fit_score, T, N_SEC, SEEDS,
                          LAM, P_MARKET, LIFE, DISP, MM_ID, VEL_LAGS,
                          K_BRACKET)

WIDTHS = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03,
          0.05, 0.075, 0.1, 0.2, 0.5, 1.0]

R2_FLAGS = [0.5, 0.1]


def simulate(seed, k):
    """sniffer_skew.run()'s simulation half, returning the feature matrix"""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = MarketMaker(horizon=T, k=k)

    views, prints, q_series = [], [], []

    t = 0.0
    while t < T:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                for f in fills:
                    prints.append(PublicPrint(t, f.price, f.size, r.side))
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mb = ma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mb = o.price if mb is None else max(mb, o.price)
            else:
                ma = o.price if ma is None else min(ma, o.price)
        if mb is None or ma is None:
            continue
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is None or ra is None or ra <= rb:
            continue
        views.append(SkewView(t, mb, ma, 0.5 * (bb + ba), 0.5 * (rb + ra),
                              mm.time_remaining_years(t)))
        q_series.append(mm.q)

    rows, names, keep = build_features(views, prints, N_SEC)
    n = min(len(rows), len(q_series))
    return rows[:n], names, keep[:n], q_series[:n]


def add_norm_velocity(rows, names):
    """Append normalized-skew velocity columns, derived from the existing"""
    cn = names.index("skew_norm")
    col = [r[cn] for r in rows]
    out = list(names)
    for L in VEL_LAGS:
        out.append("skewnorm_vel_%ds" % L)
    for i, r in enumerate(rows):
        for L in VEL_LAGS:
            r.append(col[i] - (col[i - L] if i - L >= 0 else 0.0))
    return out


def coarsen(y, w):
    """Bucket to the nearest multiple of w"""
    return [w * math.floor(v / w + 0.5) for v in y]


def fit_coarse(rows, y_true, keep, cols, w):
    """Train on coarsened labels, score against TRUE q, out of sample"""
    idx = [i for i in range(len(y_true)) if keep[i]]
    cut = int(TRAIN_FRAC * len(idx))
    tr, te = idx[:cut], idx[cut:]

    ylab = coarsen(y_true, w) if w > 0 else y_true
    Xtr = [[rows[i][c] for c in cols] for i in tr]
    beta = ols(Xtr, [ylab[i] for i in tr])

    yte = [y_true[i] for i in te]
    mte = statistics.mean(yte)
    ss_tot = sum((a - mte) ** 2 for a in yte)

    pred = [sum(b * rows[i][c] for b, c in zip(beta, cols)) for i in te]
    ss_lit = sum((a - p) ** 2 for a, p in zip(yte, pred))
    if w > 0:
        snap = [w * math.floor(p / w + 0.5) for p in pred]
    else:
        snap = pred
    ss_snap = sum((a - p) ** 2 for a, p in zip(yte, snap))

    if ss_tot <= 0:
        return float("nan"), float("nan")
    return 1.0 - ss_lit / ss_tot, 1.0 - ss_snap / ss_tot


def fit_delta(rows, y_true, keep, cols, lag):
    """Same fit/score protocol, but the target is the true inventory change"""
    dy = [y_true[i] - (y_true[i - lag] if i - lag >= 0 else y_true[0])
          for i in range(len(y_true))]
    kp = [keep[i] and i >= lag for i in range(len(keep))]
    return fit_score(rows, dy, kp, cols)


def run_seed(seed, k):
    rows, names, keep, y = simulate(seed, k)
    names = add_norm_velocity(rows, names)
    c = {nm: i for i, nm in enumerate(names)}

    idx = [i for i in range(len(y)) if keep[i]]
    cut = int(TRAIN_FRAC * len(idx))
    yte = [y[i] for i in idx[cut:]]
    var_test = statistics.variance(yte)

    lvl = [c["const"], c["skew_norm"]]

    out = {"var_test": var_test, "sd_test": math.sqrt(var_test)}

    a0, b0 = fit_coarse(rows, y, keep, lvl, 0.0)
    out["w0_literal"], out["w0_snapped"] = a0, b0
    for w in WIDTHS:
        a, b = fit_coarse(rows, y, keep, lvl, w)
        out["lit_%g" % w] = a
        out["snap_%g" % w] = b

    vraw = [c["const"]] + [c["skew_vel_%ds" % L] for L in VEL_LAGS]
    vnrm = [c["const"]] + [c["skewnorm_vel_%ds" % L] for L in VEL_LAGS]
    both_raw = lvl + vraw[1:]
    both_nrm = lvl + vnrm[1:]

    for tag, cols in (("level", lvl), ("vel_raw", vraw), ("vel_norm", vnrm),
                      ("both_raw", both_raw), ("both_norm", both_nrm)):
        r2, rmse = fit_score(rows, y, keep, cols)
        out["s9_%s_r2" % tag] = r2
        out["s9_%s_rmse" % tag] = rmse

    for L in VEL_LAGS:
        r2r, _ = fit_delta(rows, y, keep,
                           [c["const"], c["skew_vel_%ds" % L]], L)
        r2n, _ = fit_delta(rows, y, keep,
                           [c["const"], c["skewnorm_vel_%ds" % L]], L)
        out["s9_dq%d_raw" % L] = r2r
        out["s9_dq%d_norm" % L] = r2n

    return out


def measure(k):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run_seed(s, k)
        if per is None:
            per = {key: [] for key in x}
        for key, v in x.items():
            per[key].append(v)
    n = len(SEEDS)
    out = {"k": k, "secs": time.time() - t0}
    for key, v in per.items():
        out[key] = statistics.mean(v)
        out[key + "_se"] = (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0
    return out


def first_below(res, prefix, thresh):
    """First bucket width at which mean R2 drops below thresh"""
    for w in WIDTHS:
        if res["%s_%g" % (prefix, w)] < thresh:
            return w
    return None


def main():
    print("=" * 122)
    print("Phase 6 steps 8 and 9; saturation (2.7.1) and level-vs-velocity "
          "(2.7.2), the inference half")
    print("=" * 122)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x %.0fs per k. gamma=DEFAULT_GAMMA=%g (not chosen). "
          "DEFAULT_K untouched. horizon=%.0fs."
          % (len(SEEDS), T, DEFAULT_GAMMA, T))
    print("Time split %d/%d, out-of-sample only. Residual mid throughout."
          % (int(100 * TRAIN_FRAC), int(100 * (1 - TRAIN_FRAC))))
    print("")
    print("Scope; 2.7.1 and 2.7.2 both put damage (bps) on the y-axis. This "
          "is the x-axis only.")
    print("Damage is step 7 (paired-seed A/B with a trading sniffer) and it "
          "needs the sniffer's")
    print("trading rule, which appendix open question 4 calls 'the one open "
          "Question that running")
    print("code will not answer ... a design choice, not a measurement'. Not "
          "chosen here.")
    print("")

    results = []
    for k, label in K_BRACKET:
        r = measure(k)
        results.append((k, label, r))
        print("  k=%.5f measured in %.0fs   sd(q) test = %.5f BTC"
              % (k, r["secs"], r["sd_test"]), flush=True)
    print("")

    print("=" * 122)
    print("Cross-check; the w -> 0 anchor must reproduce the committed step "
          "4 baseline (8cbfec0: 1.000000)")
    print("=" * 122)
    for k, _lab, r in results:
        print("  k=%.5f   no coarsening, R2 = %.6f +/- %.6f"
              % (k, r["w0_literal"], r["w0_literal_se"]))
    print("  If these are not 1.000000 the simulation loop reproduced in this "
          "file has drifted")
    print("  from sniffer_skew.run() and nothing below is trustworthy.")

    print("")
    print("=" * 122)
    print("Step 8; saturation via coarsened labels (2.7.1). "
          "x-axis of the damage plot.")
    print("=" * 122)
    for k, label, r in results:
        print("")
        print("  k = %.5f   [%s]" % (k, label))
        print("  sd(q) on the test slice = %.5f BTC" % r["sd_test"])
        print("")
        print("  %10s %10s %22s %22s %14s"
              % ("bucket w", "w / sd(q)", "8a literal R2", "8b bucket-res R2",
                 "8b analytic"))
        vt = r["var_test"]
        for w in WIDTHS:
            pred = 1.0 - (w * w / 12.0) / vt
            print("  %10.4f %10.2f %11.6f +/- %-8.6f %11.6f +/- %-8.6f "
                  "%14.6f"
                  % (w, w / r["sd_test"],
                     r["lit_%g" % w], r["lit_%g_se" % w],
                     r["snap_%g" % w], r["snap_%g_se" % w], pred))
        print("")
        for thresh in R2_FLAGS:
            wl = first_below(r, "lit", thresh)
            ws = first_below(r, "snap", thresh)
            print("    first w with R2 < %.1f :  8a literal = %s     "
                  "8b bucket-resolution = %s"
                  % (thresh,
                     ("%.4f BTC (%.1f x sd)" % (wl, wl / r["sd_test"]))
                     if wl is not None else "never in this grid",
                     ("%.4f BTC (%.1f x sd)" % (ws, ws / r["sd_test"]))
                     if ws is not None else "never in this grid"))
    print("")
    print("  8a literal       = 2.7.1's construction exactly: coarsen the "
          "label, refit OLS, score the")
    print("                     raw OLS output against TRUE q.")
    print("  8b bucket-res    = same fit, prediction snapped to the nearest "
          "bucket centre. A sniffer")
    print("                     trained on bucket labels cannot know finer "
          "than a bucket. This is the")
    print("                     curve that has the shape 2.7.1 describes "
          "('destroys within-bucket")
    print("                     magnitude, preserves sign/bucket exactly').")
    print("  8b analytic      = 1 - (w^2/12)/Var_test(q), the pure "
          "quantization-noise prediction for a")
    print("                     model that is otherwise perfect. Agreement "
          "means the curve carries no")
    print("                     information beyond sd(q).")
    print("")
    print("  Reading it: 2.7.1; 'If the curve saturates, the tension largely "
          "dissolves ... If the")
    print("  curve is linear or convex, the tension is real and probably "
          "fundamental.' That reading")
    print("  applies to the damage curve, not to this one. What this page "
          "fixes is which bucket")
    print("  widths are worth sweeping in Phase 7; widths where R2 is still "
          "~1 destroy nothing an")
    print("  adversary was using, and widths past the collapse are the ones "
          "with something to trade off.")

    print("")
    print("=" * 122)
    print("Step 9; level vs velocity (2.7.2 / 2.8.6b Route 2)")
    print("=" * 122)
    print("  Predicting the inventory level q, out of sample:")
    print("")
    print("  %-10s %20s %20s %20s %20s %20s"
          % ("k", "level only", "velocity only(raw)", "velocity only(norm)",
             "both (raw vel)", "both (norm vel)"))
    for k, _lab, r in results:
        print("  %-10.5f %10.6f+/-%-9.6f %10.4f+/-%-9.4f %10.4f+/-%-9.4f "
              "%10.6f+/-%-9.6f %10.6f+/-%-9.6f"
              % (k,
                 r["s9_level_r2"], r["s9_level_r2_se"],
                 r["s9_vel_raw_r2"], r["s9_vel_raw_r2_se"],
                 r["s9_vel_norm_r2"], r["s9_vel_norm_r2_se"],
                 r["s9_both_raw_r2"], r["s9_both_raw_r2_se"],
                 r["s9_both_norm_r2"], r["s9_both_norm_r2_se"]))
    print("")
    print("  level-alone is 1.000000 by construction, not by measurement: "
          "skew is -q*C, so the level")
    print("  feature is an invertible function of q and nothing can beat it. "
          "'Which carries the")
    print("  predictive weight' is therefore answered before the run. The "
          "informative column is")
    print("  velocity-alone: it is what stays readable if a defense freezes "
          "the level.")
    print("")
    print("  raw velocity  = skew[t] - skew[t-L]. Contaminated: skew = -q*C "
          "and C decays with the")
    print("                  horizon, so the difference mixes inventory change "
          "with clock decay --")
    print("                  the same artifact that depressed raw-skew R2 in "
          "step 4.")
    print("  norm velocity = skewnorm[t] - skewnorm[t-L], proportional to the "
          "inventory change alone.")
    print("")
    print("  Velocity against THE TRUE inventory change dq over the same lag "
          "-- the quantity Route 2")
    print("  actually names ('he's at 12 and climbing 5/minute'), which is a "
          "different question from")
    print("  predicting the level:")
    print("")
    print("  %-10s %26s %26s" % ("k", "dq over 60s", "dq over 300s"))
    for k, _lab, r in results:
        print("  %-10.5f   raw %8.4f / norm %8.4f     raw %8.4f / norm %8.4f"
              % (k, r["s9_dq60_raw"], r["s9_dq60_norm"],
                 r["s9_dq300_raw"], r["s9_dq300_norm"]))
    print("")
    print("  2.7.2: 'If velocity dominates ... the project reorients around "
          "hiding rate-of-accumulation")
    print("  rather than level. If level dominates, quantization stays "
          "central.' Both readings are")
    print("  about which the damage tracks. This page reports predictive "
          "weight only.")


if __name__ == "__main__":
    main()
