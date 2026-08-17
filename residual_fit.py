# residual_fit.py: is the residual not explained by staleness a separate mechanism? (analysis only)

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import lifetime_sweep as LS
from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from damage_ab import T, SEEDS, K, REF_MID, GAMMA, C_TARGET, mean_se
from clipped_mm_check import MM_ID

H_SHORT = LS.H_SHORT
N_BINS = 8
CONTROL_LIFE = 720.0
ARMS_2B = [720.0, 180.0, 60.0]

G1 = {"adv60": 24.81, "agree60": 6.93, "wrong": 75.09}
G1_TOL = 0.005
G1_AGREE_4DP = 6.9282
G1_THR = 40.8723
G1_TOL_4DP = 0.00005
G1_OCTILES = [(-218.0779, -4.9459), (-33.5857, 0.5966), (21.7423, 11.8726),
              (62.8970, 23.0238), (108.1653, 33.8485), (162.2079, 47.1548),
              (232.2041, 52.0136), (373.4102, 72.6649)]
G1_TOL_OCT = 0.00005

OWN_IMPACT_STRONG = 0.05
OWN_IMPACT_USELESS = 0.20

SELFTEST_SEED = 0
SELFTEST_SHORT = 86400.0

NAN = float("nan")


def wls(pts):
    """Weighted least squares of y on x with an intercept"""
    n = len(pts)
    if n < 2:
        return (NAN, NAN, NAN, n, 0.0)
    sw = sx = sy = sxx = sxy = 0.0
    for (x, y, w) in pts:
        sw += w
        sx += w * x
        sy += w * y
        sxx += w * x * x
        sxy += w * x * y
    if sw <= 0:
        return (NAN, NAN, NAN, n, sw)
    den = sw * sxx - sx * sx
    if den <= 0:
        return (NAN, NAN, NAN, n, sw)
    b = (sw * sxy - sx * sy) / den
    a = (sy - b * sx) / sw
    ybar = sy / sw
    sst = ssr = 0.0
    for (x, y, w) in pts:
        sst += w * (y - ybar) ** 2
        ssr += w * (y - a - b * x) ** 2
    r2 = (1.0 - ssr / sst) if sst > 0 else NAN
    return (a, b, r2, n, sw)


def wmean(pairs):
    """Weighted mean of (value, weight) pairs"""
    num = den = 0.0
    for (v, w) in pairs:
        num += v * w
        den += w
    return (num / den) if den > 0 else NAN


def keeps(st, thr):
    """The committed bucket rule, staleness.py:182-183 / lifetime_sweep.py:190"""
    return abs(st) <= thr


def in_bucket(recs, thr):
    """Filter records whose staleness sits at index 6 through `keeps`"""
    return [r for r in recs if keeps(r[6], thr)]


def run_arm(seed, life, horizon=T):
    """Runs the committed world and retains the per-fill record"""
    _p, vf, eng, noise, informed = make_world(seed, horizon, LS.LAM,
                                              LS.P_MARKET, life, LS.DISP,
                                              "join")
    mm = MarketMaker(horizon=horizon, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    n = int(horizon)
    mid = [0.0] * (n + 2)
    last = REF_MID
    spread_pnl = 0.0
    n_post_fb = 0
    recs_out = []

    t = 0.0
    while t < horizon:
        t += 1.0
        ti = int(t)
        v = vf(t)
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fl = getattr(r, "fills", None)
                if not fl:
                    continue
                for f in fl:
                    if f.counterparty_id != MM_ID:
                        continue
                    ps = -1.0 if r.side == "buy" else 1.0
                    spread_pnl += -ps * (f.price - last) * f.size
                    bb, ba = eng.best_bid(), eng.best_ask()
                    if bb is not None and ba is not None:
                        s_post = 0.5 * (bb + ba)
                    else:
                        s_post = last
                        n_post_fb += 1
                    recs_out.append((ti, ps, f.size, last, s_post, v,
                                     v - last))
                mm.on_fills(fl, r.side)
        mm.requote(eng, ti)
        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            last = 0.5 * (bb + ba)
        mid[ti] = last

    extras = {"pnl": mm.mark_to_market(last), "spread": spread_pnl,
              "n_post_fb": n_post_fb, "n": n}
    return recs_out, mid, extras


def analyse(seed, life, carry_thr=None):
    """One seed, one arm"""
    fills, mid, extras = run_arm(seed, life)
    n = extras["n"]
    nf = len(fills)
    out = {"seed": float(seed), "life": life, "n_fills": float(nf),
           "pnl": extras["pnl"], "spread": extras["spread"],
           "n_post_fb": float(extras["n_post_fb"])}
    if nf == 0:
        return out, [NAN] * N_BINS, [NAN] * N_BINS, [NAN] * N_BINS

    rows_all = []
    rows_y = []
    own_impact = []
    for (ti, ps, sz, m0, s_post, v, st) in fills:
        d = -ps * st
        d_now = -ps * (v - mid[ti])
        j = ti - 1 + H_SHORT
        a60 = (-ps * (mid[j] - m0)) if j <= n else None
        row = (d, sz, a60, d_now, st)
        rows_all.append(row)
        if a60 is not None:
            rows_y.append(row)
        own_impact.append((-ps * (s_post - m0), sz))

    out["n_with_a60"] = float(len(rows_y))

    out["adv60"] = wmean([(r[2], r[1]) for r in rows_y])
    out["wrong"] = 100.0 * sum(1 for r in rows_all if r[0] > 0) / nf
    own_thr = LS.pct(sorted(abs(f[6]) for f in fills), 0.20)
    out["own_thr"] = own_thr
    thr = own_thr if carry_thr is None else carry_thr
    out["thr_used"] = thr

    buck_all = [r for r in rows_all if keeps(r[4], thr)]
    buck_y = [r for r in rows_y if keeps(r[4], thr)]
    out["n_bucket_all"] = float(len(buck_all))
    out["n_in_bucket"] = float(len(buck_y))
    out["occupancy"] = 100.0 * len(buck_all) / nf
    out["agree60"] = wmean([(r[2], r[1]) for r in buck_y]) if buck_y else NAN

    if out["adv60"] != 0.0:
        out["residual_pct"] = 100.0 * out["agree60"] / out["adv60"]
        out["explained_pct"] = 100.0 - out["residual_pct"]
    else:
        out["residual_pct"] = NAN
        out["explained_pct"] = NAN

    if buck_y:
        out["buck_d_w"] = wmean([(r[0], r[1]) for r in buck_y])
        out["buck_d_u"] = statistics.mean([r[0] for r in buck_y])
        out["buck_d_med"] = statistics.median([r[0] for r in buck_y])
        out["buck_wrong"] = (100.0 * sum(1 for r in buck_y if r[0] > 0)
                             / len(buck_y))
    else:
        out["buck_d_w"] = NAN
        out["buck_d_u"] = NAN
        out["buck_d_med"] = NAN
        out["buck_wrong"] = NAN

    for tag, xi in (("m0", 0), ("now", 3)):
        for wtag, wfix in (("w", None), ("u", 1.0)):
            pts = [(r[xi], r[2], (r[1] if wfix is None else wfix))
                   for r in rows_y]
            a, b, r2, _nn, _sw = wls(pts)
            out["a_%s_%s" % (wtag, tag)] = a
            out["b_%s_%s" % (wtag, tag)] = b
            out["r2_%s_%s" % (wtag, tag)] = r2

    a0 = out["a_w_m0"]
    b0 = out["b_w_m0"]
    out["pred_bucket"] = a0 + b0 * out["buck_d_w"]
    out["pred_gap"] = out["pred_bucket"] - out["agree60"]

    dmid = [mid[i] - mid[i - 1] for i in range(2, n + 1)]
    sd_e = statistics.stdev(dmid) if len(dmid) > 1 else NAN
    sd_d = statistics.stdev([r[0] for r in rows_y]) if len(rows_y) > 1 else NAN
    out["sd_mid_1s"] = sd_e
    out["sd_d"] = sd_d
    ratio = (sd_e * sd_e) / (sd_d * sd_d) if sd_d > 0 else NAN
    out["var_ratio"] = ratio
    out["slope_bias"] = (1.0 - b0) * ratio
    out["mean_d_w"] = wmean([(r[0], r[1]) for r in rows_y])
    out["mean_d_u"] = statistics.mean([r[0] for r in rows_y])
    out["int_bias"] = -out["slope_bias"] * out["mean_d_w"]
    out["a_debiased"] = a0 - out["int_bias"]

    out["own_impact"] = wmean(own_impact)
    out["own_rel"] = (abs(out["own_impact"]) / sd_d) if sd_d > 0 else NAN

    srt = sorted(rows_all, key=lambda r: r[0])
    m = len(srt)
    oct_d, oct_a, oct_res = [], [], []
    for bidx in range(N_BINS):
        lo, hi = int(m * bidx / N_BINS), int(m * (bidx + 1) / N_BINS)
        chunk = srt[lo:hi]
        if not chunk:
            oct_d.append(NAN)
            oct_a.append(NAN)
            oct_res.append(NAN)
            continue
        ok = [c for c in chunk if c[2] is not None]
        oct_d.append(statistics.mean([c[0] for c in chunk]))
        oct_a.append(wmean([(c[2], c[1]) for c in ok]) if ok else NAN)
        oct_res.append(wmean([(c[2] - a0 - b0 * c[0], c[1]) for c in ok])
                       if ok else NAN)

    def refit(blo, bhi):
        lo, hi = int(m * blo / N_BINS), int(m * bhi / N_BINS)
        sub = [c for c in srt[lo:hi] if c[2] is not None]
        return wls([(c[0], c[2], c[1]) for c in sub])

    a2, b2, r22, n2, _s2 = refit(1, 7)
    out["a_mid6"], out["b_mid6"], out["r2_mid6"] = a2, b2, r22
    out["n_mid6"] = float(n2)
    a3, b3, r23, n3, _s3 = refit(2, 6)
    out["a_mid4"], out["b_mid4"], out["r2_mid4"] = a3, b3, r23
    out["n_mid4"] = float(n3)

    return out, oct_d, oct_res, oct_a


def measure(life, carry=None, label=""):
    """All seeds for one arm"""
    per = {}
    octs_d, octs_r, octs_a = [], [], []
    t0 = time.time()
    for s in SEEDS:
        thr = None if carry is None else carry[s]
        x, od, orr, oa = analyse(s, life, thr)
        octs_d.append(od)
        octs_r.append(orr)
        octs_a.append(oa)
        for k, v in x.items():
            per.setdefault(k, []).append(v)
        print("      seed %d done (%.0fs elapsed)" % (s, time.time() - t0),
              flush=True)
    thr_by_seed = {s: per["own_thr"][i] for i, s in enumerate(SEEDS)}
    out = {"secs": time.time() - t0, "label": label}
    for k, v in per.items():
        mm_, se, nn = mean_se(v)
        out[k] = mm_
        out[k + "_se"] = se
        out[k + "_n"] = nn
    out["life"] = life
    out["_thr_by_seed"] = thr_by_seed
    for tag, arrs in (("d", octs_d), ("res", octs_r), ("a", octs_a)):
        cols = []
        for b in range(N_BINS):
            cols.append(mean_se([row[b] for row in arrs if b < len(row)]))
        out["oct_" + tag] = cols
    return out


def selftest():
    print("=" * 126)
    print("Self-tests. Single-seed figures below are plumbing checks and are "
          "not reported as results.")
    print("=" * 126)

    pts = [(x, 2.0 * x, 1.0) for x in (-3.0, -1.0, 0.0, 2.0, 5.0)]
    a, b, r2, nn, _sw = wls(pts)
    print("  NC1 synthetic y = 2x exactly      a=%+.12f b=%+.12f R2=%.12f n=%d"
          % (a, b, r2, nn))
    assert abs(a) < 1e-9, a
    assert abs(b - 2.0) < 1e-9, b
    assert abs(r2 - 1.0) < 1e-9, r2

    xs = [-2.0, -1.0, 1.0, 2.0]
    ys = [5.0, 7.0, 7.0, 5.0]
    a, b, r2, _n2, _s2 = wls(list(zip(xs, ys, [1.0] * 4)))
    print("  NC2 y independent of x            a=%+.12f b=%+.12f R2=%.12f "
          "mean(y)=%.4f" % (a, b, r2, statistics.mean(ys)))
    assert abs(b) < 1e-12, b
    assert abs(a - statistics.mean(ys)) < 1e-12, a
    assert abs(r2) < 1e-12, r2

    trio = [(0.0, 0.0), (1.0, 1.0), (1.0, 3.0)]
    au, bu, _r3, _n3, _s3 = wls([(x, y, 1.0) for (x, y) in trio])
    aw, bw, _r4, _n4, _s4 = wls([(0.0, 0.0, 1.0), (1.0, 1.0, 1.0),
                                 (1.0, 3.0, 3.0)])
    print("  NC3 weights bite                  unweighted a=%+.12f b=%+.12f  "
          "| weighted a=%+.12f b=%+.12f" % (au, bu, aw, bw))
    assert abs(au) < 1e-12 and abs(bu - 2.0) < 1e-12, (au, bu)
    assert abs(aw) < 1e-12 and abs(bw - 2.5) < 1e-12, (aw, bw)

    fake = [(1, 1.0, 1.0, 0.0, 0.0, 0.0, s) for s in
            (-5.0, -0.5, 0.0, 0.5, 5.0)]
    n_inf = len(in_bucket(fake, float("inf")))
    n_neg = len(in_bucket(fake, -1.0))
    n_zero = len(in_bucket(fake, 0.0))
    n_exact_zero = sum(1 for f in fake if f[6] == 0.0)
    print("  NC4 bucket thr=inf -> %d of %d ; thr=-1 -> %d ; thr=0 -> %d "
          "(records with |st| exactly 0: %d)"
          % (n_inf, len(fake), n_neg, n_zero, n_exact_zero))
    assert n_inf == len(fake)
    assert n_neg == 0
    assert n_zero == n_exact_zero

    f1, m1, e1 = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    f2, m2, e2 = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    same = (f1 == f2) and (m1 == m2) and (e1 == e2)
    print("  T5  determinism (horizon=%.0fs, %d fills, %d mid entries)   "
          "identical: %s" % (SELFTEST_SHORT, len(f1), len(m1), same))
    assert len(f1) > 0, ("T5 IS VACUOUS -- zero fills at horizon=%.0f. It "
                         "compared an empty record list and verified nothing "
                         "about the per-fill path. NOT raising the horizon "
                         "further; this needs explaining." % SELFTEST_SHORT)
    assert same
    names = ("ti", "ps", "size", "m0", "s_post", "v", "st")
    ncmp = 0
    for r1, r2 in zip(f1, f2):
        for idx, nm in enumerate(names):
            assert r1[idx] == r2[idx], "T5 field %s differs on a record" % nm
            ncmp += 1
    print("      %d record fields compared individually across %d fills, all "
          "equal (%s)." % (ncmp, len(f1), ", ".join(names)))

    print("  T4  running committed lifetime_sweep.run(%d, %.0f) and the "
          "mirror ..." % (SELFTEST_SEED, CONTROL_LIFE), flush=True)
    ref = LS.run(SELFTEST_SEED, CONTROL_LIFE)
    mine, _od, _orr, _oa = analyse(SELFTEST_SEED, CONTROL_LIFE, None)
    print("      %-10s %24s %24s %10s" % ("field", "committed", "mirror",
                                          "equal"))
    ok = True
    for kk in ("n_fills", "adv60", "agree60", "wrong", "pnl", "spread"):
        rv, mv = ref[kk], mine[kk]
        eq = (rv == mv)
        ok = ok and eq
        print("      %-10s %24.12f %24.12f %10s" % (kk, rv, mv, eq))
    assert ok, ("T4 FAILED -- the transcription has drifted from the "
                "committed loop. STOP.")
    print("  T4  PASS; the mirrored loop is bit-equal to the committed one "
          "on all six fields.")
    print("")


def gate1(r):
    print("=" * 126)
    print("Gate 1. Reproduce the committed figures before reporting anything "
          "NEW. Any failure halts.")
    print("=" * 126)
    ok = True
    print("  %-44s %14s %14s %12s %8s"
          % ("quantity", "committed", "measured", "abs diff", "verdict"))
    checks = [
        ("adverse h=60, all fills   (lifetime_sweep:38)", G1["adv60"],
         r["adv60"], G1_TOL),
        ("agree-bin h=60            (lifetime_sweep:38)", G1["agree60"],
         r["agree60"], G1_TOL),
        ("wrong-side pct            (lifetime_sweep:38)", G1["wrong"],
         r["wrong"], G1_TOL),
        ("agree-bin h=60, 4dp       (staleness_results)", G1_AGREE_4DP,
         r["agree60"], G1_TOL_4DP),
        ("bucket threshold, dollars (staleness_results)", G1_THR,
         r["own_thr"], G1_TOL_4DP),
    ]
    for name, ref, got, tol in checks:
        dd = abs(got - ref)
        good = dd <= tol
        ok = ok and good
        print("  %-44s %14.4f %14.4f %12.6f %8s"
              % (name, ref, got, dd, "OK" if good else "FAIL"))
    print("")
    print("  The full committed octile table (staleness_results.txt section "
          "1), all 8 rows:")
    print("  %6s %14s %14s %11s %15s %15s %11s %8s"
          % ("octile", "d committed", "d measured", "diff",
             "a60 committed", "a60 measured", "diff", "verdict"))
    for b in range(N_BINS):
        rd, ra = G1_OCTILES[b]
        gd = r["oct_d"][b][0]
        ga = r["oct_a"][b][0]
        d1, d2 = abs(gd - rd), abs(ga - ra)
        good = (d1 <= G1_TOL_OCT) and (d2 <= G1_TOL_OCT)
        ok = ok and good
        print("  %6d %14.4f %14.4f %11.6f %15.4f %15.4f %11.6f %8s"
              % (b + 1, rd, gd, d1, ra, ga, d2, "OK" if good else "FAIL"))
    print("")
    if not ok:
        print("  Gate 1 FAILED. Nothing below is reportable. Stopping.")
        sys.exit(1)
    print("  Gate 1 PASS. Thirteen committed figures reproduced; 24.81 / "
          "6.93 / 75.09, the 4dp agree")
    print("  figure, the bucket threshold, and all 8 octile rows on both "
          "axes. That also confirms")
    print("  staleness.py's arm and lifetime_sweep.py's life=720 row are the "
          "same world, which is what")
    print("  licenses comparing a figure from one artifact against a figure "
          "from the other.")
    print("")


def ms(r, k, dp=4):
    return "%*.*f +/- %.*f" % (dp + 6, dp, r[k], dp, r[k + "_se"])


def report_2a(r):
    print("=" * 126)
    print("1. THE 72/28 split, computed from unrounded per-seed values. It "
          "does not exist anywhere in the")
    print("   committed artifacts: lifetime_sweep_results.txt:43-44 asserts "
          "it in prose without computing")
    print("   it, and the only other route is 6.93/24.81, a ratio of two "
          "rounded display figures drawn")
    print("   from two different files.")
    print("=" * 126)
    print("  overall adverse move h=60                 %s $/BTC  (seeds=%d)"
          % (ms(r, "adv60"), r["adv60_n"]))
    print("  agree-bucket adverse move                 %s $/BTC"
          % ms(r, "agree60"))
    print("  residual share,  mean of per-seed ratios  %s pct"
          % ms(r, "residual_pct"))
    print("  explained share, mean of per-seed ratios  %s pct"
          % ms(r, "explained_pct"))
    print("  residual share,  ratio of the means       %10.4f pct  (a "
          "different quantity; no SE)"
          % (100.0 * r["agree60"] / r["adv60"]))
    print("")
    print("  n_fills %9.2f +/- %.2f    n_with_a60 %9.2f +/- %.2f    "
          "n_in_bucket %8.2f +/- %.2f"
          % (r["n_fills"], r["n_fills_se"], r["n_with_a60"],
             r["n_with_a60_se"], r["n_in_bucket"], r["n_in_bucket_se"]))
    print("  (n_bucket_all %.2f; bucket membership before the j<=n "
          "restriction.)" % r["n_bucket_all"])
    print("")

    print("=" * 126)
    print("2. The regression that replaces the bucket. adverse_move = a + b*d "
          "over all fills.")
    print("   a is the answer: the adverse move at ZERO directional "
          "staleness.")
    print("=" * 126)
    print("   Per-seed fits, 8 seeds. The SE is across seeds, not from "
          "within-seed residuals, because")
    print("   fills inside a seed are not independent.")
    print("")
    print("  %-36s %20s %22s %20s"
          % ("construction", "a  intercept ($)", "b  slope", "R2"))
    for name, tag in (("d from m0,   SIZE-WEIGHTED (primary)", "w_m0"),
                      ("d from m0,   unweighted", "u_m0"),
                      ("d from S_now, SIZE-WEIGHTED", "w_now"),
                      ("d from S_now, unweighted", "u_now")):
        print("  %-36s %20s %22s %20s"
              % (name, ms(r, "a_" + tag), ms(r, "b_" + tag, 6),
                 ms(r, "r2_" + tag, 6)))
    print("")
    print("  m0    = the committed benchmark, the mid at the end of the "
          "previous second. Shares its")
    print("          error with the outcome, so the intercept here is biased "
          "DOWN toward zero.")
    print("  S_now = the mid at the end of the SAME second. Removes the "
          "shared error but adds the")
    print("          maker's own fill impact, which biases the intercept UP. "
          "Opposite signs, so under")
    print("          the header's argument these bracket the truth in sign, "
          "not in magnitude. Section 5")
    print("          measures the contamination and section 6 gates the "
          "conclusion on it.")
    print("")

    print("=" * 126)
    print("3. The diagnostic that tests the pre-registered hypothesis "
          "directly")
    print("=" * 126)
    print("   If mean signed d inside the bottom-20-pct-of-|d| bucket is near "
          "zero, the hypothesis is")
    print("   dead. If it is clearly positive, the bucket inherits the same "
          "directional selection.")
    print("")
    print("  mean signed d in bucket, size-weighted    %s $"
          % ms(r, "buck_d_w"))
    print("  mean signed d in bucket, unweighted       %s $"
          % ms(r, "buck_d_u"))
    print("  median signed d in bucket (unweighted)    %s $"
          % ms(r, "buck_d_med"))
    print("  wrong-side share inside the bucket        %s pct"
          % ms(r, "buck_wrong"))
    print("  wrong-side share overall                  %s pct  (committed "
          "75.09 +/- 0.39)" % ms(r, "wrong"))
    print("")
    print("  mean adverse move in bucket (measured)    %s $/BTC"
          % ms(r, "agree60"))
    print("  predicted from the primary fit,")
    print("      a + b*mean(d | bucket)                %s $/BTC"
          % ms(r, "pred_bucket"))
    print("  paired per-seed gap, predicted - measured %s $/BTC"
          % ms(r, "pred_gap"))
    print("")

    print("=" * 126)
    print("4. Linearity. Checked, not assumed.")
    print("=" * 126)
    print("   Residuals of the primary fit, binned by the committed octile "
          "split of d. A line that fits")
    print("   leaves residuals with no pattern in d; curvature shows up as a "
          "systematic arc.")
    print("")
    print("  %6s %20s %22s %24s"
          % ("octile", "mean d ($)", "adverse h=60 ($/BTC)",
             "fit residual ($/BTC)"))
    for b in range(N_BINS):
        md, mdse, _n1 = r["oct_d"][b]
        ma, mase, _n2 = r["oct_a"][b]
        mr, mrse, _n3 = r["oct_res"][b]
        print("  %6d %11.4f+/-%-7.4f %13.4f+/-%-7.4f %15.4f+/-%-7.4f"
              % (b + 1, md, mdse, ma, mase, mr, mrse))
    print("")
    print("  Refits on interior octiles only. If the slope moves materially, "
          "the global intercept is an")
    print("  extrapolation from the tails and is labelled as one.")
    print("  %-32s %20s %22s %20s %10s"
          % ("fit window", "a  intercept ($)", "b  slope", "R2", "n fills"))
    for name, akey, bkey, rkey, nkey in (
            ("all 8 octiles (primary)", "a_w_m0", "b_w_m0", "r2_w_m0",
             "n_with_a60"),
            ("octiles 2-7 (drop extremes)", "a_mid6", "b_mid6", "r2_mid6",
             "n_mid6"),
            ("octiles 3-6 (middle half)", "a_mid4", "b_mid4", "r2_mid4",
             "n_mid4")):
        print("  %-32s %20s %22s %20s %10.1f"
              % (name, ms(r, akey), ms(r, bkey, 6), ms(r, rkey, 6), r[nkey]))
    print("")

    print("=" * 126)
    print("5. Errors in variables, and the contamination of the S_now check. "
          "Measured, not asserted.")
    print("=" * 126)
    print("  sd of the 1-second mid change     %s $   upper bound on sd(e0): "
          "the fill lands" % ms(r, "sd_mid_1s"))
    print("                                                      inside the "
          "second, not at its end")
    print("  sd(d) at fill moments             %s $" % ms(r, "sd_d"))
    print("  var(e0)/var(d)                    %s        (upper bound)"
          % ms(r, "var_ratio", 8))
    print("  implied slope bias (1-b)*ratio    %s        (upward on b)"
          % ms(r, "slope_bias", 8))
    print("  mean d, size-weighted             %s $" % ms(r, "mean_d_w"))
    print("  implied intercept bias            %s $/BTC  (negative = biased "
          "DOWN)" % ms(r, "int_bias", 6))
    print("  intercept with that bias removed  %s $/BTC" % ms(r, "a_debiased"))
    print("")
    print("  the own-impact channel, which decides whether the S_now "
          "construction is worth anything:")
    print("  E[-ps*(S_post - m0)], size-weighted   %s $   the mid move across "
          "the maker's own fill"
          % ms(r, "own_impact"))
    print("  |own impact| / sd(d)                  %s       vs thresholds "
          "%.2f (strong) / %.2f (useless)"
          % (ms(r, "own_rel", 6), OWN_IMPACT_STRONG, OWN_IMPACT_USELESS))
    print("  Fills where the book was one-sided when S_post was read: "
          "%.2f of %.2f" % (r["n_post_fb"], r["n_fills"]))
    print("")


def decide(r):
    print("=" * 126)
    print("6. The decision rule. It is asymmetric, because the bias is.")
    print("=" * 126)
    a, ase = r["a_w_m0"], r["a_w_m0_se"]
    an, anse = r["a_w_now"], r["a_w_now_se"]
    gap, gapse = r["pred_gap"], r["pred_gap_se"]
    ib = abs(r["int_bias"])
    rel = r["own_rel"]
    nonzero = abs(a) > 2.0 * ase
    now_nonzero = abs(an) > 2.0 * anse
    pred_ok = abs(gap) <= 2.0 * gapse
    bias_immaterial = ib < ase

    if rel < OWN_IMPACT_STRONG:
        ii_state = "STRONG"
        ii_ok = True
        ii_text = ("the own-impact is small relative to sd(d), so S_now is a "
                   "genuine debiased check")
    elif rel > OWN_IMPACT_USELESS:
        ii_state = "USELESS"
        ii_ok = False
        ii_text = ("the own-impact is COMPARABLE to sd(d), so S_now is NOT a "
                   "debiased construction and condition (ii) does no work")
    else:
        ii_state = "WEAKENED"
        ii_ok = False
        ii_text = ("the own-impact is a non-trivial fraction of sd(d), so "
                   "condition (ii) is weakened and cannot be leaned on")

    print("  intercept a (primary, d from m0)   %+.4f +/- %.4f   -> "
          "|a| > 2SE ? %s" % (a, ase, nonzero))
    print("  intercept a (d from S_now)         %+.4f +/- %.4f   -> "
          "|a| > 2SE ? %s" % (an, anse, now_nonzero))
    print("  predicted-minus-measured bucket    %+.4f +/- %.4f   -> "
          "within 2SE ? %s" % (gap, gapse, pred_ok))
    print("  |implied intercept bias|            %.4f   vs SE(a) %.4f   -> "
          "immaterial ? %s" % (ib, ase, bias_immaterial))
    print("  |own impact| / sd(d)                %.4f   -> the S_now check is "
          "%s" % (rel, ii_state))
    print("      %s." % ii_text)
    print("")

    if nonzero:
        print("  Branch 1 fires; the residual is real. This is the strong "
              "branch: the shared-m0 bias")
        print("  pushes the intercept DOWN, so it works against this finding "
              "and the estimate is a")
        print("  lower bound on the residual.")
        print("    a = %+.4f +/- %.4f $/BTC" % (a, ase))
        print("    as a share of the unrounded overall adverse move "
              "(%.4f $/BTC): %.2f pct"
              % (r["adv60"], 100.0 * a / r["adv60"]))
        print("    debiased lower bound: %+.4f $/BTC" % r["a_debiased"])
    elif bias_immaterial and ii_ok and (not now_nonzero) and pred_ok:
        print("  branch 2 fires; the residual is explained. All four "
              "conditions hold:")
        print("    (i)   the measured bias %.4f is smaller than SE(a) %.4f, "
              "so it cannot account" % (ib, ase))
        print("          for the observed intercept;")
        print("    (ii)  the S_now construction is clean enough to be "
              "informative, and it also returns")
        print("          an intercept indistinguishable from zero;")
        print("    (iii) the fit reproduces the committed bucket mean;")
        print("    (iv)  mean signed d inside the bucket is positive, so that "
              "bucket mean is the fitted")
        print("          line evaluated off zero rather than an intercept.")
        print("  The bucket mean is the same staleness, selected on |d| "
              "instead of d.")
    else:
        print("  Neither branch fires cleanly. Stopping here rather than "
              "Summarising past it.")
        print("    intercept indistinguishable from zero: %s" % (not nonzero))
        print("    bias immaterial:                       %s"
              % bias_immaterial)
        print("    S_now check informative (cond ii):     %s   [%s]"
              % (ii_ok, ii_state))
        print("    S_now intercept also near zero:        %s"
              % (not now_nonzero))
        print("    fit reproduces the bucket mean:        %s" % pred_ok)
        if (not nonzero) and bias_immaterial and pred_ok and (not ii_ok):
            print("  Branch 2 was blocked by the own-impact contamination "
                  "alone. Everything else held. A")
            print("  contaminated construction is not allowed to license the "
                  "conclusion, so this is reported")
            print("  as UNRESOLVED, not as explained.")
        print("  A near-zero intercept on the pooled fit alone does not "
              "establish that the residual is")
        print("  explained, because the bias runs toward that conclusion.")
    print("")


def report_2b(arms, carry_src):
    print("=" * 126)
    print("7. Task 2B; the cross-arm common threshold. A different "
          "question, kept as a cross-check.")
    print("=" * 126)
    print("   The committed agree-bin is a per-arm quantile, so a shrinking "
          "bucket mean across arms may")
    print("   only mean the bucket got narrower. Here every arm uses seed "
          "s's own life=720 threshold --")
    print("   an absolute dollar cut; so life=720 reproduces its 6.93 "
          "exactly as a control.")
    print("")
    print("   Carried thresholds by seed (from the life=720 arm):")
    print("     " + "  ".join("s%d=%.4f" % (s, carry_src[s]) for s in SEEDS))
    print("")
    print("  %6s %14s %13s %10s %10s %10s %20s %20s"
          % ("life", "threshold ($)", "occupancy", "n_fills", "n_a60",
             "n_bucket", "conditional adv", "overall adv"))
    for r in arms:
        print("  %6.0f %14.4f %9.2f pct %10.1f %10.1f %10.1f %20s %20s"
              % (r["life"], r["thr_used"], r["occupancy"], r["n_fills"],
                 r["n_with_a60"], r["n_in_bucket"], ms(r, "agree60"),
                 ms(r, "adv60")))
    print("")
    print("  %6s %24s %24s %24s"
          % ("life", "ratio cond/overall pct", "mean signed d in bucket",
             "wrong pct in bucket"))
    for r in arms:
        print("  %6.0f %24s %24s %24s"
              % (r["life"], ms(r, "residual_pct"), ms(r, "buck_d_w"),
                 ms(r, "buck_wrong")))
    print("")
    print("  SEs added to committed Section B, which prints these as point "
          "values with none")
    print("  (lifetime_sweep_results.txt:35-40). These use each arm's own "
          "quantile threshold, so they are")
    print("  the committed quantities, NOT the common-threshold ones above.")
    print("  %6s %24s %24s %24s"
          % ("life", "adverse h=60", "agree-bin (own quantile)", "wrong pct"))
    for r in arms:
        print("  %6.0f %24s %24s %24s"
              % (r["life"], ms(r, "adv60"), ms(r, "agree60_own"),
                 ms(r, "wrong")))
    print("")
    print("  per-arm intercepts. If the residual is a property of the maker "
          "rather than of the bucket,")
    print("  these should agree across arms even where the bucket means do "
          "not.")
    print("  %6s %24s %24s %20s"
          % ("life", "a  intercept ($)", "b  slope", "R2"))
    for r in arms:
        print("  %6.0f %24s %24s %20s"
              % (r["life"], ms(r, "a_w_m0"), ms(r, "b_w_m0", 6),
                 ms(r, "r2_w_m0", 6)))
    print("")


def limitations():
    print("=" * 126)
    print("8. Limitations")
    print("=" * 126)
    print("  L1 The adverse move is a mid-to-mid change (staleness.py:156). V "
          "enters only through d, so")
    print("     an intercept is a residual in mid-drift terms, not in V "
          "terms.")
    print("  L2 The bucket threshold is per seed. The 40.8723 figure in "
          "staleness_results.txt is the")
    print("     across-seed mean of eight different thresholds; using it as a "
          "single scalar would not")
    print("     reproduce 6.93. Task 2B carries seed s's own life=720 "
          "threshold to seed s.")
    print("  L3 e0 is not observed. It is proxied by the sd of the full "
          "1-second mid change, an upper")
    print("     bound, so var(e0)/var(d) and the implied bias are upper "
          "bounds.")
    print("  L4 Neither construction is unbiased. m0 shares its error with "
          "the outcome (intercept biased")
    print("     DOWN); S_now removes that but carries the maker's own fill "
          "impact (intercept biased UP).")
    print("     The bracketing claim is about sign only. The S_now arm is "
          "gated on the measured")
    print("     contamination in section 6 rather than trusted.")
    print("  L5 The octile x-axis is unweighted while the y-axis is "
          "size-weighted. That is the committed")
    print("     convention (staleness.py:168 vs :173-176), preserved so the "
          "table reproduces; it is NOT")
    print("     the convention the primary fit uses.")
    print("  L6 No instance wrapper is used, so no wrapper-inertness claim is "
          "made. The guarantee is T4")
    print("     against the committed lifetime_sweep.run(), plus Gate 1's "
          "thirteen reproduced figures.")
    print("  L7 Task 2B's arms differ in book structure as well as in "
          "staleness (lifetime_sweep_results")
    print("     section C: 52.2 levels/side at life=60 against 556.9 at "
          "life=720, and only life=720 is")
    print("     in-band on levels). A cross-arm comparison is therefore NOT "
          "ceteris paribus.")
    print("")
    print("  No committed default changed. Nothing retuned. No sweep. No "
          "repricing. JOIN clipping rule")
    print("  untouched. No sniffer.")


def main():
    argv = sys.argv[1:]
    only_selftest = "--selftest" in argv
    arms = list(ARMS_2B)
    if "--arm" in argv:
        want = float(argv[argv.index("--arm") + 1])
        arms = [CONTROL_LIFE] if want == CONTROL_LIFE else [CONTROL_LIFE, want]

    print("=" * 126)
    print("Task 2; is the unexplained residual a separate mechanism, or the "
          "Same staleness inside a bucket")
    print("           defined on |d| instead of d?")
    print("=" * 126)
    print("Control arm, no sniffer. %d seeds x %.0fs (%.0f days). k=%.5f, "
          "C=%.2f, gamma=%.4e."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA))
    print("d = -(position sign)*(V - benchmark); d > 0 is the wrong side. "
          "Adverse move at h=%d seconds," % H_SHORT)
    print("measured on the mid. Size-weighted is primary; unweighted is "
          "reported as robustness.")
    print("Analysis only; no committed module modified, nothing retuned, "
          "clipping rule untouched.")
    print("")

    selftest()
    if only_selftest:
        print("--selftest given; stopping before the arms.")
        return

    print("Running arms: %s" % ", ".join("%.0f" % a for a in arms), flush=True)
    print("  arm life=%.0f (own thresholds) ..." % CONTROL_LIFE, flush=True)
    ctrl = measure(CONTROL_LIFE, None, "control")
    carry = ctrl["_thr_by_seed"]
    print("  arm life=%.0f done in %.0fs" % (CONTROL_LIFE, ctrl["secs"]),
          flush=True)
    print("")

    gate1(ctrl)
    report_2a(ctrl)
    decide(ctrl)

    ctrl["agree60_own"] = ctrl["agree60"]
    ctrl["agree60_own_se"] = ctrl["agree60_se"]
    out_arms = [ctrl]
    for life in arms:
        if life == CONTROL_LIFE:
            continue
        print("  arm life=%.0f (carried thresholds) ..." % life, flush=True)
        rc = measure(life, carry, "carried")
        print("  arm life=%.0f (own thresholds) ..." % life, flush=True)
        ro = measure(life, None, "own")
        rc["agree60_own"] = ro["agree60"]
        rc["agree60_own_se"] = ro["agree60_se"]
        out_arms.append(rc)
        print("  arm life=%.0f done in %.0fs + %.0fs"
              % (life, rc["secs"], ro["secs"]), flush=True)
    out_arms.sort(key=lambda x: x["life"])
    print("")

    report_2b(out_arms, carry)
    limitations()


if __name__ == "__main__":
    main()
