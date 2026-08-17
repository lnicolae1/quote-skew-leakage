# anchor_test.py: is the residual at d = 0 the adverse-move clock starting before the maker's own trade? (analysis only)

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import lifetime_sweep as LS
import residual_fit as RF
from damage_ab import SEEDS, mean_se

H_SHORT = LS.H_SHORT
N_BINS = 8
CONTROL_LIFE = 720.0

run_arm = RF.run_arm
wls = RF.wls
wmean = RF.wmean
keeps = RF.keeps

G1 = RF.G1
G1_TOL = RF.G1_TOL
G1_AGREE_4DP = RF.G1_AGREE_4DP
G1_THR = RF.G1_THR
G1_TOL_4DP = RF.G1_TOL_4DP
G1_OCTILES = RF.G1_OCTILES
G1_TOL_OCT = RF.G1_TOL_OCT

ANCHORS = ("m0", "now", "post")
WINDOWS = (5.0, 10.0, 20.0, 40.0)
DECISION_WINDOW = 10.0
N_QUART = 4
MIN_WIN_N = 30

HALF = 0.5
HALF_SE = 2.0
CONV_SE = 2.0
GRAD_SE = 2.0
SHARE_SE = 2.0

SUBSTANTIAL_SHARE = 15.0

SELFTEST_SEED = 0
SELFTEST_SHORT = 86400.0
SCALE = 100.0

NAN = float("nan")


def anchors(ps, mj, m0, s_post, s_now):
    """The three re-anchorings of the h=60 adverse move"""
    return (-ps * (mj - m0), -ps * (mj - s_now), -ps * (mj - s_post))


def solve3(a, b):
    """Gaussian elimination with partial pivoting on a 3x3 system"""
    m = [list(a[i]) + [b[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-300:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(3):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, 4):
                m[r][k] -= f * m[c][k]
    return [m[i][3] / m[i][i] for i in range(3)]


def wls2(pts):
    """Weighted quadratic fit y = a + b*x + c*x^2"""
    n = len(pts)
    if n < 4:
        return (NAN, NAN, NAN, NAN, n)
    s0 = s1 = s2 = s3 = s4 = t0 = t1 = t2 = 0.0
    for (x, y, w) in pts:
        u = x / SCALE
        u2 = u * u
        s0 += w
        s1 += w * u
        s2 += w * u2
        s3 += w * u2 * u
        s4 += w * u2 * u2
        t0 += w * y
        t1 += w * u * y
        t2 += w * u2 * y
    if s0 <= 0:
        return (NAN, NAN, NAN, NAN, n)
    sol = solve3([[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]], [t0, t1, t2])
    if sol is None:
        return (NAN, NAN, NAN, NAN, n)
    ca, cb, cc = sol
    ybar = t0 / s0
    sst = ssr = 0.0
    for (x, y, w) in pts:
        u = x / SCALE
        sst += w * (y - ybar) ** 2
        ssr += w * (y - ca - cb * u - cc * u * u) ** 2
    r2 = (1.0 - ssr / sst) if sst > 0 else NAN
    return (ca, cb / SCALE, cc / (SCALE * SCALE), r2, n)


def umean(vals):
    """Unweighted mean, NAN on empty"""
    return statistics.mean(vals) if vals else NAN


def analyse_anchor(seed, life):
    """One seed"""
    fills, mid, extras = run_arm(seed, life)
    n = extras["n"]
    nf = len(fills)
    out = {"seed": float(seed), "n_fills": float(nf),
           "n_post_fb": float(extras["n_post_fb"])}
    if nf == 0:
        return out

    _bysec = {}
    for (ti, _ps, _sz, _m0, s_post, _v, _st) in fills:
        _e = _bysec.setdefault(ti, [0, set()])
        _e[0] += 1
        _e[1].add(s_post)
    out["nc5_multi_sec"] = float(sum(1 for e in _bysec.values() if e[0] > 1))
    out["nc5_bad_sec"] = float(sum(1 for e in _bysec.values() if len(e[1]) > 1))

    rows_all = []
    rows_y = []
    for (ti, ps, sz, m0, s_post, v, st) in fills:
        d = -ps * st
        s_now = mid[ti]
        span_pre = -ps * (s_post - m0)
        span_req = -ps * (s_now - s_post)
        j = ti - 1 + H_SHORT
        if j <= n:
            a3 = anchors(ps, mid[j], m0, s_post, s_now)
        else:
            a3 = None
        row = (d, sz, a3, st, span_pre, span_req)
        rows_all.append(row)
        if a3 is not None:
            rows_y.append(row)

    out["n_with_a60"] = float(len(rows_y))

    out["adv60"] = wmean([(r[2][0], r[1]) for r in rows_y])
    out["wrong"] = 100.0 * sum(1 for r in rows_all if r[0] > 0) / nf
    thr = LS.pct(sorted(abs(f[6]) for f in fills), 0.20)
    out["thr"] = thr
    buck_y = [r for r in rows_y if keeps(r[3], thr)]
    out["n_in_bucket"] = float(len(buck_y))
    out["agree60"] = wmean([(r[2][0], r[1]) for r in buck_y]) if buck_y else NAN

    srt = sorted(rows_all, key=lambda r: r[0])
    m = len(srt)
    for bidx in range(N_BINS):
        lo, hi = int(m * bidx / N_BINS), int(m * (bidx + 1) / N_BINS)
        chunk = srt[lo:hi]
        ok = [c for c in chunk if c[2] is not None]
        out["oct_d_%d" % bidx] = umean([c[0] for c in chunk])
        out["oct_a_%d" % bidx] = (wmean([(c[2][0], c[1]) for c in ok])
                                  if ok else NAN)

    for ai, tag in enumerate(ANCHORS):
        out["ovr_%s_w" % tag] = wmean([(r[2][ai], r[1]) for r in rows_y])
        out["ovr_%s_u" % tag] = umean([r[2][ai] for r in rows_y])
        out["buck_%s_w" % tag] = (wmean([(r[2][ai], r[1]) for r in buck_y])
                                  if buck_y else NAN)
        out["buck_%s_u" % tag] = (umean([r[2][ai] for r in buck_y])
                                  if buck_y else NAN)
        aw, bw, r2w, _n, _s = wls([(r[0], r[2][ai], r[1]) for r in rows_y])
        au, bu, r2u, _n, _s = wls([(r[0], r[2][ai], 1.0) for r in rows_y])
        out["a_w_%s" % tag], out["b_w_%s" % tag] = aw, bw
        out["r2_w_%s" % tag] = r2w
        out["a_u_%s" % tag], out["b_u_%s" % tag] = au, bu
        out["r2_u_%s" % tag] = r2u
        out["conv_%s" % tag] = aw - au

    for w in WINDOWS:
        key = "win%d" % int(w)
        sub = [r for r in rows_y if abs(r[0]) <= w]
        out["%s_n" % key] = float(len(sub))
        out["%s_d_u" % key] = umean([r[0] for r in sub])
        out["%s_d_w" % key] = (wmean([(r[0], r[1]) for r in sub])
                               if sub else NAN)
        for ai, tag in enumerate(ANCHORS):
            vw = wmean([(r[2][ai], r[1]) for r in sub]) if sub else NAN
            vu = umean([r[2][ai] for r in sub]) if sub else NAN
            out["%s_%s_w" % (key, tag)] = vw
            out["%s_%s_u" % (key, tag)] = vu
            out["%s_%s_share" % (key, tag)] = (100.0 * vw / out["adv60"]
                                               if out["adv60"] else NAN)

    dk = "win%d" % int(DECISION_WINDOW)
    out["half_margin"] = HALF * out["%s_m0_w" % dk] - out["%s_post_w" % dk]

    out["local_wu_gap"] = out["%s_post_w" % dk] - out["%s_post_u" % dk]

    out["span_pre_w"] = wmean([(r[4], r[1]) for r in rows_all])
    out["span_pre_u"] = umean([r[4] for r in rows_all])
    out["span_req_w"] = wmean([(r[5], r[1]) for r in rows_all])
    out["span_req_u"] = umean([r[5] for r in rows_all])

    bysz = sorted(rows_all, key=lambda r: r[1])
    q = len(bysz)
    for qi in range(N_QUART):
        lo, hi = int(q * qi / N_QUART), int(q * (qi + 1) / N_QUART)
        chunk = bysz[lo:hi]
        ok = [c for c in chunk if c[2] is not None]
        out["q%d_n" % qi] = float(len(chunk))
        out["q%d_size" % qi] = umean([c[1] for c in chunk])
        out["q%d_pre_u" % qi] = umean([c[4] for c in chunk])
        out["q%d_pre_w" % qi] = (wmean([(c[4], c[1]) for c in chunk])
                                 if chunk else NAN)
        out["q%d_req_u" % qi] = umean([c[5] for c in chunk])
        out["q%d_m0_w" % qi] = (wmean([(c[2][0], c[1]) for c in ok])
                                if ok else NAN)
        out["q%d_post_w" % qi] = (wmean([(c[2][2], c[1]) for c in ok])
                                  if ok else NAN)
    out["size_grad"] = out["q3_pre_u"] - out["q0_pre_u"]

    for ai, tag in enumerate(ANCHORS):
        qa, qb, qc, qr2, _n = wls2([(r[0], r[2][ai], r[1]) for r in rows_y])
        out["qd_a_%s" % tag] = qa
        out["qd_b_%s" % tag] = qb
        out["qd_c_%s" % tag] = qc
        out["qd_r2_%s" % tag] = qr2
        out["qd_gap_%s" % tag] = qa - out["%s_%s_w" % (dk, tag)]

    out["sd_d"] = (statistics.stdev([r[0] for r in rows_y])
                   if len(rows_y) > 1 else NAN)
    return out


def measure(life):
    """All seeds, one arm"""
    per = []
    t0 = time.time()
    for s in SEEDS:
        x = analyse_anchor(s, life)
        per.append(x)
        nms = x.get("nc5_multi_sec", 0.0)
        nbs = x.get("nc5_bad_sec", 0.0)
        assert nms > 0, ("NC5 IS VACUOUS ON SEED %d -- no second held two "
                         "fills, so s_post constancy was never exercised on "
                         "the arm's own data." % s)
        assert nbs == 0, ("s_post VARIED WITHIN A SECOND on seed %d (%d "
                          "seconds affected). The header's per-second reading "
                          "is WRONG and nothing downstream should be believed."
                          % (s, int(nbs)))
        print("    seed %d done (%.0fs elapsed)  NC5: %d multi-fill seconds, "
              "%d with a varying s_post"
              % (s, time.time() - t0, int(nms), int(nbs)), flush=True)
    keys = set()
    for p in per:
        keys.update(p.keys())
    res = {}
    for k in sorted(keys):
        vals = [p[k] for p in per if k in p]
        res[k] = mean_se(vals)
    res["_secs"] = (time.time() - t0, 0.0, len(per))
    res["_nseed"] = (float(len(per)), 0.0, len(per))
    return res


def selftest():
    print("=" * 126)
    print("Self-test. Negative controls, the anchoring collapse, the exact")
    print("identity, and the per-second s_post property that this file's whole")
    print("reading of span_pre depends on.")
    print("=" * 126)

    pts = [(x * 1.0, 3.0 + 0.5 * x, 1.0 + 0.1 * x) for x in range(1, 40)]
    a, b, r2, nn, _sw = wls(pts)
    print("  NC1  wls on an exact line y = 3 + 0.5x   a=%.10f b=%.10f R2=%.10f"
          % (a, b, r2))
    assert abs(a - 3.0) < 1e-8 and abs(b - 0.5) < 1e-8

    pts2 = [(x * 1.0, 7.0 - 0.25 * x + 0.004 * x * x, 1.0 + 0.03 * x)
            for x in range(-200, 401, 7)]
    qa, qb, qc, qr2, _n = wls2(pts2)
    print("  NC2  wls2 on an exact parabola 7 - 0.25x + 0.004x^2   "
          "a=%.8f b=%.8f c=%.8f R2=%.8f" % (qa, qb, qc, qr2))
    assert abs(qa - 7.0) < 1e-6, "wls2 intercept off: %r" % qa
    assert abs(qb + 0.25) < 1e-8, "wls2 slope off: %r" % qb
    assert abs(qc - 0.004) < 1e-10, "wls2 curvature off: %r" % qc

    lin = [(x * 1.0, 2.0 + 0.3 * x, 1.0) for x in range(-300, 301, 3)]
    la, lb, lc, _r, _n = wls2(lin)
    print("  NC2b wls2 on a straight line   a=%.8f b=%.8f c=%.3e (must be 0)"
          % (la, lb, lc))
    assert abs(lc) < 1e-12 and abs(la - 2.0) < 1e-8

    bad = 0
    for ps in (-1.0, 1.0):
        for m0 in (61990.0, 62010.0):
            for mj in (61950.0, 62050.0):
                t3 = anchors(ps, mj, m0, m0, m0)
                if not (t3[0] == t3[1] == t3[2] == -ps * (mj - m0)):
                    bad += 1
    print("  NC3  anchors() with s_post = s_now = m0 collapse to one value: "
          "%d disagreements in 8 cases" % bad)
    assert bad == 0
    t4 = anchors(1.0, 62050.0, 62000.0, 62010.0, 62020.0)
    print("  NC3b anchors() with three distinct starts -> %s  (must be three "
          "different numbers)" % (tuple(round(x, 4) for x in t4),))
    assert len(set(t4)) == 3

    fills, mid, extras = run_arm(SELFTEST_SEED, CONTROL_LIFE,
                                 horizon=SELFTEST_SHORT)
    n = extras["n"]
    assert len(fills) > 0, ("NC4/NC5 ARE VACUOUS -- zero fills at horizon=%.0f."
                            % SELFTEST_SHORT)

    worst = 0.0
    ncmp = 0
    for (ti, ps, sz, m0, s_post, v, st) in fills:
        j = ti - 1 + H_SHORT
        if j > n:
            continue
        s_now = mid[ti]
        a_m0, a_now, a_post = anchors(ps, mid[j], m0, s_post, s_now)
        span_pre = -ps * (s_post - m0)
        span_req = -ps * (s_now - s_post)
        worst = max(worst, abs(a_m0 - (span_pre + span_req + a_now)))
        worst = max(worst, abs(a_post - (a_m0 - span_pre)))
        ncmp += 2
    print("  NC4  exact decomposition on %d real fills (%d checks): worst "
          "absolute error %.3e" % (len(fills), ncmp, worst))
    assert ncmp > 0, "NC4 checked nothing -- every fill had j > n"
    assert worst < 1e-6, "the decomposition identity FAILED by %r" % worst

    bysec = {}
    for (ti, ps, sz, m0, s_post, v, st) in fills:
        bysec.setdefault(ti, set()).add(s_post)
    multi = [t for t, s in bysec.items() if len(s) > 1]
    shared = [t for t, s in bysec.items() if len(s) == 1]
    nmulti_sec = sum(1 for t in bysec if
                     sum(1 for f in fills if f[0] == t) > 1)
    print("  NC5  s_post constant within a second: %d seconds hold a fill, %d "
          "hold more than one fill, %d show more than one s_post value"
          % (len(bysec), nmulti_sec, len(multi)))
    if nmulti_sec == 0:
        print("       Warn: no second held two fills at this horizon, so "
              "constancy was NOT exercised here. Not an error; the binding "
              "check runs in measure() on the arm.")
    assert not multi, "s_post VARIED within a second at %r -- header is wrong" \
                      % multi[:5]
    assert shared

    print("  NC6  keeps(thr, thr)=%s  keeps(-thr, thr)=%s  keeps(thr+eps)=%s "
          "(committed rule is `if abs(st) > thr: continue`)"
          % (keeps(40.0, 40.0), keeps(-40.0, 40.0), keeps(40.0001, 40.0)))
    assert keeps(40.0, 40.0) and keeps(-40.0, 40.0)
    assert not keeps(40.0001, 40.0)

    f1, m1, e1 = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    f2, m2, e2 = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    same = (f1 == f2) and (m1 == m2) and (e1 == e2)
    print("  T5   determinism (horizon=%.0fs, %d fills, %d mid entries)   "
          "identical: %s" % (SELFTEST_SHORT, len(f1), len(m1), same))
    assert len(f1) > 0, ("T5 IS VACUOUS -- zero fills at horizon=%.0f."
                         % SELFTEST_SHORT)
    assert same
    names = ("ti", "ps", "size", "m0", "s_post", "v", "st")
    nf = 0
    for r1, r2 in zip(f1, f2):
        for idx, nm in enumerate(names):
            assert r1[idx] == r2[idx], "T5 field %s differs on a record" % nm
            nf += 1
    print("       %d per-field comparisons, all equal" % nf)

    print("  T6   run_arm is residual_fit.run_arm: %s   wls: %s   wmean: %s   "
          "keeps: %s" % (run_arm is RF.run_arm, wls is RF.wls,
                         wmean is RF.wmean, keeps is RF.keeps))
    assert run_arm is RF.run_arm and wls is RF.wls
    assert wmean is RF.wmean and keeps is RF.keeps

    print("  Self-test PASS.")
    print("")


def gate1(r):
    """Reproduce the committed figures"""
    print("=" * 126)
    print("Gate 1. Reproduce the committed figures before reporting anything "
          "NEW. Any failure halts.")
    print("=" * 126)
    bad = []

    def chk(label, got, want, tol, src):
        d = abs(got - want)
        ok = d <= tol
        if not ok:
            bad.append(label)
        print("  %-34s got %12.4f   want %12.4f   diff %10.6f   %-4s  %s"
              % (label, got, want, d, "OK" if ok else "FAIL", src))

    chk("adverse move h=60", r["adv60"][0], G1["adv60"], G1_TOL,
        "lifetime_sweep_results.txt:38")
    chk("agree-bucket adverse move (2dp)", r["agree60"][0], G1["agree60"],
        G1_TOL, "lifetime_sweep_results.txt:38")
    chk("agree-bucket adverse move (4dp)", r["agree60"][0], G1_AGREE_4DP,
        G1_TOL_4DP, "staleness_results.txt")
    chk("wrong-side pct", r["wrong"][0], G1["wrong"], G1_TOL,
        "lifetime_sweep_results.txt:38")
    chk("bucket threshold $", r["thr"][0], G1_THR, G1_TOL_4DP,
        "staleness_results.txt")
    for i, (wd, wa) in enumerate(G1_OCTILES):
        chk("octile %d mean d" % (i + 1), r["oct_d_%d" % i][0], wd,
            G1_TOL_OCT, "staleness_results.txt sec.1")
        chk("octile %d adverse h=60" % (i + 1), r["oct_a_%d" % i][0], wa,
            G1_TOL_OCT, "staleness_results.txt sec.1")

    print("")
    if bad:
        print("  Gate 1 FAIL on %d figures: %s" % (len(bad), ", ".join(bad)))
        print("  HALTING. Nothing new is reported off a run that does not "
              "reproduce what is already committed.")
        return False
    print("  Gate 1 PASS. All %d committed figures reproduced (3 headline, "
          "the 4dp agree, the threshold, and 8 octile rows on both axes)."
          % (5 + 2 * len(G1_OCTILES)))
    print("")
    return True


def ms(r, k, dp=4):
    m, se, _k = r[k]
    return ("%.*f +/- %.*f" % (dp, m, dp, se))


def val(r, k):
    return r[k][0]


def se(r, k):
    return r[k][1]


def report_a(r):
    print("=" * 126)
    print("Test A. Re-anchor the adverse move. Same 60-second endpoint mid[j], "
          "three different start instants.")
    print("=" * 126)
    print("   m0   = mid[ti-1], end of the previous second, post-requote. THE "
          "committed anchor.")
    print("   post = s_post, end of second ti, pre-requote. NOT 'just after "
          "the fill'; see the header, NC5.")
    print("   now  = mid[ti], end of second ti, post-requote. Differs from "
          "`post` only by the maker's own requote.")
    print("   d is from m0 throughout, so only the outcome changes between "
          "rows. Windows are 60s / 59s+ / 59s+ (L2).")
    print("")
    print("  anchor      overall mean (sw)      agree-bucket (sw)        "
          "a intercept          b slope           R2")
    for tag in ANCHORS:
        print("  %-6s  %20s  %20s  %19s  %17s  %s"
              % (tag, ms(r, "ovr_%s_w" % tag), ms(r, "buck_%s_w" % tag),
                 ms(r, "a_w_%s" % tag), ms(r, "b_w_%s" % tag, 6),
                 ms(r, "r2_w_%s" % tag, 6)))
    print("")
    print("  unweighted, the robustness arm:")
    print("  anchor      overall mean (u)       agree-bucket (u)         "
          "a intercept          b slope           R2")
    for tag in ANCHORS:
        print("  %-6s  %20s  %20s  %19s  %17s  %s"
              % (tag, ms(r, "ovr_%s_u" % tag), ms(r, "buck_%s_u" % tag),
                 ms(r, "a_u_%s" % tag), ms(r, "b_u_%s" % tag, 6),
                 ms(r, "r2_u_%s" % tag, 6)))
    print("")
    print("  size-weighted minus unweighted intercept, paired per seed.")
    print("  Diagnostic only, not adjudicating. Condition (ii) no longer reads "
          "this; it was superseded because")
    print("  intercepts on this data are extrapolations through a convex "
          "relationship, missing their one")
    print("  checkable point by 14.1 SE. (ii) now reads the local |d| <= %.0f "
          "window instead, which assumes" % DECISION_WINDOW)
    print("  no functional form. These rows describe the misspecification; "
          "they do not decide anything.")
    for tag in ANCHORS:
        g, g_se, _k = r["conv_%s" % tag]
        print("    %-6s  a(sw) - a(u) = %9.4f +/- %-9.4f   within %.0f SE of "
              "zero ? %s" % (tag, g, g_se, CONV_SE,
                             abs(g) <= CONV_SE * g_se if g_se > 0 else False))
    print("")
    print("  reminder, from the header: the overall `post` mean is an algebraic "
          "consequence of the overall")
    print("  `m0` mean minus span_pre, not an independent measurement. NC4 "
          "asserts the identity numerically.")
    print("  The bucket means, the local windows and every intercept are "
          "independent measurements.")
    print("")


def report_b(r):
    print("=" * 126)
    print("Test B. Local estimates around d = 0. No global line is assumed. "
          "This replaces the intercept.")
    print("=" * 126)
    print("   Fills binned by signed d in symmetric windows about zero. This "
          "is the honest local answer:")
    print("   it assumes nothing about functional form, which matters because "
          "the life=720 run showed the")
    print("   relationship is convex in d and the global intercept is an "
          "extrapolation.")
    print("")
    print("  window       n fills        mean d (u)       a60_m0 (sw)      "
          "a60_post (sw)     a60_now (sw)    post as pct of %.4f"
          % val(r, "adv60"))
    for w in WINDOWS:
        k = "win%d" % int(w)
        nn = val(r, "%s_n" % k)
        flag = "  THIN" if nn < MIN_WIN_N else ""
        print("  |d|<=%-5.0f %6.1f+/-%-5.1f %14s %17s %17s %16s %14s%s"
              % (w, nn, se(r, "%s_n" % k), ms(r, "%s_d_u" % k, 2),
                 ms(r, "%s_m0_w" % k, 3), ms(r, "%s_post_w" % k, 3),
                 ms(r, "%s_now_w" % k, 3), ms(r, "%s_post_share" % k, 2),
                 flag))
    print("")
    print("  unweighted within the same windows:")
    print("  window       mean d (sw)      a60_m0 (u)       a60_post (u)     "
          "a60_now (u)")
    for w in WINDOWS:
        k = "win%d" % int(w)
        print("  |d|<=%-5.0f %15s %16s %16s %16s"
              % (w, ms(r, "%s_d_w" % k, 3), ms(r, "%s_m0_u" % k, 3),
                 ms(r, "%s_post_u" % k, 3), ms(r, "%s_now_u" % k, 3)))
    print("")
    print("  A window with fewer than %d fills per seed is marked thin and "
          "must not be leaned on (L3)." % MIN_WIN_N)
    print("")


def report_c(r):
    print("=" * 126)
    print("Test C. Decompose the 17.09. And the part that cannot be done.")
    print("=" * 126)
    print("   span_pre = -ps*(s_post - m0), m0 -> end of second ti, "
          "pre-requote. This is the figure printed as")
    print("   'own impact' in residual_fit_720.txt. The label was wrong. It is "
          "the full one-second adverse move")
    print("   across the fill's own second: the maker's own impact together "
          "with every other trade and quote in")
    print("   that second. NC5 shows s_post takes one value per second, shared "
          "by every fill in it.")
    print("")
    print("   span_req = -ps*(mid[ti] - s_post), the remainder of the second "
          "after s_post; which is exactly and")
    print("   only the maker's own requote's effect on the mid, since no other "
          "agent acts between the two reads.")
    print("")
    print("  span_pre  size-weighted   %20s $/BTC        unweighted  %s"
          % (ms(r, "span_pre_w"), ms(r, "span_pre_u")))
    print("  span_req  size-weighted   %20s $/BTC        unweighted  %s"
          % (ms(r, "span_req_w"), ms(r, "span_req_u")))
    print("  a60_m0    size-weighted   %20s $/BTC   (= span_pre + span_req + "
          "a60_now, exactly)" % ms(r, "ovr_m0_w"))
    print("")
    print("  the requested further split; own-fill impact versus drift "
          "before the fill; is not possible.")
    print("  There is no mid observation between m0 and s_post, because m0 is "
          "the mid at the start of second ti,")
    print("  and the entire second's flow is applied to the engine before any "
          "fill record is inspected (header,")
    print("  points 1-3). Recovering a pre-fill mid would require interleaving "
          "the flow, i.e. Changing the")
    print("  simulation. It is not estimated. The size gradient below is the "
          "discriminator that survives.")
    print("")
    print("  fill-size quartile     n      mean size (BTC)    span_pre (u)     "
          "span_pre (sw)    span_req (u)     a60_m0 (sw)    a60_post (sw)")
    for qi in range(N_QUART):
        print("  Q%d  %14.1f %18s %16s %16s %15s %15s %15s"
              % (qi + 1, val(r, "q%d_n" % qi), ms(r, "q%d_size" % qi, 6),
                 ms(r, "q%d_pre_u" % qi, 3), ms(r, "q%d_pre_w" % qi, 3),
                 ms(r, "q%d_req_u" % qi, 4), ms(r, "q%d_m0_w" % qi, 3),
                 ms(r, "q%d_post_w" % qi, 3)))
    print("")
    g, g_se, _k = r["size_grad"]
    print("  size gradient, paired per seed: span_pre(Q4,u) - span_pre(Q1,u) = "
          "%.4f +/- %.4f" % (g, g_se))
    print("    Unweighted within a size quartile on purpose: size-weighting "
          "inside a size bin would")
    print("    re-introduce the very gradient being tested.")
    print("    rising with size at %.0f SE ? %s"
          % (GRAD_SE, g > GRAD_SE * g_se if g_se > 0 else False))
    print("")


def report_d(r):
    print("=" * 126)
    print("Test D. Characterise the convexity. A description, NOT A model.")
    print("=" * 126)
    print("   adverse = a + b*d + c*d^2, size-weighted, against the linear fit "
          "from Test A. The quadratic")
    print("   intercept is still an extrapolation to d = 0; it is only worth "
          "anything where it agrees with")
    print("   Test B, which assumes no functional form. The gap is reported "
          "paired per seed.")
    print("")
    print("  anchor      a (quadratic)        b                    c          "
          "          R2 (quad)      R2 (linear)")
    for tag in ANCHORS:
        print("  %-6s  %19s %20s %22s %13s %15s"
              % (tag, ms(r, "qd_a_%s" % tag), ms(r, "qd_b_%s" % tag, 6),
                 ms(r, "qd_c_%s" % tag, 8), ms(r, "qd_r2_%s" % tag, 6),
                 ms(r, "r2_w_%s" % tag, 6)))
    print("")
    print("  quadratic intercept minus the local |d| <= %.0f estimate, paired "
          "per seed:" % DECISION_WINDOW)
    for tag in ANCHORS:
        g, g_se, _k = r["qd_gap_%s" % tag]
        print("    %-6s  %9.4f +/- %-9.4f   agree within 2 SE ? %s"
              % (tag, g, g_se,
                 abs(g) <= 2.0 * g_se if g_se > 0 else False))
    print("")
    print("  If these disagree materially, Test B is what should be quoted.")
    print("")


def decide(r):
    print("=" * 126)
    print("The decision rule. Both branches carry conditions this time.")
    print("=" * 126)
    dk = "win%d" % int(DECISION_WINDOW)
    n_win = val(r, "%s_n" % dk)
    loc_m0 = val(r, "%s_m0_w" % dk)
    loc_post = val(r, "%s_post_w" % dk)
    share, share_se, _k = r["%s_post_share" % dk]

    hm, hm_se, _k = r["half_margin"]
    cv, cv_se, _k = r["local_wu_gap"]
    sg, sg_se, _k = r["size_grad"]

    c_i = (hm > HALF_SE * hm_se) if hm_se > 0 else False
    c_ii = (abs(cv) <= CONV_SE * cv_se) if cv_se > 0 else False
    c_iii = (sg > GRAD_SE * sg_se) if sg_se > 0 else False
    share_lo = share - SHARE_SE * share_se

    print("  the decision window is |d| <= %.0f, holding %.1f fills per seed "
          "(thin threshold %d)" % (DECISION_WINDOW, n_win, MIN_WIN_N))
    print("")
    print("  (i)   local a60_post below half local a60_m0, by more than %.0f SE"
          % HALF_SE)
    print("          local a60_m0   %s        local a60_post %s"
          % (ms(r, "%s_m0_w" % dk), ms(r, "%s_post_w" % dk)))
    print("          half*m0 - post = %.4f - %.4f = %.4f   (equals the paired "
          "mean below, by linearity)"
          % (HALF * loc_m0, loc_post, HALF * loc_m0 - loc_post))
    print("          margin paired per seed  %.4f +/- %.4f   needs > %.4f "
          "  -> (i) %s" % (hm, hm_se, HALF_SE * hm_se, c_i))
    print("  (ii)  Local a60_post agrees size-weighted vs unweighted, "
          "inside |d| <= %.0f" % DECISION_WINDOW)
    print("          local a60_post (sw) %s     (u) %s"
          % (ms(r, "%s_post_w" % dk), ms(r, "%s_post_u" % dk)))
    print("          paired difference %.4f +/- %.4f, within %.0f SE (%.4f) ? "
          "-> (ii) %s" % (cv, cv_se, CONV_SE, CONV_SE * cv_se, c_ii))
    print("          NOT the regression intercepts: the committed run showed "
          "those are extrapolations")
    print("          through a convex relationship, missing their one "
          "checkable point by 14.1 SE.")
    print("  (iii) span_pre rises with fill size")
    print("          Q4 - Q1 paired %.4f +/- %.4f, above %.0f SE ? -> (iii) %s"
          % (sg, sg_se, GRAD_SE, c_iii))
    print("")
    print("  local a60_post as a share of the unrounded overall a60_m0 "
          "(%s $/BTC):" % ms(r, "adv60"))
    print("          %s pct   lower %.0f-SE bound %.4f pct   must exceed "
          "%.1f pct" % (ms(r, "%s_post_share" % dk, 2), SHARE_SE, share_lo,
                        SUBSTANTIAL_SHARE))
    print("          %.1f was pre-registered before any result existed. It sits "
          "well below the status quo" % SUBSTANTIAL_SHARE)
    print("          (committed bucket 27.93 pct, identity-forced overall "
          "a60_post 31.10 pct), so the branch")
    print("          is not rigged to confirm, and well clear of zero, so "
          "'substantial' means something.")
    print("")

    artefact = c_i and c_ii and c_iii
    real = share_lo > SUBSTANTIAL_SHARE
    if artefact and not real:
        print("  Branch artefact fires. All three conditions hold and the "
              "local a60_post estimate is not a")
        print("  substantial share of the unrounded overall a60_m0. THE "
              "residual at d = 0 is largely the")
        print("  Clock starting before the")
        print("  Maker's own trade.")
        print("    local a60_post, |d| <= %.0f : %s $/BTC  (%s pct of %s)"
              % (DECISION_WINDOW, ms(r, "%s_post_w" % dk),
                 ms(r, "%s_post_share" % dk, 2), ms(r, "adv60")))
        return
    if real and not artefact:
        print("  branch real fires. The local a60_post estimate remains a "
              "substantial share of the overall")
        print("  adverse move and the artefact conditions do not all hold. "
              "The estimate is Test B's local")
        print("  Window, not any regression intercept:")
        print("    local a60_post, |d| <= %.0f : %s $/BTC"
              % (DECISION_WINDOW, ms(r, "%s_post_w" % dk)))
        print("    as a share of the unrounded overall adverse move %s : %s pct"
              % (ms(r, "adv60"), ms(r, "%s_post_share" % dk, 2)))
        return
    print("  Neither branch fires cleanly. Stopping rather than picking the "
          "nearer.")
    print("    artefact conditions (i)/(ii)/(iii) = %s/%s/%s  -> all three ? %s"
          % (c_i, c_ii, c_iii, artefact))
    print("    substantial share ? %s   (share %.4f pct, lower %.0f-SE bound "
          "%.4f pct, threshold %.1f pct)"
          % (real, share, SHARE_SE, share_lo, SUBSTANTIAL_SHARE))
    print("  Both fired, or neither did. The task says say so and stop, and "
          "that is what this is.")
    print("  No estimate is quoted from an ambiguous rule.")


def limitations(r):
    print("")
    print("=" * 126)
    print("Limitations. Carried, not buried.")
    print("=" * 126)
    print("  L1 s_post is a per-second quantity, not per-fill (NC5). Test C's "
          "requested split of span_pre into")
    print("     own-fill impact versus prior drift is not possible and is not "
          "estimated.")
    print("  L2 The three anchorings differ in window length as well as start "
          "instant: 60s for m0, 59s+ for")
    print("     the other two, because the endpoint mid[j] is shared as "
          "specified.")
    print("  L3 Local windows are narrow slices of a distribution with sd(d) = "
          "%s. n is printed for every"
          % ms(r, "sd_d", 2))
    print("     window; any window below %d fills per seed is marked thin."
          % MIN_WIN_N)
    print("  L4 SUBSTANTIAL_SHARE = %.1f pct was pre-registered before any "
          "result existed, and the real branch" % SUBSTANTIAL_SHARE)
    print("     tests the lower %.0f-SE bound of the share against it. It sits "
          "below the status quo (committed" % SHARE_SE)
    print("     bucket 27.93 pct, identity-forced overall a60_post 31.10 pct) "
          "so the branch cannot fire on")
    print("     essentially the existing answer, and clear of zero so "
          "'substantial' is not 'nonzero'. The")
    print("     share and its SE are printed so another threshold can be "
          "applied without re-running.")
    print("  L5 One arm, life=720, %d seeds, one simulated world. Nothing here "
          "is measured on real BTCUSD data." % int(val(r, "_nseed")))
    print("  L6 SEs are across seeds. Every difference is computed per seed "
          "first, never as a difference of means.")
    print("  L7 The quadratic is a description of curvature, not a model. Its "
          "intercept is still an")
    print("     extrapolation to d = 0.")
    print("")
    print("  No committed default changed. Nothing retuned. No sweep. No "
          "repricing. JOIN clipping untouched.")
    print("  No sniffer.")


def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest()
        return
    selftest()
    print("running life=%.0f, %d seeds ..." % (CONTROL_LIFE, len(SEEDS)),
          flush=True)
    r = measure(CONTROL_LIFE)
    print("  arm done in %.0fs" % val(r, "_secs"))
    print("")
    print("  n_fills %s   n_with_a60 %s   n_in_bucket %s   s_post fallbacks %s"
          % (ms(r, "n_fills", 2), ms(r, "n_with_a60", 2),
             ms(r, "n_in_bucket", 2), ms(r, "n_post_fb", 2)))
    print("")
    if not gate1(r):
        sys.exit(1)
    report_a(r)
    report_b(r)
    report_c(r)
    report_d(r)
    decide(r)
    limitations(r)


if __name__ == "__main__":
    main()
