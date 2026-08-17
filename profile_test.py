# profile_test.py: time profile a(h) of the maker's adverse move, h = 1..300 s (life=720, 8 seeds)
# question: does the local loss revert (transient own impact) or keep growing (permanent staleness)?
# - a(h) = size-weighted -ps * (mid[ti-1+h] - m0), m0 = mid[ti-1]; fills past the run end skipped
# - d = -ps * st (directional staleness; d > 0 = maker on wrong side)
# - primary profile on the common subset (ti - 1 + HMAX <= n): every h uses the same fills
# - SEs across seeds; differences paired per seed
# - GATE 1: own-sample h=60 figures must reproduce the committed ones, else halt
# pre-registered reading rule (local |d| <= 10; each test: mean > 2 SE):
# - TRANSIENT iff (a) peak - a(60) > 2 SE, (b) peak at h <= 5, (c) Q4 decay - Q1 decay > 2 SE
# - PERMANENT iff mean profile non-decreasing h=1..60 and a(60) - a(1) > 2 SE
# - both or neither: no estimate quoted
# predictions P1-P4 scored separately; analysis only, no committed module modified

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

run_arm = RF.run_arm
wls = RF.wls
wmean = RF.wmean
keeps = RF.keeps

N_BINS = 8
CONTROL_LIFE = 720.0

HORIZONS = (1, 2, 5, 10, 20, 30, 60, 120, 300)
HMAX = max(HORIZONS)
H_COMMITTED = 60
WINDOWS = (5.0, 10.0, 20.0, 40.0)
DECISION_WINDOW = 10.0
MIN_WIN_N = 30                    # below: THIN
N_QUART = 4

G1 = RF.G1
G1_TOL = RF.G1_TOL
G1_AGREE_4DP = RF.G1_AGREE_4DP
G1_THR = RF.G1_THR
G1_TOL_4DP = RF.G1_TOL_4DP
G1_OCTILES = RF.G1_OCTILES
G1_TOL_OCT = RF.G1_TOL_OCT
# anchor_test_720.txt Test B, |d| <= 10, a60_m0 (sw)
G1_LOCAL10 = 6.290
G1_TOL_LOCAL = 0.0005

SE_MULT = 2.0
PEAK_MAX_H = 5

SELFTEST_SEED = 0
# must give non-zero fills (3600 gives none)
SELFTEST_SHORT = 86400.0

NAN = float("nan")


# --- estimators ---

def adv(ps, m0, ti, h, mid, n):
    """Committed adverse move at horizon h; None past the run end."""
    j = ti - 1 + h
    if j > n:
        return None
    return -ps * (mid[j] - m0)


def umean(vals):
    """Unweighted mean; NAN on empty."""
    return statistics.mean(vals) if vals else NAN


def argmax_h(prof):
    """Argmax over HORIZONS of prof {h: value}; NAN-safe, ties to smaller h."""
    best_h, best_v = None, None
    for h in HORIZONS:
        v = prof.get(h)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if best_v is None or v > best_v:
            best_h, best_v = h, v
    return best_h, best_v


def quartile_decay(per, hp):
    """Per-seed (Q4 decay) - (Q1 decay), decay from hp (mean-profile argmax) to H_COMMITTED."""
    assert hp in HORIZONS, "quartile_decay got a non-horizon hp: %r" % hp
    out = []
    for p in per:
        top = (p["q%d_h%d_w" % (N_QUART - 1, hp)]
               - p["q%d_h%d_w" % (N_QUART - 1, H_COMMITTED)])
        bot = p["q0_h%d_w" % hp] - p["q0_h%d_w" % H_COMMITTED]
        out.append(top - bot)
    return out


# --- per-seed analysis ---

def analyse_profile(seed, life):
    """One seed -> flat dict of scalars."""
    fills, mid, extras = run_arm(seed, life)
    n = extras["n"]
    nf = len(fills)
    out = {"seed": float(seed), "n_fills": float(nf),
           "n_post_fb": float(extras["n_post_fb"])}
    if nf == 0:
        return out

    # carry-forward seconds (upper bound)
    ncarry = 0
    for i in range(2, n + 1):
        if mid[i] == mid[i - 1]:
            ncarry += 1
    out["n_carry_ub"] = float(ncarry)
    out["carry_pct_ub"] = 100.0 * ncarry / max(1, n - 1)

    rows = []
    for (ti, ps, sz, m0, _s_post, _v, st) in fills:
        d = -ps * st
        a_own = {}
        for h in HORIZONS:
            a_own[h] = adv(ps, m0, ti, h, mid, n)
        in_common = (ti - 1 + HMAX) <= n
        rows.append((d, sz, st, a_own, in_common))

    common = [r for r in rows if r[4]]
    out["n_common"] = float(len(common))
    out["common_pct"] = 100.0 * len(common) / nf

    # GATE 1 quantities: own sample at h=60, as committed
    y60 = [r for r in rows if r[3][H_COMMITTED] is not None]
    out["n_with_a60"] = float(len(y60))
    out["adv60"] = wmean([(r[3][H_COMMITTED], r[1]) for r in y60])
    out["wrong"] = 100.0 * sum(1 for r in rows if r[0] > 0) / nf
    thr = LS.pct(sorted(abs(f[6]) for f in fills), 0.20)
    out["thr"] = thr
    b60 = [r for r in y60 if keeps(r[2], thr)]
    out["n_in_bucket"] = float(len(b60))
    out["agree60"] = wmean([(r[3][H_COMMITTED], r[1]) for r in b60]) \
        if b60 else NAN
    loc60 = [r for r in y60 if abs(r[0]) <= DECISION_WINDOW]
    out["gate_local10"] = wmean([(r[3][H_COMMITTED], r[1]) for r in loc60]) \
        if loc60 else NAN

    srt = sorted(rows, key=lambda r: r[0])
    m = len(srt)
    for bidx in range(N_BINS):
        lo, hi = int(m * bidx / N_BINS), int(m * (bidx + 1) / N_BINS)
        chunk = srt[lo:hi]
        ok = [c for c in chunk if c[3][H_COMMITTED] is not None]
        out["oct_d_%d" % bidx] = umean([c[0] for c in chunk])
        out["oct_a_%d" % bidx] = (
            wmean([(c[3][H_COMMITTED], c[1]) for c in ok]) if ok else NAN)

    # A: overall a(h), common subset and own sample
    for h in HORIZONS:
        sub = [r for r in common if r[3][h] is not None]
        out["ovr_c_h%d" % h] = wmean([(r[3][h], r[1]) for r in sub]) \
            if sub else NAN
        own = [r for r in rows if r[3][h] is not None]
        out["ovr_o_h%d" % h] = wmean([(r[3][h], r[1]) for r in own]) \
            if own else NAN
        out["n_own_h%d" % h] = float(len(own))

    # B: local windows, common subset, weighted and unweighted
    for w in WINDOWS:
        wk = "win%d" % int(w)
        sub_w = [r for r in common if abs(r[0]) <= w]
        out["%s_n" % wk] = float(len(sub_w))
        out["%s_d_u" % wk] = umean([r[0] for r in sub_w])
        for h in HORIZONS:
            ok = [r for r in sub_w if r[3][h] is not None]
            out["%s_h%d_w" % (wk, h)] = wmean([(r[3][h], r[1]) for r in ok]) \
                if ok else NAN
            out["%s_h%d_u" % (wk, h)] = umean([r[3][h] for r in ok]) \
                if ok else NAN

    # C: agree-bucket at every horizon, committed bucket definition
    buck_c = [r for r in common if keeps(r[2], thr)]
    out["n_bucket_common"] = float(len(buck_c))
    for h in HORIZONS:
        ok = [r for r in buck_c if r[3][h] is not None]
        out["agree_c_h%d" % h] = wmean([(r[3][h], r[1]) for r in ok]) \
            if ok else NAN
        ovr = out["ovr_c_h%d" % h]
        out["share_c_h%d" % h] = (100.0 * out["agree_c_h%d" % h] / ovr) \
            if ovr not in (0.0, None) and not math.isnan(ovr) else NAN

    # D: local |d| <= DECISION_WINDOW by fill-size quartile
    dw = [r for r in common if abs(r[0]) <= DECISION_WINDOW]
    bysz = sorted(dw, key=lambda r: r[1])
    q = len(bysz)
    for qi in range(N_QUART):
        lo, hi = int(q * qi / N_QUART), int(q * (qi + 1) / N_QUART)
        chunk = bysz[lo:hi]
        out["q%d_n" % qi] = float(len(chunk))
        out["q%d_size" % qi] = umean([c[1] for c in chunk])
        for h in HORIZONS:
            ok = [c for c in chunk if c[3][h] is not None]
            out["q%d_h%d_w" % (qi, h)] = wmean([(c[3][h], c[1]) for c in ok]) \
                if ok else NAN

    # paired per-seed quantities the reading rule consumes
    dk = "win%d" % int(DECISION_WINDOW)
    prof = dict((h, out["%s_h%d_w" % (dk, h)]) for h in HORIZONS)
    ph, pv = argmax_h(prof)
    out["peak_h"] = float(ph) if ph is not None else NAN
    out["peak_val"] = pv if pv is not None else NAN
    out["peak_minus_h60"] = (pv - prof[H_COMMITTED]) if pv is not None else NAN
    out["h60_minus_h1"] = prof[H_COMMITTED] - prof[1]
    upto = [h for h in HORIZONS if h <= H_COMMITTED]
    mono = all(prof[upto[i + 1]] >= prof[upto[i]] for i in range(len(upto) - 1))
    out["mono_up_to_h60"] = 1.0 if mono else 0.0

    out["sd_d"] = statistics.stdev([r[0] for r in rows]) if len(rows) > 1 \
        else NAN
    return out


def measure(life):
    """All seeds -> ({key: (mean, se, k)}, per-seed)."""
    per = []
    t0 = time.time()
    for s in SEEDS:
        x = analyse_profile(s, life)
        per.append(x)
        print("    seed %d done (%.0fs elapsed)  n_fills %d, common subset %d "
              "(%.2f pct), carry-forward UB %.2f pct"
              % (s, time.time() - t0, int(x.get("n_fills", 0)),
                 int(x.get("n_common", 0)), x.get("common_pct", float("nan")),
                 x.get("carry_pct_ub", float("nan"))), flush=True)
    keys = set()
    for p in per:
        keys.update(p.keys())
    res = {}
    for k in sorted(keys):
        res[k] = mean_se([p[k] for p in per if k in p])
    res["_secs"] = (time.time() - t0, 0.0, len(per))
    res["_nseed"] = (float(len(per)), 0.0, len(per))
    return res, per


# --- self-tests: negative controls first, then machinery ---

def selftest():
    print("=" * 126)
    print("Self-test")
    print("=" * 126)

    print("  T_ID  run_arm is RF.run_arm: %s   wls: %s   wmean: %s   keeps: %s"
          % (run_arm is RF.run_arm, wls is RF.wls, wmean is RF.wmean,
             keeps is RF.keeps))
    assert run_arm is RF.run_arm and wls is RF.wls
    assert wmean is RF.wmean and keeps is RF.keeps

    nn = 1000
    cmid = [62000.0] * (nn + 2)
    worst = 0.0
    ncmp = 0
    for ps in (-1.0, 1.0):
        for ti in (1, 7, 400):
            for h in HORIZONS:
                a = adv(ps, cmid[ti - 1], ti, h, cmid, nn)
                if a is None:
                    continue
                worst = max(worst, abs(a))
                ncmp += 1
    print("  NC_CONST  constant mid -> a(h) over %d cases: worst |a| = %.3e "
          "(must be exactly 0)" % (ncmp, worst))
    assert ncmp > 0, "NC_CONST checked nothing"
    assert worst == 0.0, "constant mid gave a non-zero adverse move: %r" % worst

    slope = 0.037
    base = 62000.0
    lmid = [base + slope * i for i in range(nn + 2)]
    worst = 0.0
    ncmp = 0
    for ps in (-1.0, 1.0):
        for ti in (1, 7, 400):
            for h in HORIZONS:
                a = adv(ps, lmid[ti - 1], ti, h, lmid, nn)
                if a is None:
                    continue
                worst = max(worst, abs(a - (-ps * slope * h)))
                ncmp += 1
    print("  NC_LIN    linear mid (slope %.3f) -> a(h) vs -ps*slope*h over %d "
          "cases: worst error %.3e" % (slope, ncmp, worst))
    assert ncmp > 0, "NC_LIN checked nothing"
    assert worst < 1e-9, "linear mid gave a non-linear profile: %r" % worst

    b1 = adv(1.0, 0.0, nn - HMAX + 1, HMAX, cmid, nn)
    b2 = adv(1.0, 0.0, nn - HMAX + 2, HMAX, cmid, nn)
    print("  NC_OFFEND j == n -> %r (must be a number)   j == n+1 -> %r "
          "(must be None)" % (b1, b2))
    assert b1 is not None and b2 is None

    planted = dict((h, 0.0) for h in HORIZONS)
    planted[2] = 9.0
    planted[60] = 4.0
    gh, gv = argmax_h(planted)
    print("  NC_ARGMAX planted peak at h=2 -> argmax_h returns h=%r value=%r"
          % (gh, gv))
    assert gh == 2 and gv == 9.0
    nanprof = dict((h, float("nan")) for h in HORIZONS)
    nanprof[10] = 1.0
    assert argmax_h(nanprof)[0] == 10, "argmax_h is not NAN-safe"

    print("  NC_SPOST  not applicable (s_post unused)")

    f1, m1, e1 = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    assert len(f1) > 0, ("zero fills at "
                         "horizon=%.0f" % SELFTEST_SHORT)
    ns = e1["n"]
    ncommon = sum(1 for r in f1 if (r[0] - 1 + HMAX) <= ns)
    print("  NC_SUBSET %d fills at horizon=%.0f, common subset (ti-1+%d <= %d) "
          "holds %d" % (len(f1), SELFTEST_SHORT, HMAX, ns, ncommon))
    assert ncommon > 0, "common subset empty; profile undefined"

    f2, m2, e2 = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    same = (f1 == f2) and (m1 == m2) and (e1 == e2)
    print("  T_DET     determinism (horizon=%.0fs, %d fills, %d mid entries)   "
          "identical: %s" % (SELFTEST_SHORT, len(f1), len(m1), same))
    assert same
    names = ("ti", "ps", "size", "m0", "s_post", "v", "st")
    nfld = 0
    for r1, r2 in zip(f1, f2):
        for idx, _nm in enumerate(names):
            assert r1[idx] == r2[idx], "T_DET field %s differs" % _nm
            nfld += 1
    print("            %d per-field comparisons, all equal" % nfld)

    print("  NC_KEEPS  keeps(thr,thr)=%s  keeps(-thr,thr)=%s  "
          "keeps(thr+eps,thr)=%s" % (keeps(40.0, 40.0), keeps(-40.0, 40.0),
                          keeps(40.0001, 40.0)))
    assert keeps(40.0, 40.0) and keeps(-40.0, 40.0)
    assert not keeps(40.0001, 40.0)

    print("  Self-test PASS.")
    print("")


# --- GATE 1 ---

def gate1(r):
    """Reproduce committed figures; False on any failure."""
    print("=" * 126)
    print("GATE 1: reproduce COMMITTED figures (own sample, h=60)")
    print("=" * 126)
    print("")
    bad = []
    nchk = 0

    def chk(label, got, want, tol, src):
        dif = abs(got - want)
        ok = dif <= tol
        if not ok:
            bad.append(label)
        print("  %-34s got %12.4f   want %12.4f   diff %10.6f   %-4s  %s"
              % (label, got, want, dif, "OK" if ok else "FAIL", src))

    checks = [
        ("adverse move h=60", r["adv60"][0], G1["adv60"], G1_TOL,
         "lifetime_sweep_results.txt:38"),
        ("agree-bucket h=60 (2dp)", r["agree60"][0], G1["agree60"], G1_TOL,
         "lifetime_sweep_results.txt:38"),
        ("agree-bucket h=60 (4dp)", r["agree60"][0], G1_AGREE_4DP, G1_TOL_4DP,
         "staleness_results.txt"),
        ("wrong-side pct", r["wrong"][0], G1["wrong"], G1_TOL,
         "lifetime_sweep_results.txt:38"),
        ("bucket threshold $", r["thr"][0], G1_THR, G1_TOL_4DP,
         "staleness_results.txt"),
        ("local |d|<=10 a60 (sw)", r["gate_local10"][0], G1_LOCAL10,
         G1_TOL_LOCAL, "anchor_test_720.txt Test B"),
    ]
    for i, (wd, wa) in enumerate(G1_OCTILES):
        checks.append(("octile %d mean d" % (i + 1), r["oct_d_%d" % i][0], wd,
                       G1_TOL_OCT, "staleness_results.txt sec.1"))
        checks.append(("octile %d adverse h=60" % (i + 1),
                       r["oct_a_%d" % i][0], wa, G1_TOL_OCT,
                       "staleness_results.txt sec.1"))
    for c in checks:
        chk(*c)
        nchk += 1

    print("")
    if bad:
        print("  GATE 1 FAIL on %d of %d checks: %s"
              % (len(bad), nchk, ", ".join(bad)))
        return False
    print("  GATE 1 PASS. %d checks reproduced." % nchk)
    print("  FLAGGED: residual_fit.py prints 'thirteen'; anchor_test.py computes 21.")
    print("")
    return True


# --- reporting ---

def ms(r, k, dp=4):
    m, se, _k = r[k]
    return "%.*f +/- %.*f" % (dp, m, dp, se)


def val(r, k):
    return r[k][0]


def sef(r, k):
    return r[k][1]


def hdr(label):
    return ("  %-26s" % label) + "".join("%14s" % ("h=%d" % h)
                                         for h in HORIZONS)


def row(label, vals):
    out = "  %-26s" % label
    for v in vals:
        out += "%14s" % ("%.3f" % v if not math.isnan(v) else "--")
    return out


def report_a(r):
    print("=" * 126)
    print("A. overall a(h)")
    print("=" * 126)
    print("   common subset: ti-1+%d <= n" % HMAX)
    print("   subset size %s of %s fills per seed (%s pct)"
          % (ms(r, "n_common", 1), ms(r, "n_fills", 1),
             ms(r, "common_pct", 2)))
    print("")
    print(hdr("a(h) size-weighted"))
    print(row("COMMON SUBSET", [val(r, "ovr_c_h%d" % h) for h in HORIZONS]))
    print(row("  +/- SE", [sef(r, "ovr_c_h%d" % h) for h in HORIZONS]))
    print(row("own sample (secondary)",
              [val(r, "ovr_o_h%d" % h) for h in HORIZONS]))
    print(row("  +/- SE", [sef(r, "ovr_o_h%d" % h) for h in HORIZONS]))
    print(row("own-sample n per seed",
              [val(r, "n_own_h%d" % h) for h in HORIZONS]))
    print("")
    d = [abs(val(r, "ovr_c_h%d" % h) - val(r, "ovr_o_h%d" % h))
         for h in HORIZONS]
    print("  max common-vs-own gap: %.4f $/BTC" % max(d))
    print("")


def report_b(r):
    print("=" * 126)
    print("B. local a(h), windows on signed d")
    print("=" * 126)
    print("   size-weighted vs unweighted in anchor_test:")
    print("   diverge by 7.77 SE at h=60, so both are reported.")

    print("")
    for w in WINDOWS:
        wk = "win%d" % int(w)
        nn = val(r, "%s_n" % wk)
        flag = "   THIN (< %d per seed)" % MIN_WIN_N if nn < MIN_WIN_N else ""
        print("  |d| <= %-4.0f   n = %s per seed   mean d (u) = %s%s"
              % (w, ms(r, "%s_n" % wk, 1), ms(r, "%s_d_u" % wk, 3), flag))
        print(hdr("    size-weighted"))
        print(row("    a(h) sw",
                  [val(r, "%s_h%d_w" % (wk, h)) for h in HORIZONS]))
        print(row("      +/- SE",
                  [sef(r, "%s_h%d_w" % (wk, h)) for h in HORIZONS]))
        print(row("    a(h) unweighted",
                  [val(r, "%s_h%d_u" % (wk, h)) for h in HORIZONS]))
        print(row("      +/- SE",
                  [sef(r, "%s_h%d_u" % (wk, h)) for h in HORIZONS]))
        print("")


def report_c(r):
    print("=" * 126)
    print("C. agree-bucket a(h) and share of overall")
    print("=" * 126)
    print("   bucket: bottom 20 pct of |staleness| per seed")
    print("   bucket holds %s fills per seed on the common subset"
          % ms(r, "n_bucket_common", 1))
    print("")
    print(hdr("agree-bucket a(h) sw"))
    print(row("agree a(h)", [val(r, "agree_c_h%d" % h) for h in HORIZONS]))
    print(row("  +/- SE", [sef(r, "agree_c_h%d" % h) for h in HORIZONS]))
    print(row("share of overall pct",
              [val(r, "share_c_h%d" % h) for h in HORIZONS]))
    print(row("  +/- SE", [sef(r, "share_c_h%d" % h) for h in HORIZONS]))
    print("")
    print("  share = mean of per-seed ratios")
    print("")


def report_d(r):
    print("=" * 126)
    print("D. local |d| <= %.0f a(h) by fill-size "
          "quartile" % DECISION_WINDOW)
    print("=" * 126)

    print("")
    print(hdr("a(h) sw by size quartile"))
    for qi in range(N_QUART):
        print(row("Q%d  n=%.0f size=%.6f"
                  % (qi + 1, val(r, "q%d_n" % qi), val(r, "q%d_size" % qi)),
                  [val(r, "q%d_h%d_w" % (qi, h)) for h in HORIZONS]))
    print("")
    for qi in range(N_QUART):
        print(row("Q%d +/- SE" % (qi + 1),
                  [sef(r, "q%d_h%d_w" % (qi, h)) for h in HORIZONS]))
    print("")


def score_predictions(r, per):
    print("=" * 126)
    print("Pre-registered predictions")
    print("=" * 126)
    dk = "win%d" % int(DECISION_WINDOW)
    ovr = [val(r, "ovr_c_h%d" % h) for h in HORIZONS]
    loc = [val(r, "%s_h%d_w" % (dk, h)) for h in HORIZONS]

    p1_mono = all(ovr[i + 1] >= ovr[i] for i in range(len(ovr) - 1))
    print("  P1  overall a(h) rises monotonically (plumbing check)")
    print("      a(h) = %s" % ", ".join("%.3f" % v for v in ovr))
    print("      monotonically non-decreasing over all %d horizons ? %s"
          % (len(HORIZONS), p1_mono))
    print("      P1: %s" % ("HELD" if p1_mono else "REFUTED"))
    print("")

    ph, pv = argmax_h(dict(zip(HORIZONS, loc)))
    p2_decays = (pv is not None and ph is not None
                 and ph <= PEAK_MAX_H
                 and loc[HORIZONS.index(H_COMMITTED)] < pv)
    print("  P2  THE REAL PREDICTION. LOCAL |d| <= %.0f a(h) peaks at or near "
          "h=1 near 12.7, then DECAYS to" % DECISION_WINDOW)
    print("      about 6.29 at h=60")
    print("      a(h) = %s" % ", ".join("%.3f" % v for v in loc))
    print("      peak at h=%s, value %.4f;  a(60) = %.4f"
          % (ph, pv if pv is not None else float("nan"),
             loc[HORIZONS.index(H_COMMITTED)]))
    print("      P2: %s" % ("HELD" if p2_decays else "REFUTED"))
    print("")

    hp = ph if ph is not None else H_COMMITTED
    gm, gse, _k = mean_se(quartile_decay(per, hp))
    p3 = gm > SE_MULT * gse if gse > 0 else False
    print("  P3  decay steeper in Q4 than Q1")
    print("      decay from h=%d to h=%d" % (hp, H_COMMITTED))
    print("      per-seed argmax mean %s"
          % ms(r, "peak_h", 2))
    print("      (Q4 decay) - (Q1 decay), paired per seed = %.4f +/- %.4f, "
          "needs > %.4f" % (gm, gse, SE_MULT * gse))
    print("      P3: %s" % ("HELD" if p3 else "REFUTED"))
    print("")

    i60 = HORIZONS.index(H_COMMITTED)
    tail = [loc[i] for i in range(i60, len(HORIZONS))]
    p4_decay = all(tail[i + 1] <= tail[i] for i in range(len(tail) - 1))
    print("  P4  local a(h) keeps decaying beyond h=60 (low confidence)")
    print("      a(h) for h >= 60 = %s"
          % ", ".join("%.3f" % v for v in tail))
    print("      non-increasing beyond h=60 ? %s" % p4_decay)
    print("      P4: %s" % ("HELD" if p4_decay else "REFUTED"))
    print("")
    return {"P1": p1_mono, "P2": p2_decays, "P3": p3, "P4": p4_decay}


def decide(r, per):
    print("=" * 126)
    print("Reading rule")
    print("=" * 126)
    dk = "win%d" % int(DECISION_WINDOW)
    loc = [val(r, "%s_h%d_w" % (dk, h)) for h in HORIZONS]
    ph, pv = argmax_h(dict(zip(HORIZONS, loc)))

    pm, pm_se, _k = r["peak_minus_h60"]
    c_a = (pm > SE_MULT * pm_se) if pm_se > 0 else False
    c_b = (ph is not None and ph <= PEAK_MAX_H)

    hp = ph if ph is not None else H_COMMITTED
    gm, gse, _k = mean_se(quartile_decay(per, hp))
    c_c = (gm > SE_MULT * gse) if gse > 0 else False

    hm, hm_se, _k = r["h60_minus_h1"]
    # mean profile, not per-seed monotonicity: equal burden for both branches
    upto = [h for h in HORIZONS if h <= H_COMMITTED]
    lup = [loc[HORIZONS.index(h)] for h in upto]
    p_mono = all(lup[i + 1] >= lup[i] for i in range(len(lup) - 1))
    mono_frac = val(r, "mono_up_to_h60")
    p_rise = (hm > SE_MULT * hm_se) if hm_se > 0 else False

    print("  window |d| <= %.0f: %s fills per seed "
          "(THIN < %d)"
          % (DECISION_WINDOW, ms(r, "%s_n" % dk, 1), MIN_WIN_N))
    print("  peak h=%s, value %.4f; per-seed peak h "
          "mean %s" % (ph, pv if pv is not None else float("nan"),
                       ms(r, "peak_h", 2)))
    print("")
    print("  TRANSIENT requires all three:")
    print("    (a) peak exceeds a(60) by more than %.0f SE, paired per seed"
          % SE_MULT)
    print("          %.4f +/- %.4f   needs > %.4f   -> (a) %s"
          % (pm, pm_se, SE_MULT * pm_se, c_a))
    print("    (b) the peak occurs at h <= %d" % PEAK_MAX_H)
    print("          mean-profile peak h = %s   -> (b) %s" % (ph, c_b))
    print("    (c) top-quartile decay exceeds bottom-quartile decay by more "
          "than %.0f SE, paired per seed" % SE_MULT)
    print("          %.4f +/- %.4f   needs > %.4f   -> (c) %s"
          % (gm, gse, SE_MULT * gse, c_c))
    print("")
    print("  PERMANENT requires both:")
    print("    mean profile non-decreasing h=1..60")
    print("          a(h) for h <= %d = %s"
          % (H_COMMITTED, ", ".join("%.3f" % v for v in lup)))
    print("          mean profile non-decreasing ? -> %s" % p_mono)
    print("          diagnostic: %.1f of %d seeds "
          "monotone"
          % (mono_frac * int(val(r, "_nseed")), int(val(r, "_nseed"))))
    print("    a(60) - a(1) clears %.0f SE above zero, paired per seed"
          % SE_MULT)
    print("          %.4f +/- %.4f   needs > %.4f   -> %s"
          % (hm, hm_se, SE_MULT * hm_se, p_rise))
    print("")

    transient = c_a and c_b and c_c
    permanent = p_mono and p_rise
    if transient and not permanent:
        print("  branch TRANSIENT fires: local component reverts (60 s loss is mark-to-market)")
        print("    local a(h) peak  h=%s : %.4f $/BTC" % (ph, pv))
        print("    local a(60)            : %s $/BTC"
              % ms(r, "%s_h%d_w" % (dk, H_COMMITTED)))
        print("    local a(300)           : %s $/BTC"
              % ms(r, "%s_h%d_w" % (dk, HMAX)))
        return
    if permanent and not transient:
        print("  branch PERMANENT fires")
        print("  rise > %.0f SE; local component does not "
              "revert" % SE_MULT)
        print("    local a(1)   : %s $/BTC" % ms(r, "%s_h1_w" % dk))
        print("    local a(60)  : %s $/BTC"
              % ms(r, "%s_h%d_w" % (dk, H_COMMITTED)))
        return
    print("  no clean branch")
    print("    TRANSIENT (a)/(b)/(c) = %s/%s/%s  -> all three ? %s"
          % (c_a, c_b, c_c, transient))
    print("    PERMANENT monotone/rise = %s/%s  -> both ? %s"
          % (p_mono, p_rise, permanent))
    print("  no estimate quoted")


def limitations(r):
    print("")
    print("=" * 126)
    print("Limitations")
    print("=" * 126)
    print("  L1 common subset excludes last %d s: "
          "%s of %s per seed (%s pct)"
          % (HMAX - 1, ms(r, "n_common", 1), ms(r, "n_fills", 1),
             ms(r, "common_pct", 2)))
    print("  L2 sd(d) = "
          "%s; local windows are narrow" % ms(r, "sd_d", 2))
    print("     THIN below %d fills per seed"
          % MIN_WIN_N)
    print("  L3 quartiles cut per seed")
    print("  L4 SEs across seeds; differences paired per seed")
    print("  L5 one arm, life=720, %d seeds, "
          "simulation only" % int(val(r, "_nseed")))
    print("  L6 longest horizon h=%d" % HMAX)
    print("  L7 s_post unused (anchor m0)")
    print("  L8 carry-forward seconds (upper bound): %s pct"
          % ms(r, "carry_pct_ub", 3))


def main():
    args = sys.argv[1:]
    selftest()
    if "--selftest" in args:
        return
    print("running life=%.0f, %d seeds, horizons %s ..."
          % (CONTROL_LIFE, len(SEEDS), ",".join(str(h) for h in HORIZONS)),
          flush=True)
    r, per = measure(CONTROL_LIFE)
    print("  arm done in %.0fs" % val(r, "_secs"))
    print("")
    print("  n_fills %s   n_with_a60 %s   n_in_bucket %s   common subset %s"
          % (ms(r, "n_fills", 2), ms(r, "n_with_a60", 2),
             ms(r, "n_in_bucket", 2), ms(r, "n_common", 2)))
    print("")
    if not gate1(r):
        sys.exit(1)
    report_a(r)
    report_b(r)
    report_c(r)
    report_d(r)
    score_predictions(r, per)
    decide(r, per)
    limitations(r)


if __name__ == "__main__":
    main()
