# vhat_test.py: can the maker estimate V from the book it observes? (analysis only)

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
import sim_run
from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from damage_ab import T, SEEDS, K, REF_MID, GAMMA, mean_se
from clipped_mm_check import MM_ID

run_arm = RF.run_arm
wls = RF.wls
wmean = RF.wmean
keeps = RF.keeps

SEED_ID = "seed"
CONTROL_LIFE = 720.0
H_SHORT = LS.H_SHORT

SAMPLE_EVERY = 60

NEAR_BP = (10.0, 50.0)
TOPK = 3
AVG_W = 300

REF_ED = 63.4072
REF_H = 9.1810

G1 = RF.G1
G1_TOL = RF.G1_TOL
G1_AGREE_4DP = RF.G1_AGREE_4DP
G1_THR = RF.G1_THR
G1_TOL_4DP = RF.G1_TOL_4DP

SE_MULT = 2.0
HALF = 0.5

SELFTEST_SEED = 0
SELFTEST_SHORT = 86400.0

NAN = float("nan")

ESTIMATORS = [
    ("E0", "engine.mid() -- committed anchor", False),
    ("E1", "residual-book midpoint (best non-MM/seed)", False),
    ("E2a", "depth-wtd mid, <=%.0fbp" % NEAR_BP[0], True),
    ("E2b", "depth-wtd mid, <=%.0fbp" % NEAR_BP[1], True),
    ("E3", "centroid of ALL non-MM/seed orders", True),
    ("E4", "time-average of E0 over %ds (NULL)" % AVG_W, False),
    ("E5a", "touch-excluded midpoint (2nd best)", False),
    ("E5b", "top-%d-excluded midpoint" % TOPK, False),
]
EST_KEYS = [e[0] for e in ESTIMATORS]
WEIGHTED = [e[0] for e in ESTIMATORS if e[2]]


def residual_levels(eng):
    """(bid_levels, ask_levels), each [(price, size)], non-MM and non-seed"""
    bl, al = [], []
    for book, store in ((eng.bids, bl), (eng.asks, al)):
        for p, q in book.items():
            s = 0.0
            for o in q:
                if o.agent_id == MM_ID or o.agent_id == SEED_ID:
                    continue
                s += o.size
            if s > 0.0:
                store.append((p, s))
    bl.sort(key=lambda x: -x[0])
    al.sort(key=lambda x: x[0])
    return bl, al


def _side_avg(levels, weighted):
    """Size-weighted or unweighted mean price of a list of (price, size)"""
    if not levels:
        return None
    if weighted:
        num = den = 0.0
        for (p, s) in levels:
            num += p * s
            den += s
        return (num / den) if den > 0 else None
    return sum(p for (p, _s) in levels) / len(levels)


def estimate(eng, bl, al, mid_hist, ti):
    """Every candidate estimator of V, from observables only"""
    v = {}
    n = {}

    v["E0"] = eng.mid()
    n["E0"] = 2 if v["E0"] is not None else 0

    if bl and al:
        v["E1"] = 0.5 * (bl[0][0] + al[0][0])
        n["E1"] = 2
    else:
        v["E1"] = None
        n["E1"] = 0

    for tag, bp in (("E2a", NEAR_BP[0]), ("E2b", NEAR_BP[1])):
        if v["E1"] is None:
            v[tag] = None
            v[tag + "_u"] = None
            n[tag] = 0
            continue
        c = v["E1"]
        lo, hi = c * (1.0 - bp * 1e-4), c * (1.0 + bp * 1e-4)
        sb = [x for x in bl if x[0] >= lo]
        sa = [x for x in al if x[0] <= hi]
        bw, aw = _side_avg(sb, True), _side_avg(sa, True)
        bu, au = _side_avg(sb, False), _side_avg(sa, False)
        v[tag] = 0.5 * (bw + aw) if (bw is not None and aw is not None) else None
        v[tag + "_u"] = (0.5 * (bu + au)
                         if (bu is not None and au is not None) else None)
        n[tag] = len(sb) + len(sa)

    allv = bl + al
    v["E3"] = _side_avg(allv, True)
    v["E3_u"] = _side_avg(allv, False)
    n["E3"] = len(allv)

    lo = max(1, ti - AVG_W + 1)
    win = [mid_hist[i] for i in range(lo, ti + 1) if mid_hist[i] > 0.0]
    v["E4"] = (sum(win) / len(win)) if win else None
    n["E4"] = len(win)

    if len(bl) >= 2 and len(al) >= 2:
        v["E5a"] = 0.5 * (bl[1][0] + al[1][0])
        n["E5a"] = 2
    else:
        v["E5a"] = None
        n["E5a"] = 0
    if len(bl) > TOPK and len(al) > TOPK:
        v["E5b"] = 0.5 * (bl[TOPK][0] + al[TOPK][0])
        n["E5b"] = 2
    else:
        v["E5b"] = None
        n["E5b"] = 0

    return v, n


def run_sampled(seed, life, horizon, fill_seconds):
    """Mirrors residual_fit.run_arm EXACTLY, and additionally samples the book"""
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
    uniform = []
    prev = {}

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

        want_u = (ti % SAMPLE_EVERY == 0)
        want_p = ((ti + 1) in fill_seconds)
        if want_u or want_p:
            bl, al = residual_levels(eng)
            est, cnt = estimate(eng, bl, al, mid, ti)
            if want_u:
                uniform.append((v, est, cnt, len(bl), len(al)))
            if want_p:
                prev[ti] = est

    extras = {"pnl": mm.mark_to_market(last), "spread": spread_pnl,
              "n_post_fb": n_post_fb, "n": n}
    return recs_out, mid, extras, uniform, prev


def analyse_vhat(seed, life):
    """One seed, two passes"""
    fills, mid1, ex1 = run_arm(seed, life)
    fill_secs = set(f[0] for f in fills)
    recs, mid2, ex2, uniform, prev = run_sampled(seed, life, T, fill_secs)
    assert recs == fills, ("PASS 2 DID NOT REPRODUCE PASS 1's FILLS on seed %d "
                           "-- the mirrored loop is NOT the same world" % seed)
    assert mid2 == mid1, ("PASS 2 DID NOT REPRODUCE PASS 1's MID ARRAY on seed "
                          "%d" % seed)

    n = ex1["n"]
    out = {"seed": float(seed), "n_fills": float(len(fills)),
           "n_uniform": float(len(uniform)), "n_prev": float(len(prev))}

    y60 = [f for f in fills if (f[0] - 1 + H_SHORT) <= n]
    out["adv60"] = wmean([(-f[1] * (mid1[f[0] - 1 + H_SHORT] - f[3]), f[2])
                          for f in y60])
    out["wrong"] = 100.0 * sum(1 for f in fills if -f[1] * f[6] > 0) / len(fills)
    thr = LS.pct(sorted(abs(f[6]) for f in fills), 0.20)
    out["thr"] = thr
    b60 = [f for f in y60 if keeps(f[6], thr)]
    out["agree60"] = wmean([(-f[1] * (mid1[f[0] - 1 + H_SHORT] - f[3]), f[2])
                            for f in b60]) if b60 else NAN
    out["Ed_committed_w"] = wmean([(-f[1] * f[6], f[2]) for f in fills])
    out["Ed_committed_u"] = statistics.mean([-f[1] * f[6] for f in fills])

    keys = list(EST_KEYS) + [k + "_u" for k in WEIGHTED]
    for k in keys:
        errs = []
        for (v, est, _c, _nb, _na) in uniform:
            e = est.get(k)
            if e is None:
                continue
            errs.append(e - v)
        out["n_ok_%s" % k] = float(len(errs))
        if not errs:
            for suf in ("med", "mean", "sd", "signed"):
                out["%s_%s" % (suf, k)] = NAN
            continue
        aerr = [abs(x) for x in errs]
        out["med_%s" % k] = statistics.median(aerr)
        out["mean_%s" % k] = statistics.mean(aerr)
        out["sd_%s" % k] = statistics.stdev(errs) if len(errs) > 1 else NAN
        out["signed_%s" % k] = statistics.mean(errs)

    for k in keys:
        ncl = nfl = ntot = 0
        for (v, est, _c, _nb, _na) in uniform:
            e, e0 = est.get(k), est.get("E0")
            if e is None or e0 is None:
                continue
            ntot += 1
            if abs(e - v) < abs(e0 - v):
                ncl += 1
            if (e - v) * (e0 - v) < 0.0:
                nfl += 1
        out["closer_%s" % k] = (100.0 * ncl / ntot) if ntot else NAN
        out["flip_%s" % k] = (100.0 * nfl / ntot) if ntot else NAN

    for k in EST_KEYS:
        cs = [c.get(k, 0) for (_v, _e, c, _nb, _na) in uniform]
        out["N_%s" % k] = statistics.mean(cs) if cs else NAN
    out["n_bid_levels"] = statistics.mean(
        [nb for (_v, _e, _c, nb, _na) in uniform]) if uniform else NAN
    out["n_ask_levels"] = statistics.mean(
        [na for (_v, _e, _c, _nb, na) in uniform]) if uniform else NAN

    for k in keys:
        num = den = 0.0
        vals = []
        nwrong = ntot = 0
        for (ti, ps, sz, _m0, _sp, vv, _st) in fills:
            est = prev.get(ti - 1)
            if est is None:
                continue
            e = est.get(k)
            if e is None:
                continue
            d = -ps * (vv - e)
            num += d * sz
            den += sz
            vals.append(d)
            ntot += 1
            if d > 0:
                nwrong += 1
        out["Ed_w_%s" % k] = (num / den) if den > 0 else NAN
        out["Ed_u_%s" % k] = statistics.mean(vals) if vals else NAN
        out["wrong_%s" % k] = (100.0 * nwrong / ntot) if ntot else NAN
        out["n_fill_ok_%s" % k] = float(ntot)

    m0 = out.get("med_E0", NAN)
    d0 = out.get("Ed_w_E0", NAN)
    m4 = out.get("med_E4", NAN)
    for k in keys:
        out["beatE0_%s" % k] = HALF * m0 - out.get("med_%s" % k, NAN)
        out["beatEd_%s" % k] = d0 - out.get("Ed_w_%s" % k, NAN)
        out["beatE4_%s" % k] = m4 - out.get("med_%s" % k, NAN)
    for k in WEIGHTED:
        out["wu_gap_%s" % k] = (out.get("med_%s" % k, NAN)
                                - out.get("med_%s_u" % k, NAN))
    return out


def measure(life):
    """All seeds, one arm"""
    per = []
    t0 = time.time()
    for s in SEEDS:
        x = analyse_vhat(s, life)
        per.append(x)
        print("    seed %d done (%.0fs elapsed)  fills %d, uniform samples %d, "
              "fill-adjacent samples %d, levels %.0f/%.0f per side"
              % (s, time.time() - t0, int(x["n_fills"]), int(x["n_uniform"]),
                 int(x["n_prev"]), x["n_bid_levels"], x["n_ask_levels"]),
              flush=True)
    keys = set()
    for p in per:
        keys.update(p.keys())
    res = {}
    for k in sorted(keys):
        res[k] = mean_se([p[k] for p in per if k in p])
    res["_secs"] = (time.time() - t0, 0.0, len(per))
    res["_nseed"] = (float(len(per)), 0.0, len(per))
    return res, per


class _FakeOrder(object):
    def __init__(self, price, size, agent_id):
        self.price = price
        self.size = size
        self.agent_id = agent_id


class _FakeEng(object):
    """Minimal stand-in exposing exactly what residual_levels/estimate read"""

    def __init__(self, bids, asks):
        self.bids = bids
        self.asks = asks

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid(self):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)


def _book(bid_specs, ask_specs):
    b = {}
    a = {}
    for (p, s, ag) in bid_specs:
        b.setdefault(p, []).append(_FakeOrder(p, s, ag))
    for (p, s, ag) in ask_specs:
        a.setdefault(p, []).append(_FakeOrder(p, s, ag))
    return _FakeEng(b, a)


def selftest():
    print("=" * 126)
    print("Self-test. Estimator negative controls, MM/seed exclusion, the")
    print("mirrored-loop equivalence, and determinism.")
    print("=" * 126)

    print("  T_ID     run_arm is RF.run_arm: %s   wls: %s   wmean: %s   "
          "keeps: %s" % (run_arm is RF.run_arm, wls is RF.wls,
                         wmean is RF.wmean, keeps is RF.keeps))
    assert run_arm is RF.run_arm and wls is RF.wls
    assert wmean is RF.wmean and keeps is RF.keeps

    BID_SPEC = [(61990.0, 1.0, "n1"), (61975.0, 3.0, "n2"),
                (61950.0, 2.0, "n3"), (61900.0, 5.0, "n4"),
                (61800.0, 1.0, "n5")]
    ASK_SPEC = [(62020.0, 2.0, "n6"), (62030.0, 1.0, "n7"),
                (62060.0, 4.0, "n8"), (62100.0, 1.0, "n9")]
    eng = _book(BID_SPEC, ASK_SPEC)
    bl, al = residual_levels(eng)
    hist = [0.0] * 20
    for i in range(1, 20):
        hist[i] = 62000.0 + i
    est, cnt = estimate(eng, bl, al, hist, 19)
    vtrue = eng.mid()

    print("  NC_VMID  skewed fixture (5 bids / 4 asks, unequal sizes and "
          "spacing). V := the mid = %.2f." % vtrue)
    print("           E0 err = %.17g   (asserted == 0.0, the only forced "
          "quantity)" % (est["E0"] - vtrue))
    assert est["E0"] - vtrue == 0.0, "E0 error is not exactly zero"
    shown = [k for k in ("E1", "E2a", "E2b", "E3", "E4", "E5a", "E5b")
             if est.get(k) is not None]
    print("           Reported, not asserted: " + ", ".join(
        "%s %.4f" % (k, est[k]) for k in shown))
    coinc = []
    for i in range(len(shown)):
        for j in range(i + 1, len(shown)):
            if est[shown[i]] == est[shown[j]]:
                coinc.append("%s==%s" % (shown[i], shown[j]))
    print("           exact coincidences among the above: %s"
          % (", ".join(coinc) if coinc else "None; all estimators separate"))
    print("           (E0 == E1 = %.2f is STRUCTURAL, not a coincidence: this "
          "book holds no MM or seed" % est["E1"])
    print("            orders, so the residual midpoint is the raw midpoint.)")

    vdis = vtrue - 37.5
    print("  NC_DISP  V displaced to %.2f -> E0 error must be exactly %+.1f: "
          "%+.1f" % (vdis, 37.5, est["E0"] - vdis))
    assert abs((est["E0"] - vdis) - 37.5) < 1e-12

    e2 = _book(BID_SPEC + [(61999.0, 99.0, MM_ID), (61995.0, 99.0, SEED_ID)],
               ASK_SPEC + [(62008.0, 99.0, MM_ID), (62015.0, 99.0, SEED_ID)])
    bl2, al2 = residual_levels(e2)
    est2, _c2 = estimate(e2, bl2, al2, hist, 19)
    print("  NC_EXCL  asymmetric plants: MM bid 61999 / MM ask 62008, seed "
          "61995 / 62015, all 99 BTC,")
    print("           all inside the touch. E0 must move; E1 and E3 must NOT.")
    print("           E0 %.4f -> %.4f   moved: %s"
          % (est["E0"], est2["E0"], est2["E0"] != est["E0"]))
    print("           E1 %.4f -> %.4f   unchanged: %s"
          % (est["E1"], est2["E1"], est["E1"] == est2["E1"]))
    print("           E3 %.4f -> %.4f   unchanged: %s  (99-BTC plants would "
          "dominate a size-weighted"
          % (est["E3"], est2["E3"], est["E3"] == est2["E3"]))
    print("           centroid if they leaked)")
    assert est2["E0"] != est["E0"], "the planted orders did not even move E0"
    assert est["E1"] == est2["E1"], "MM/seed orders LEAKED into E1"
    assert est["E3"] == est2["E3"], "MM/seed orders LEAKED into E3"

    print("  NC_TOUCH E1 = %.4f (best %.0f/%.0f), E5a = %.4f (2nd %.0f/%.0f), "
          "E5b = %.4f (%dth %.0f/%.0f)"
          % (est["E1"], BID_SPEC[0][0], ASK_SPEC[0][0],
             est["E5a"], BID_SPEC[1][0], ASK_SPEC[1][0],
             est["E5b"], TOPK + 1, BID_SPEC[TOPK][0], ASK_SPEC[TOPK][0]))
    assert est["E5a"] == 0.5 * (BID_SPEC[1][0] + ASK_SPEC[1][0])
    assert est["E5b"] == 0.5 * (BID_SPEC[TOPK][0] + ASK_SPEC[TOPK][0])

    e3 = _book([(61990.0, 1.0, "n1")], [])
    bl3, al3 = residual_levels(e3)
    est3, _c3 = estimate(e3, bl3, al3, hist, 19)
    print("  NC_EMPTY one-sided book -> E0 %r, E1 %r, E5a %r (all must be None)"
          % (est3["E0"], est3["E1"], est3["E5a"]))
    assert est3["E0"] is None and est3["E1"] is None and est3["E5a"] is None

    f1, m1, e1x = run_arm(SELFTEST_SEED, CONTROL_LIFE, horizon=SELFTEST_SHORT)
    assert len(f1) > 0, ("T_MIRROR AND T_DET ARE VACUOUS -- zero fills at "
                         "horizon=%.0f. NOT raising the horizon; this needs "
                         "explaining." % SELFTEST_SHORT)
    fs = set(f[0] for f in f1)
    f2, m2, e2x, uni, prv = run_sampled(SELFTEST_SEED, CONTROL_LIFE,
                                        SELFTEST_SHORT, fs)
    print("  T_MIRROR pass1 %d fills vs pass2 %d fills; mid arrays %d vs %d"
          % (len(f1), len(f2), len(m1), len(m2)))
    assert f2 == f1, "the mirrored loop is NOT the same world (fills differ)"
    assert m2 == m1, "the mirrored loop is NOT the same world (mid differs)"
    print("           records and mid array identical; the mirrored loop is "
          "the same world.")
    print("  NC_SAMP  uniform samples %d, fill-adjacent samples %d (both must "
          "be > 0)" % (len(uni), len(prv)))
    assert len(uni) > 0, "the uniform sample set is EMPTY"
    assert len(prv) > 0, "the fill-adjacent sample set is EMPTY"

    f3, m3, e3x, uni2, prv2 = run_sampled(SELFTEST_SEED, CONTROL_LIFE,
                                          SELFTEST_SHORT, fs)
    same = (f3 == f2) and (m3 == m2) and (len(uni2) == len(uni))
    print("  T_DET    run_sampled repeated (horizon=%.0fs, %d fills): "
          "identical: %s" % (SELFTEST_SHORT, len(f3), same))
    assert same
    nb = 0
    for (a, b) in zip(uni, uni2):
        assert a[0] == b[0], "T_DET sample V differs"
        for k in EST_KEYS:
            assert a[1].get(k) == b[1].get(k), "T_DET estimator %s differs" % k
            nb += 1
    print("           %d per-sample estimator comparisons, all equal" % nb)

    print("  NC_KEEPS keeps(thr,thr)=%s keeps(-thr,thr)=%s keeps(thr+eps)=%s"
          % (keeps(40.0, 40.0), keeps(-40.0, 40.0), keeps(40.0001, 40.0)))
    assert keeps(40.0, 40.0) and keeps(-40.0, 40.0)
    assert not keeps(40.0001, 40.0)

    print("  Self-test PASS.")
    print("")


def gate1(r):
    print("=" * 126)
    print("Gate 1. Reproduce the committed figures before reporting anything "
          "NEW. Any failure halts.")
    print("=" * 126)
    bad = []
    nchk = 0

    def chk(label, got, want, tol, src):
        dif = abs(got - want)
        ok = dif <= tol
        if not ok:
            bad.append(label)
        print("  %-32s got %12.4f   want %12.4f   diff %10.6f   %-4s  %s"
              % (label, got, want, dif, "OK" if ok else "FAIL", src))

    for c in (("adverse move h=60", r["adv60"][0], G1["adv60"], G1_TOL,
               "lifetime_sweep_results.txt:38"),
              ("agree-bucket (2dp)", r["agree60"][0], G1["agree60"], G1_TOL,
               "lifetime_sweep_results.txt:38"),
              ("agree-bucket (4dp)", r["agree60"][0], G1_AGREE_4DP,
               G1_TOL_4DP, "staleness_results.txt"),
              ("wrong-side pct", r["wrong"][0], G1["wrong"], G1_TOL,
               "lifetime_sweep_results.txt:38"),
              ("bucket threshold $", r["thr"][0], G1_THR, G1_TOL_4DP,
               "staleness_results.txt")):
        chk(*c)
        nchk += 1

    print("")
    print("  cited, not gated; the exact population behind 00b1773's E[d] is "
          "not recoverable from its prose,")
    print("  so these are printed for comparison and NOT asserted:")
    print("    E[d] size-weighted, all fills   %s   against the cited %.4f"
          % (ms(r, "Ed_committed_w"), REF_ED))
    print("    E[d] unweighted,    all fills   %s"
          % ms(r, "Ed_committed_u"))
    print("    half-spread h                   cited %.4f" % REF_H)
    print("")
    if bad:
        print("  Gate 1 FAIL on %d of %d checks: %s"
              % (len(bad), nchk, ", ".join(bad)))
        print("  HALTING.")
        return False
    print("  Gate 1 PASS. %d checks reproduced (count computed, not a string "
          "literal)." % nchk)
    print("")
    return True


def ms(r, k, dp=4):
    if k not in r:
        return "--"
    m, se, _k = r[k]
    return "%.*f +/- %.*f" % (dp, m, dp, se)


def val(r, k):
    return r[k][0] if k in r else NAN


def report_ab(r):
    print("=" * 126)
    print("A / B. Tracking error against V. Uniform samples only (L7).")
    print("=" * 126)
    print("   Read against the scale arithmetic: one unclipped order sits "
          "~$272.08 from V, the committed mid's")
    print("   median error is $101, and averaging buys ~272/sqrt(N). N is the "
          "mean level count per sample.")
    print("")
    print("  %-42s %18s %18s %18s %18s %9s"
          % ("estimator", "median |err| $", "mean |err| $", "sd(err) $",
             "mean signed err $", "N/sample"))
    for (k, label, w) in ESTIMATORS:
        print("  %-42s %18s %18s %18s %18s %9.1f"
              % (label, ms(r, "med_%s" % k, 2), ms(r, "mean_%s" % k, 2),
                 ms(r, "sd_%s" % k, 2), ms(r, "signed_%s" % k, 3),
                 val(r, "N_%s" % k)))
        if w:
            print("  %-42s %18s %18s %18s %18s"
                  % ("    ^ unweighted variant", ms(r, "med_%s_u" % k, 2),
                     ms(r, "mean_%s_u" % k, 2), ms(r, "sd_%s_u" % k, 2),
                     ms(r, "signed_%s_u" % k, 3)))
    print("")
    print("  A biased estimator is a different problem from a noisy one, which "
          "is why mean signed error is")
    print("  reported beside sd. The C2b midpoint's predicted upward bias is "
          "+$0.9377 (V*disp^2/2).")
    print("  book levels per side: bid %s   ask %s"
          % (ms(r, "n_bid_levels", 1), ms(r, "n_ask_levels", 1)))
    print("")


def report_c(r):
    print("=" * 126)
    print("C. Direction versus magnitude, relative to E0. Uniform samples.")
    print("=" * 126)
    print("   V is the truth, so there is no 'correct side' in the abstract. "
          "The two honest questions are")
    print("   whether the estimator is closer than E0, and whether it sits on "
          "the opposite side of V from E0.")
    print("")
    print("  %-42s %22s %22s" % ("estimator", "pct closer than E0",
                                 "pct opposite side to E0"))
    for (k, label, w) in ESTIMATORS:
        print("  %-42s %22s %22s"
              % (label, ms(r, "closer_%s" % k, 2), ms(r, "flip_%s" % k, 2)))
        if w:
            print("  %-42s %22s %22s"
                  % ("    ^ unweighted variant", ms(r, "closer_%s_u" % k, 2),
                     ms(r, "flip_%s_u" % k, 2)))
    print("")


def report_d(r):
    print("=" * 126)
    print("D. At fill instants. Ehat replaces mid[ti-1] in the committed d.")
    print("=" * 126)
    print("   *** Necessary but not sufficient. *** E[d] here is measured at "
          "the current fill population --")
    print("   the fills the maker got while quoting off the mid. A maker "
          "quoting off a better anchor would")
    print("   quote different prices and get different fills. This cannot "
          "Predict counterfactual performance.")
    print("   It says only whether the anchor error is small enough that a fix "
          "is worth building.")
    print("")
    print("  %-42s %20s %20s %20s" % ("estimator", "E[d] size-wtd $",
                                      "E[d] unweighted $",
                                      "wrong-side pct"))
    for (k, label, w) in ESTIMATORS:
        print("  %-42s %20s %20s %20s"
              % (label, ms(r, "Ed_w_%s" % k, 3), ms(r, "Ed_u_%s" % k, 3),
                 ms(r, "wrong_%s" % k, 2)))
        if w:
            print("  %-42s %20s %20s %20s"
                  % ("    ^ unweighted variant", ms(r, "Ed_w_%s_u" % k, 3),
                     ms(r, "Ed_u_%s_u" % k, 3), ms(r, "wrong_%s_u" % k, 2)))
    print("")
    print("  break-even is E[d] < h, with h cited at %.4f. committed "
          "wrong-side is %s pct." % (REF_H, ms(r, "wrong", 4)))
    print("  an estimator clearing E[d] < h here has not shown that a maker "
          "using it would break even. It has")
    print("  shown that the anchor error is small enough to be worth fixing. "
          "The break-even test belongs to a")
    print("  counterfactual policy this run cannot evaluate.")
    print("")


def report_weighting(r):
    print("=" * 126)
    print("The weighting rule. If the two weightings disagree by more than the "
          "Effect, the effect is not")
    print("established.")
    print("=" * 126)
    print("   Noise sizes are heavy-tailed bootstrapped draws, so a "
          "size-weighted centroid can be dominated")
    print("   by a few large orders. anchor_test_720.txt found the two "
          "weightings diverging by 7.77 SE.")
    print("")
    ok = {}
    for k in WEIGHTED:
        gap, gse, _n = r["wu_gap_%s" % k]
        eff, ese, _n2 = r["beatE0_%s" % k]
        est = abs(gap) <= abs(eff)
        ok[k] = est
        print("    %-6s |med(sw) - med(u)| = %.4f +/- %.4f    effect "
              "(0.5*med_E0 - med) = %.4f +/- %.4f    established: %s"
              % (k, gap, gse, eff, ese, est))
    print("")
    return ok


def score_predictions(r, wok):
    print("=" * 126)
    print("Pre-registered predictions, SCORED. Fixed before any result "
          "existed.")
    print("=" * 126)
    m0 = val(r, "med_E0")
    m1 = val(r, "med_E1")
    m4 = val(r, "med_E4")

    p1 = m1 < HALF * m0
    print("  P1  E1 (residual-book midpoint) tracks V materially better than "
          "E0; median |E1 - V| below")
    print("      half of E0's. Confidence lowered from medium after Step 0's "
          "clipping finding, before any")
    print("      result existed: JOIN clipping reprices marketable orders to "
          "the touch, which is exactly")
    print("      where E1 reads. The scale arithmetic independently predicts "
          "E1 loses (N=2 -> $192 vs $101).")
    print("      median |E0 - V| = %.4f, half = %.4f; median |E1 - V| = %.4f"
          % (m0, HALF * m0, m1))
    print("      P1: %s" % ("HELD" if p1 else "REFUTED"))
    print("")

    dm, dse, _n = r["med_E1"]
    dm0, dse0, _n0 = r["med_E0"]
    wrong0 = val(r, "wrong_E0")
    wrong1 = val(r, "wrong_E1")
    rel_mag = (m0 - m1) / m0 if m0 else NAN
    rel_dir = (wrong0 - wrong1) / wrong0 if wrong0 else NAN
    p2 = rel_mag > rel_dir
    print("  P2  E1's improvement is larger in magnitude than in direction; "
          "|E1 - V| falls more than the")
    print("      wrong-side share does. Low confidence.")
    print("      relative fall in median |err| = %.4f; relative fall in "
          "wrong-side share = %.4f" % (rel_mag, rel_dir))
    print("      P2: %s" % ("HELD" if p2 else "REFUTED"))
    print("")

    gap, gse, _n = r["beatE4_E1"]
    p3 = abs(gap) > SE_MULT * gse if gse > 0 else False
    print("  P3  E4, the plain time-average null candidate, does NOT match E1. "
          "If it does, the book")
    print("      structure is not carrying the information and P1's mechanism "
          "is wrong even if its number")
    print("      is right. Medium confidence, and the prediction most worth "
          "checking.")
    print("      median |E4 - V| = %.4f; median |E1 - V| = %.4f; paired "
          "difference %.4f +/- %.4f" % (m4, m1, gap, gse))
    print("      distinguishable at %.0f SE ? %s" % (SE_MULT, p3))
    print("      P3: %s" % ("HELD" if p3 else "REFUTED"))
    print("")

    best = None
    for k in EST_KEYS:
        if k == "E0":
            continue
        e = val(r, "Ed_w_%s" % k)
        if not math.isnan(e) and (best is None or e < best[1]):
            best = (k, e)
    p4 = best is not None and best[1] < REF_H
    print("  P4  At least one estimator brings E[d] at the current fill "
          "population below h = %.4f." % REF_H)
    print("      Low confidence; E[d] is 6.91x away and there was no basis "
          "for a number.")
    print("      lowest E[d] is %s at %.4f" % (best[0] if best else "--",
                                               best[1] if best else NAN))
    print("      P4: %s" % ("HELD" if p4 else "REFUTED"))
    print("")

    m5a = val(r, "med_E5a")
    m5b = val(r, "med_E5b")
    p5 = (m1 >= HALF * m0) and (min(m5a, m5b) < m1)
    print("  P5  added after Step 0, before any result. E1 is the most "
          "contaminated estimator, not the least:")
    print("      E1 tracks V no better than E0 (or only marginally), while E5 "
          "and the deeper/aggregate")
    print("      estimators do better. This partially inverts P1.")
    print("      median |err|: E0 %.4f, E1 %.4f, E5a %.4f, E5b %.4f, E3 %.4f"
          % (m0, m1, m5a, m5b, val(r, "med_E3")))
    print("      E1 failed to beat half of E0 ? %s;  some E5 beats E1 ? %s"
          % (m1 >= HALF * m0, min(m5a, m5b) < m1))
    print("      P5: %s" % ("HELD" if p5 else "REFUTED"))
    print("")
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5}


def decide(r, wok):
    print("=" * 126)
    print("The reading rule. Both branches carry conditions, equal burden.")
    print("=" * 126)
    print("   A fix is available from the observable book if all of:")
    print("     (a) median |Ehat - V| below half E0's, paired per seed, "
          "clearing %.0f SE" % SE_MULT)
    print("     (b) that SAME Ehat's E[d] at fill instants below E0's by more "
          "than %.0f SE, paired." % SE_MULT)
    print("         (b) Is evidence that the anchor error is small enough to "
          "Be worth fixing. It is not A")
    print("         predicted post-fix E[d]. Every estimator is scored on the "
          "fills the maker got while")
    print("         anchored to the mid; a differently-anchored maker gets "
          "different fills, so no")
    print("         counterfactual is available from this run.")
    print("     (c) it beats E4, the null candidate, by more than %.0f SE on "
          "median |err|, paired --" % SE_MULT)
    print("         so a win attributable to smoothing rather than structure "
          "does not count.")
    print("")
    print("  %-6s %26s %26s %26s %10s"
          % ("est", "(a) 0.5*medE0 - med", "(b) EdE0 - Ed", "(c) medE4 - med",
             "all 3"))
    winners = []
    for k in EST_KEYS:
        if k == "E0":
            continue
        a, ase, _n = r["beatE0_%s" % k]
        b, bse, _n2 = r["beatEd_%s" % k]
        c, cse, _n3 = r["beatE4_%s" % k]
        ca = (a > SE_MULT * ase) if ase > 0 else False
        cb = (b > SE_MULT * bse) if bse > 0 else False
        cc = (c > SE_MULT * cse) if cse > 0 else False
        allok = ca and cb and cc
        if allok:
            winners.append((k, a, b, c))
        print("  %-6s %14.4f+/-%-10.4f %14.4f+/-%-10.4f %14.4f+/-%-10.4f %10s"
              % (k, a, ase, b, bse, c, cse, allok))
    print("")

    any_beat_a = False
    any_beat_d = False
    for k in EST_KEYS:
        if k == "E0":
            continue
        a, ase, _n = r["beatE0_%s" % k]
        b, bse, _n2 = r["beatEd_%s" % k]
        if ase > 0 and a > SE_MULT * ase:
            any_beat_a = True
        if bse > 0 and abs(b) > SE_MULT * bse:
            any_beat_d = True
    no_fix = (not any_beat_a) and (not any_beat_d)

    if winners and not no_fix:
        winners.sort(key=lambda w: val(r, "med_%s" % w[0]))
        k, a, b, c = winners[0]
        est_ok = wok.get(k, True)
        print("  Branch fix-available fires for %d estimator(s): %s"
              % (len(winners), ", ".join(w[0] for w in winners)))
        if not est_ok:
            print("  But the weighting rule blocks it: %s's size-weighted and "
                  "unweighted medians disagree by" % k)
            print("  more than the effect, so the effect is not established. "
                  "No estimate is quoted.")
            return
        print("  Read this as: the anchor error is small enough that a fix "
              "Built from the maker's existing")
        print("  information set is worth building; only how it processes "
              "what it already sees would change.")
        print("  It does not say the fix would deliver these numbers.")
        print("    best (smallest median |err|): %s   median |err| %s against "
              "E0's %s"
              % (k, ms(r, "med_%s" % k, 2), ms(r, "med_E0", 2)))
        print("    E[d] at the current fill population: %s against E0's %s"
              % (ms(r, "Ed_w_%s" % k, 3), ms(r, "Ed_w_E0", 3)))
        print("    NOT A prediction of post-fix E[d]. A differently-anchored "
              "maker quotes differently and")
        print("    therefore gets different fills. No counterfactual is "
              "available from this run. See section D.")
        return
    if no_fix:
        print("  Branch no-fix-this-way fires. Every estimator failed to beat "
              "half of E0's median error at")
        print("  %.0f SE, and every estimator's E[d] is within %.0f SE of "
              "E0's. The observable book does not" % (SE_MULT, SE_MULT))
        print("  carry a materially better V-estimate than the mid the maker "
              "already uses.")
        return
    print("  Neither branch fires cleanly. Stopping rather than picking the "
          "nearer.")
    print("    some estimator beat E0's half-median at %.0f SE ? %s"
          % (SE_MULT, any_beat_a))
    print("    some estimator's E[d] differs from E0's at %.0f SE ? %s"
          % (SE_MULT, any_beat_d))
    print("    estimators clearing all three conditions: %d" % len(winners))
    print("  No estimate is quoted from an ambiguous rule.")


def limitations(r):
    print("")
    print("=" * 126)
    print("Limitations. Carried, not buried.")
    print("=" * 126)
    print("  L1 depth has no accessor. best_bid/best_ask/mid/depth_at are the "
          "whole public query surface, and")
    print("     depth_at does not separate by agent. This file iterates "
          "engine.bids/engine.asks directly --")
    print("     the committed convention (lifetime_sweep.py:156-162), so not a "
          "new violation, but attribute")
    print("     access rather than an API.")
    print("  L2 Per-order clip status is unrecoverable: n_clipped is a flow "
          "counter and run_arm does not")
    print("     return the flow. The book cannot be partitioned into clipped "
          "and unclipped, so E1-vs-E5 is")
    print("     an indirect test of the touch-echo hypothesis. No "
          "instrumentation was added to get it.")
    print("  L3 Section D cannot predict counterfactual performance. Different "
          "quotes would mean different")
    print("     fills; E[d] there is measured at the fill population the mid "
          "anchor produced.")
    print("  L4 E3 combines both sides into one centroid, so depth asymmetry "
          "shifts it regardless of V.")
    print("     Levels per side: bid %s, ask %s."
          % (ms(r, "n_bid_levels", 1), ms(r, "n_ask_levels", 1)))
    print("  L5 SEs are across seeds. Every difference is computed per seed "
          "first.")
    print("  L6 One arm, life=720, %d seeds, one simulated world. Nothing here "
          "is measured on real BTCUSD data." % int(val(r, "_nseed")))
    print("  L7 Uniform samples (every %ds, sections A-C) and fill-adjacent "
          "samples (section D) are never" % SAMPLE_EVERY)
    print("     pooled. %s uniform and %s fill-adjacent per seed."
          % (ms(r, "n_uniform", 1), ms(r, "n_prev", 1)))
    print("  L8 the seed book is still built. clipped_placement.make_world:133 "
          "calls sim_run._seed_book,")
    print("     posting 5 levels a side with agent_id 'seed'. Those orders are "
          "NOT V-priced and are excluded")
    print("     here, as the committed calibration excludes them. Flagged, not "
          "chased, not fixed.")
    print("")
    print("  No committed default changed. Nothing retuned. No sweep. No "
          "repricing. The maker's quoting rule")
    print("  is unchanged. JOIN clipping untouched. No sniffer. No fix built.")


def main():
    args = sys.argv[1:]
    selftest()
    if "--selftest" in args:
        return
    print("running life=%.0f, %d seeds, sampling every %ds ..."
          % (CONTROL_LIFE, len(SEEDS), SAMPLE_EVERY), flush=True)
    r, per = measure(CONTROL_LIFE)
    print("  arm done in %.0fs" % val(r, "_secs"))
    print("")
    if not gate1(r):
        sys.exit(1)
    report_ab(r)
    report_c(r)
    report_d(r)
    wok = report_weighting(r)
    score_predictions(r, wok)
    decide(r, wok)
    limitations(r)


if __name__ == "__main__":
    main()
