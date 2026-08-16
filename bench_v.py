# bench_v.py: PnL decomposition benchmarked against V instead of the mid, two arms (analysis only)

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

ALL_ARMS = [720.0, 60.0]
GATE2_TOL = 1e-6
SELFTEST_SEED = 0
SELFTEST_LIFE = 720.0

G1_TOL_ABS = 0.005
G1_TOL_SHARE = 0.05

WRONG_REF = 75.09
WRONG_REF_SE = 0.39
WRONG_TOL_PTS = 1.2

POST_FALLBACK_WARN = 0.01

REF = {"pnl": -525.18, "spread": 63.83, "inv": -589.01, "inv_share": 90.2}


def run_arm(seed, life):
    """Transcription of lifetime_sweep.py:103-174, instrumented to retain the"""
    _p, vf, eng, noise, informed = make_world(seed, T, LS.LAM, LS.P_MARKET,
                                              life, LS.DISP, "join")
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    n = int(T)
    mid = [0.0] * (n + 2)
    last = REF_MID
    spread_lag = 0.0
    n_post_fb = 0
    recs_out = []

    t = 0.0
    while t < T:
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
                    sgn = -ps
                    spread_lag += sgn * (f.price - last) * f.size
                    bb, ba = eng.best_bid(), eng.best_ask()
                    if bb is not None and ba is not None:
                        s_post = 0.5 * (bb + ba)
                    else:
                        s_post = last
                        n_post_fb += 1
                    recs_out.append((ti, sgn, f.size, f.price, last, s_post, v))
                mm.on_fills(fl, r.side)
        mm.requote(eng, ti)
        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            last = 0.5 * (bb + ba)
        mid[ti] = last

    halfs = []
    qs = []
    for row in mm.log.private_view():
        qs.append(row.true_inventory_q)
        b, a = row.quoted_bid, row.quoted_ask
        if b is None or a is None:
            continue
        halfs.append(0.5 * (a - b))

    v_final = vf(T)
    return {
        "sd_q": statistics.stdev(qs) if len(qs) > 1 else float("nan"),
        "h_median": statistics.median(halfs) if halfs else float("nan"),
        "h_mean": statistics.mean(halfs) if halfs else float("nan"),
        "n_quotes": len(halfs),
        "seed": seed, "life": life, "fills": recs_out, "mid": mid,
        "mid_final": last, "v_final": v_final,
        "q_final": mm.q, "cash": mm.cash,
        "pnl_mid": mm.mark_to_market(last),
        "pnl_v": mm.mark_to_market(v_final),
        "spread_lag_inline": spread_lag,
        "n_post_fb": n_post_fb,
    }


def spread_against(fills, which, mid):
    """Spread = sum_j v_j * (p_j - S_j), committed sign convention"""
    tot = 0.0
    for (ti, sgn, sz, p, s_lag, s_post, v) in fills:
        if which == "lag":
            s = s_lag
        elif which == "now":
            s = mid[ti]
        elif which == "post":
            s = s_post
        elif which == "V":
            s = v
        else:
            raise ValueError(which)
        tot += sgn * (p - s) * sz
    return tot


def gate3_rhs(fills, mid, which="lag"):
    """-sum_j v_j * (V_j - S_j), computed independently of spread_against"""
    tot = 0.0
    nwrong = 0
    for (ti, sgn, sz, p, s_lag, s_post, v) in fills:
        s = s_lag if which == "lag" else (mid[ti] if which == "now" else s_post)
        tot += -(sgn * sz) * (v - s)
        if sgn * (v - s) > 0:
            nwrong += 1
    frac = (100.0 * nwrong / len(fills)) if fills else float("nan")
    return tot, frac


def inv_share(spr, inv):
    tot = abs(spr) + abs(inv)
    return (100.0 * abs(inv) / tot) if tot > 0 else float("nan")


def self_test():
    ok = True

    def rep(name, good, detail=""):
        print("  [%s] %s%s" % ("PASS" if good else "FAIL", name,
                               ("   " + detail) if detail else ""))
        return good

    synth = []
    m = [0.0] * 10
    for i, (sgn, sz, p, s) in enumerate([(1.0, 0.5, 101.0, 100.0),
                                         (-1.0, 0.3, 99.0, 100.0),
                                         (1.0, 1.2, 100.5, 100.25)], start=1):
        m[i] = s
        synth.append((i, sgn, sz, p, s, s, s))
    s_lag = spread_against(synth, "lag", m)
    s_v = spread_against(synth, "V", m)
    rhs, _fr = gate3_rhs(synth, m, "lag")
    good = (s_lag == s_v and rhs == 0.0)
    ok &= rep("T1 NEGATIVE: V==mid gives SPREAD(V)==SPREAD(mid), Gate3 RHS==0",
              good, "spread_lag=%r spread_V=%r rhs=%r" % (s_lag, s_v, rhs))

    synth2 = [(1, 1.0, 2.0, 101.0, 100.0, 100.0, 103.0),
              (2, -1.0, 0.5, 99.0, 100.0, 100.0, 98.0)]
    m2 = [0.0] * 5
    m2[1] = m2[2] = 100.0
    want = -((1.0 * 2.0) * 3.0 + (-1.0 * 0.5) * (-2.0))
    rhs2, fr2 = gate3_rhs(synth2, m2, "lag")
    good2 = (abs(rhs2 - want) < 1e-12 and fr2 == 100.0)
    ok &= rep("T2 NEGATIVE: displaced V gives the hand-computed RHS, not 0",
              good2, "rhs=%r want=%r wrong%%=%r" % (rhs2, want, fr2))

    synth3 = [(1, 1.0, 2.0, 101.0, 100.0, 100.0, 97.0),
              (2, -1.0, 0.5, 99.0, 100.0, 100.0, 102.0)]
    _r3, fr3 = gate3_rhs(synth3, m2, "lag")
    ok &= rep("T3 NEGATIVE: favourable V gives wrong-side 0%", fr3 == 0.0,
              "wrong%%=%r" % fr3)

    print("  ... T4 runs the committed lifetime_sweep.run(seed=%d, life=%.0f) "
          "for comparison; this takes a minute." % (SELFTEST_SEED, SELFTEST_LIFE))
    t0 = time.time()
    ref = LS.run(SELFTEST_SEED, SELFTEST_LIFE)
    mine = run_arm(SELFTEST_SEED, SELFTEST_LIFE)
    spr = spread_against(mine["fills"], "lag", mine["mid"])
    pnl = mine["pnl_mid"]
    good4 = (abs(pnl - ref["pnl"]) < 1e-9
             and abs(spr - ref["spread"]) < 1e-9
             and abs((pnl - spr) - ref["inv"]) < 1e-9
             and len(mine["fills"]) == int(ref["n_fills"]))
    ok &= rep("T4 mirrored loop == committed lifetime_sweep.run()", good4,
              "pnl %.10f vs %.10f | spread %.10f vs %.10f | fills %d vs %d "
              "| %.0fs" % (pnl, ref["pnl"], spr, ref["spread"],
                           len(mine["fills"]), int(ref["n_fills"]),
                           time.time() - t0))

    good5 = abs(spr - mine["spread_lag_inline"]) < 1e-9
    ok &= rep("T5 retained fills reproduce the inline SPREAD", good5,
              "%.10f vs %.10f" % (spr, mine["spread_lag_inline"]))
    return ok


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="run the self-test ONLY and exit; no arms")
    ap.add_argument("--arm", type=float, action="append", default=None,
                    help="run only this lifetime arm; repeatable. "
                         "Default: %s" % ALL_ARMS)
    args = ap.parse_args(argv)
    arms = ALL_ARMS if not args.arm else list(args.arm)

    print("=" * 120)
    print("Task 1; PnL decomposition re-benchmarked against V")
    print("=" * 120)
    print("Committed convention: spread = sum v_j*(p_j - S_j) direct, "
          "INV = PnL - spread residual,")
    print("PnL = cash_T + q_T*S_final. v_j = sgn*size, sgn=+1 when the MM sold.")
    print("%d seeds x %.0fs (%.2f days), MM present, k=%.5f C=%.2f "
          "gamma=%.4e." % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA))
    print("lam=%.4f p_market=%.4f disp=%.4f, JOIN clipping. Arms this run: %s"
          % (LS.LAM, LS.P_MARKET, LS.DISP,
             "Self-test only" if args.selftest
             else ", ".join("life=%.0f" % a for a in arms)))
    print("")
    print("with INV as a residual, INV verifies nothing; every error in the")
    print("computation lands there by construction.")
    print("")
    print("Gate 2 and Gate 3 are both near-tautological. Gate 2 holds because")
    print("cash is benchmark-independent; Gate 3 holds because p cancels:")
    print("Spread(V) - spread(lag) = sum sgn*sz*(S_lag - V) identically, so it")
    print("passes unless a field name is mistyped. Neither is evidence the legs")
    print("Are right. The two genuinely independent checks here are:")
    print("  (i)  Self-test T4, against the committed lifetime_sweep.run()")
    print("  (ii) The wrong-side fraction against staleness.py's committed")
    print("       75.09% +/- 0.39%; a different script, different")
    print("       bookkeeping, same population. This one halts on failure.")
    print("")
    print("What T4 does not cover: it compares only lagged-benchmark")
    print("quantities, so the mid[] array feeding spread(now) is unvalidated by")
    print("it. mid[ti] = last is unconditional in lifetime_sweep.py:147 (the")
    print("continue at :151 gates the book-stats block, not the mid array) and")
    print("the transcription here matches. Read from the file, not assumed.")
    print("")
    print("Self-test")
    if not self_test():
        print("")
        print("self-test FAILED; refusing to report numbers.")
        return 1
    print("  All PASS")
    print("")
    if args.selftest:
        print("=" * 120)
        print("Self-test only (--selftest). No arms were run. T4 is the check")
        print("that decides whether the remaining runtime is worth spending;")
        print("it passed, so the life=720 arm is worth running next.")
        print("=" * 120)
        return 0

    store = {}
    comps = {}
    for life in arms:
        print("=" * 120)
        print("ARM life=%.0fs" % life)
        print("=" * 120)
        t0 = time.time()
        rows = []
        for s in SEEDS:
            r = run_arm(s, life)
            f, m = r["fills"], r["mid"]
            d = {"seed": s}
            for tag in ("lag", "now", "post", "V"):
                d["spr_" + tag] = spread_against(f, tag, m)
            d["pnl_mid"] = r["pnl_mid"]
            d["pnl_v"] = r["pnl_v"]
            for tag in ("lag", "now", "post"):
                d["inv_" + tag] = d["pnl_mid"] - d["spr_" + tag]
            d["inv_V"] = d["pnl_v"] - d["spr_V"]
            d["vol"] = sum(x[2] for x in f)
            d["n"] = len(f)
            d["n_post_fb"] = r["n_post_fb"]
            d["h_median"] = r["h_median"]
            d["h_mean"] = r["h_mean"]
            d["n_quotes"] = r["n_quotes"]
            d["sd_q"] = r["sd_q"]
            d["hgap"] = (d["h_median"] - d["spr_lag"] / d["vol"]
                         if d["vol"] else float("nan"))
            d["Ed"] = ((d["spr_lag"] - d["spr_V"]) / d["vol"]
                       if d["vol"] else float("nan"))
            d["q_final"] = r["q_final"]
            d["mid_final"] = r["mid_final"]
            d["v_final"] = r["v_final"]
            rhs, frac = gate3_rhs(f, m, "lag")
            d["g3_rhs"] = rhs
            d["wrong"] = frac
            rhs_now, frac_now = gate3_rhs(f, m, "now")
            d["g3_rhs_now"] = rhs_now
            d["wrong_now"] = frac_now
            d["g2_lhs"] = d["pnl_v"] - d["pnl_mid"]
            d["g2_rhs"] = r["q_final"] * (r["v_final"] - r["mid_final"])
            d["g3_lhs"] = d["spr_V"] - d["spr_lag"]
            rows.append(d)
            print("  seed %d done (%d fills, %.1f BTC) %.0fs"
                  % (s, d["n"], d["vol"], time.time() - t0), flush=True)
        store[life] = rows
        print("")

        print("  per seed")
        print("  %4s %10s %10s %10s %10s %10s %10s %8s %7s"
              % ("seed", "PnL(mid)", "PnL(V)", "SPR(lag)", "SPR(now)",
                 "SPR(V)", "INV(lag)", "vol BTC", "fills"))
        for d in rows:
            print("  %4d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %8.3f %7d"
                  % (d["seed"], d["pnl_mid"], d["pnl_v"], d["spr_lag"],
                     d["spr_now"], d["spr_V"], d["inv_lag"], d["vol"], d["n"]))
        print("")

        def agg(key):
            return mean_se([d[key] for d in rows])

        print("  aggregate (mean +/- SE over %d seeds)" % len(SEEDS))
        m_vol, se_vol, _ = agg("vol")
        m_pm, se_pm, _ = agg("pnl_mid")
        m_pv, se_pv, _ = agg("pnl_v")
        print("    volume           %10.4f +/- %-9.4f BTC   fills %.1f"
              % (m_vol, se_vol, statistics.mean(d["n"] for d in rows)))
        print("    PnL   S=mid      %10.2f +/- %-9.2f" % (m_pm, se_pm))
        print("    PnL   S=V        %10.2f +/- %-9.2f" % (m_pv, se_pv))
        print("")
        print("    %-14s %12s %12s %12s %12s"
              % ("benchmark", "spread", "SE", "$/BTC", "INV share %"))
        for tag, lbl in (("lag", "mid (lagged)"), ("now", "mid (contemp)"),
                         ("post", "mid (post-fill)"), ("V", "V")):
            m_s, se_s, _ = agg("spr_" + tag)
            m_i, se_i, _ = agg("inv_" + tag)
            per = m_s / m_vol if m_vol else float("nan")
            print("    %-14s %12.2f %12.2f %12.4f %12.1f"
                  % (lbl, m_s, se_s, per, inv_share(m_s, m_i)))
        print("")
        print("    %-14s %12s %12s   %s" % ("benchmark", "INV", "SE", "note"))
        for tag, lbl in (("lag", "mid (lagged)"), ("now", "mid (contemp)"),
                         ("post", "mid (post-fill)"), ("V", "V")):
            m_i, se_i, _ = agg("inv_" + tag)
            note = ("DIAGNOSTIC ONLY -- not a decomposition"
                    if tag in ("now", "post") else "")
            print("    %-14s %12.2f %12.2f   %s" % (lbl, m_i, se_i, note))
        print("")
        print("    why inv_now and inv_post are diagnostics and not benchmarks")
        print("    on equal footing with lag and V: they subtract a")
        print("    contemporaneous-mid spread from a PnL marked at the final")
        print("    lagged mid. The two halves use different price series, so")
        print("    the sum is not an internally consistent decomposition of")
        print("    anything. Only the lag row and the V row are.")
        print("")
        tot_fb = sum(d["n_post_fb"] for d in rows)
        tot_n = sum(d["n"] for d in rows)
        share = (tot_fb / tot_n) if tot_n else float("nan")
        m_hmed, se_hmed, _ = agg("h_median")
        m_hmean, se_hmean, _ = agg("h_mean")
        m_sprlag, _a, _b = agg("spr_lag")
        m_vol2, _a, _b = agg("vol")
        lagper = m_sprlag / m_vol2 if m_vol2 else float("nan")
        print("    is spread(lag)/BTC just the maker's own half-spread h?")
        print("    No; and an earlier version of this file said it should be.")
        print("    That two-term identity was wrong and is withdrawn:")
        print("        Wrong:    spread(X)/BTC = h - E[d_X]")
        print("    The maker quotes around its reservation price r = S - q*C,")
        print("    not around S, so the per-fill edge carries a third term:")
        print("        edge = sgn*(p - S) = h - q*C*sgn")
        print("        spread(lag)/BTC = h - C*E[q*sgn]")
        print("    The h gap is therefore NOT error. It is the realised")
        print("    inventory-skew cost; the one term this project is about.")
        print("    h is read from the maker's private quote log, not inferred:")
        print("      h  median over %.0f requotes : %10.4f +/- %-8.4f $"
              % (statistics.mean(d["n_quotes"] for d in rows), m_hmed, se_hmed))
        print("      h  mean                      : %10.4f +/- %-8.4f $"
              % (m_hmean, se_hmean))
        print("      spread(lag)/BTC              : %10.4f $" % lagper)
        m_gap, se_gap, _ = agg("hgap")
        m_sdq, se_sdq, _ = agg("sd_q")
        print("")
        print("      C*E[q*sgn], realised inventory-skew cost per BTC")
        print("      = h(median) - spread(lag)/BTC, per seed then averaged:")
        print("        %+10.4f +/- %-9.4f $/BTC" % (m_gap, se_gap))
        print("        C (C_TARGET)          : %10.4f" % C_TARGET)
        implied = m_gap / C_TARGET if C_TARGET else float("nan")
        print("        implied E[q*sgn]      : %+10.6f BTC  (= gap / C)"
              % implied)
        print("        run sd(q)             : %10.6f +/- %-9.6f BTC"
              % (m_sdq, se_sdq))
        bad_sign = (lagper > m_hmed)
        bad_mag = abs(m_gap) > 0.20 * abs(m_hmed)
        bad_coh = abs(implied) > m_sdq
        if bad_sign:
            print("      -> problem: wrong sign. Spread(lag)/BTC is ABOVE h,")
            print("         which the mechanism forbids; skewing toward the")
            print("         reservation price can only give up edge against S,")
            print("         never add to it. Something is wrong.")
        elif bad_mag:
            print("      -> Problem: the gap is more than 20% of h, which is")
            print("         too large to be inventory skew at this C.")
        else:
            print("      -> Coherent. Correct sign (skew gives up edge) and")
            print("         under 20% of h. Note the threshold is 20%, not 5%:")
            print("         5% of this h is comparable to the gap itself and")
            print("         would print a pass over the one term that matters.")
        if bad_coh:
            print("      -> Incoherent: |implied E[q*sgn]| exceeds the run's")
            print("         sd(q). A mean signed inventory cannot be larger")
            print("         than the inventory's own dispersion, so the")
            print("         attribution of this gap to skew is wrong.")
        else:
            print("         |implied E[q*sgn]| is within sd(q), so attributing")
            print("         the gap to inventory skew is at least coherent.")
        print("")
        print("      Stated limitation: per-fill q was not captured. The skew")
        print("      term above is inferred from the h gap, not measured.")
        print("      Capturing it directly would change recs_out's arity and")
        print("      touch spread_against, gate3_rhs and all five self-tests,")
        print("      for a ~0.4 $/BTC refinement against a ~63 $/BTC effect.")
        print("      Deliberately not done.")
        print("      Note: median and mean of h are both printed because the")
        print("      quoted half-spread carries the inventory skew and need not")
        print("      be symmetric; a mean alone could hide that.")
        print("")
        m_ed, se_ed, _ = agg("Ed")
        print("    E[d] = spread(lag)/BTC - spread(V)/BTC, computed per seed")
        print("    and averaged (never differenced off the display table):")
        print("      mean fill-time directional staleness: %+10.4f +/- %-8.4f $/BTC"
              % (m_ed, se_ed))
        print("")
        print("    S_post fallback: %d of %d fills (%.4f%%) were read on a"
              % (tot_fb, tot_n, 100.0 * share))
        print("    one-sided book and fell back to S_lag.")
        if share > POST_FALLBACK_WARN:
            print("    ABOVE %.1f%%; THE post column is not clean. Do not"
                  % (100.0 * POST_FALLBACK_WARN))
            print("    read it as a contemporaneous mid; a material share of")
            print("    it is just S_lag copied forward.")
        else:
            print("    Below %.1f%%, so the post column is clean enough to read."
                  % (100.0 * POST_FALLBACK_WARN))
        print("")

        if life == 720.0:
            print("  Gate 1; reproduction of the committed figures")
            m_s, _x, _y = agg("spr_lag")
            m_i, _x, _y = agg("inv_lag")
            checks = [("PnL", m_pm, REF["pnl"], G1_TOL_ABS),
                      ("SPREAD", m_s, REF["spread"], G1_TOL_ABS),
                      ("INV", m_i, REF["inv"], G1_TOL_ABS),
                      ("INV share %", inv_share(m_s, m_i), REF["inv_share"],
                       G1_TOL_SHARE)]
            print("    tolerances are absolute, set by the committed figures'")
            print("    printing precision: %.3f on 2dp values, %.3f on the 1dp"
                  % (G1_TOL_ABS, G1_TOL_SHARE))
            print("    INV share. Not relative.")
            g1 = True
            for nm, got, want, tol in checks:
                d_ = abs(got - want)
                okk = d_ <= tol
                g1 &= okk
                print("    %-12s got %12.4f   committed %10.2f   |d|=%.4f  "
                      "tol %.3f  %s"
                      % (nm, got, want, d_, tol, "OK" if okk else "mismatch"))
            print("    Gate 1: %s" % ("PASS" if g1 else "FAIL; stop"))
            if not g1:
                print("    Everything below is void until this is explained.")
                return 1
            print("")

        print("  Gate 2; PnL(V) - PnL(mid) == q_final*(V_final - mid_final)")
        print("    tolerance %.1e dollars, absolute" % GATE2_TOL)
        worst = 0.0
        for d in rows:
            dev = abs(d["g2_lhs"] - d["g2_rhs"])
            worst = max(worst, dev)
            print("      seed %d  lhs %14.6f  rhs %14.6f  |d| %.3e  q_f %+9.4f"
                  % (d["seed"], d["g2_lhs"], d["g2_rhs"], dev, d["q_final"]))
        print("    worst |d| = %.3e  -> %s"
              % (worst, "PASS" if worst < GATE2_TOL else "FAIL"))
        print("    note: cash is benchmark-independent, so this identity is")
        print("    near-automatic. It confirms cash was not disturbed; it is")
        print("    NOT evidence that the legs are right.")
        print("")

        print("  Gate 3; spread(V) - spread(lag) == -sum_j v_j*(V_j - S_j)")
        print("    RHS computed independently in gate3_rhs().")
        worst3 = 0.0
        for d in rows:
            dev = abs(d["g3_lhs"] - d["g3_rhs"])
            worst3 = max(worst3, dev)
            print("      seed %d  lhs %14.4f  rhs %14.4f  |d| %.3e  "
                  "wrong-side %6.2f%%"
                  % (d["seed"], d["g3_lhs"], d["g3_rhs"], dev, d["wrong"]))
        m_rhs, se_rhs, _ = agg("g3_rhs")
        m_wr, se_wr, _ = agg("wrong")
        print("    worst |d| = %.3e  -> %s"
              % (worst3, "PASS" if worst3 < 1e-6 else "FAIL"))
        print("    Gate 3 RHS total   %+10.2f +/- %-9.2f dollars"
              % (m_rhs, se_rhs))
        print("    wrong-side frac    %10.2f%% +/- %-8.2f%%   "
              "[sign(v_j)*(V_j - S_j) > 0]" % (m_wr, se_wr))
        if life == 720.0:
            dev = abs(m_wr - WRONG_REF)
            wok = dev <= WRONG_TOL_PTS
            print("")
            print("    cross-script check (ii); the one that halts.")
            print("    This is one of only two genuinely independent checks in")
            print("    this file: staleness.py (00b1773) computes the same")
            print("    fraction over the same population with different")
            print("    bookkeeping. Gates 2 and 3 cannot fail informatively;")
            print("    this can.")
            print("      committed staleness.py : %6.2f%% +/- %.2f%%"
                  % (WRONG_REF, WRONG_REF_SE))
            print("      measured here          : %6.2f%% +/- %.2f%%"
                  % (m_wr, se_wr))
            print("      |d| = %.2f pts, tolerance %.2f pts (3 SE)"
                  % (dev, WRONG_TOL_PTS))
            print("      -> %s" % ("PASS" if wok else "FAIL"))
            if not wok:
                print("")
                print("    the two scripts disagree. bench_v.py and")
                print("    staleness.py do not see the same population, or do")
                print("    not sign it the same way. Stopping before the")
                print("    life=60 ARM; that identification comes first and")
                print("    nothing below is worth computing until it is made.")
                return 1
        print("")

        print("  How much of Gate 3 is the one-second bookkeeping lag?")
        m_lag, _a, _b = agg("spr_lag")
        m_now, _a, _b = agg("spr_now")
        m_post, _a, _b = agg("spr_post")
        m_V, _a, _b = agg("spr_V")
        m_rhs_now, se_rn, _ = agg("g3_rhs_now")
        print("    Spread(lag) - spread(now)  = %+10.2f   <- pure 1s lag"
              % (m_lag - m_now))
        print("    spread(now) - spread(V)    = %+10.2f   <- genuine V-mid gap"
              % (m_now - m_V))
        print("    spread(lag) - spread(V)    = %+10.2f   <- total, = -Gate3"
              % (m_lag - m_V))
        comp_lag = m_lag - m_now
        comp_v = m_now - m_V
        comps[life] = {"lag": comp_lag, "v": comp_v, "own": m_now - m_post,
                       "tot": m_lag - m_V}
        den = abs(comp_lag) + abs(comp_v)
        print("    A share-of-total is not reported here. The two components")
        print("    can have opposite signs, and when they do each one can")
        print("    exceed the total in magnitude, so dividing by the total is")
        print("    meaningless. Magnitudes as a fraction of the sum of")
        print("    absolute values, which stays well-defined either way:")
        if den > 0:
            print("      |1s lag|      / (|1s lag| + |V-mid|) = %6.2f%%"
                  % (100.0 * abs(comp_lag) / den))
            print("      |V-mid gap|   / (|1s lag| + |V-mid|) = %6.2f%%"
                  % (100.0 * abs(comp_v) / den))
            print("      components %s in sign"
                  % ("agree" if comp_lag * comp_v > 0 else "oppose"))
        print("    spread(now) - spread(post) = %+10.2f   <- the MM's own fill"
              % (m_now - m_post))
        print("    Gate 3 RHS vs contemporaneous mid: %+10.2f +/- %-9.2f"
              % (m_rhs_now, se_rn))
        m_wrn, se_wrn, _ = agg("wrong_now")
        print("    wrong-side vs contemporaneous mid: %9.2f%% +/- %-8.2f%%"
              % (m_wrn, se_wrn))
        print("")

    print("=" * 120)
    print("Cross-arm; is spread inflated more in the stale arm?")
    print("=" * 120)
    if len(arms) < 2:
        print("  Only one arm was run (%s). The cross-arm comparison is the"
              % ", ".join("life=%.0f" % a for a in arms))
        print("  point of Task 1 and it is NOT answered by this run. The table")
        print("  below carries the single arm only; do not read a comparison")
        print("  into it.")
    print("  The pre-registered prediction: spread falls under S=V on both")
    print("  arms, and falls more on life=720 than life=60.")
    print("")
    print("  %-8s %12s %12s %12s %10s %12s %12s"
          % ("arm", "SPR(lag)", "SPR(V)", "fall", "fall %", "SPR(lag)/BTC",
             "SPR(V)/BTC"))
    for life in arms:
        rows = store[life]
        m_l, _a, _b = mean_se([d["spr_lag"] for d in rows])
        m_v, _a, _b = mean_se([d["spr_V"] for d in rows])
        m_vol, _a, _b = mean_se([d["vol"] for d in rows])
        fall = m_l - m_v
        print("  life=%-3.0f %12.2f %12.2f %12.2f %9.1f%% %12.4f %12.4f"
              % (life, m_l, m_v, fall,
                 (100.0 * fall / abs(m_l)) if m_l else float("nan"),
                 m_l / m_vol, m_v / m_vol))
    print("")
    print("")
    print("  E[d]; mean fill-time directional staleness, $/BTC")
    print("  Computed per seed as (spread(lag) - spread(V))/volume and then")
    print("  averaged. NOT differenced off the table above.")
    print("  %-10s %14s %12s %12s" % ("arm", "E[d] $/BTC", "SE", "h median $"))
    for life in arms:
        rws = store[life]
        m_e, se_e, _ = mean_se([d["Ed"] for d in rws])
        m_h, _s, _n = mean_se([d["h_median"] for d in rws])
        print("  life=%-5.0f %14.4f %12.4f %12.4f" % (life, m_e, se_e, m_h))
    if len(arms) >= 2:
        a0, a1 = arms[0], arms[1]
        r0, r1 = store[a0], store[a1]
        pair = [(d0["Ed"] / d1["Ed"]) for d0, d1 in zip(r0, r1)
                if d1["Ed"] != 0.0]
        m_r, se_r, _ = mean_se(pair)
        me0, _x, _y = mean_se([d["Ed"] for d in r0])
        me1, _x, _y = mean_se([d["Ed"] for d in r1])
        print("")
        print("    ratio E[d](life=%.0f) / E[d](life=%.0f):" % (a0, a1))
        print("      Paired per-seed mean : %10.4f +/- %-9.4f" % (m_r, se_r))
        print("      ratio of the means   : %10.4f"
              % (me0 / me1 if me1 else float("nan")))
        print("      The arms share seeds and therefore share the V path, so")
        print("      the paired figure is the one to quote; an unpaired SE")
        print("      would assume an independence that does not hold.")
        print("")
        print("  Open observation; not a finding, not chased, and not needed")
        print("  for any task 1 conclusion. Between the arms the two components")
        print("  of the spread(lag) - spread(V) gap both grow in magnitude while")
        print("  the total collapses, because the V-mid component reverses sign:")
        print("    %-10s %14s %14s %14s %12s"
              % ("arm", "1s lag", "V-mid gap", "total", "own fill"))
        for life in arms:
            c = comps[life]
            print("    life=%-5.0f %14.2f %14.2f %14.2f %12.2f"
                  % (life, c["lag"], c["v"], c["tot"], c["own"]))
        print("  The own-fill term is small in both arms and S_post ~ S_now, so")
        print("  the maker's own market impact is ruled out as the driver. What")
        print("  drives it is not explained here and no mechanism is offered.")
    print("")
    print("  Spread per BTC is the column that decides how to read this:")
    print("  lifetime_sweep_results.txt section B shows spread going 63.83 ->")
    print("  178.43 as lifetime shortens, and totals alone cannot say whether")
    print("  the per-BTC edge moved at all.")
    print("")
    print("  Does not overturn: this says nothing about whether quote width can")
    print("  fix the loss. 009fc18 settled that independently; 10x tighter")
    print("  quoting left INV at -556.44 against -589.01. Task 1 is about which")
    print("  column the loss is booked in, not whether the loss is real.")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
