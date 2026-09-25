# onehour_confirm.py: one-hour confirmation of extraction_diag's one live cell (cluster B, clip 0.12, h = 3600 s)
# - committed 8-seed cell: TMB = -5.2016 +/- 2.3827 bp vs round trip 1.4704 +/- 0.0100 bp (STRADDLE under P2 and P3)
# - control arm only, 7 days per seed; extraction_diag.run_diag imported unchanged
# - 8 anchor seeds (0-7) must reproduce the committed clip-0.120 block; 96 scored seeds: 113..201 + 291..297
#   (corrected before the run from 100-195, which collides on 20 RNG streams)
# - pre-registered verdict (P3), per trade, Z = 1.96:
#   |TMB|/2 band = [max(0, |m|/2 - Z se/2), |m|/2 + Z se/2]; cost band = [c - Z se_c, c + Z se_c]
#   STRATEGY AT 1h iff |TMB|/2 lower > cost upper; STRUCTURAL AT 1h iff |TMB|/2 upper < cost lower; else INCONCLUSIVE
# - (P2) (factor 1.0) reported beside it; calls also computed at t(95) = 1.985 and flagged if they change
# - predicted sign negative; tails reported separately; R2: no seed > 40% of the squared deviation; h = 900 descriptive

import contextlib
import io
import math
import os
import random
import re
import statistics
import sys
import time
from statistics import NormalDist

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import extraction_diag as ED
import sim_run
from clipped_placement import make_world
from damage_ab import mean_se
from damage_qs12 import _finite, _ratio
from fundamental_value import generate_value_path
from phase7_nobs import JITTER_SEED_BASE
from phase8_reference import dominance
from clipped_window import Z

SIZE = 0.12
HORIZON = 3600
DESC_H = 900
ANCHOR = list(range(8))
SCORED = list(range(113, 202)) + list(range(291, 298))
N_SCORED = 96
T95 = 1.985
DOM_BAR = 0.40
COMMITTED_FILE = os.path.join(_HERE, "extraction_diag_results.txt")

# (S) stream map; LIFETIME_OFFSET is a literal in noise_traders (checked by T-streams)
LIFETIME_OFFSET = 90001
USED_SEEDS = range(24)
USED_JITTER = [JITTER_SEED_BASE + s for s in USED_SEEDS]
USED_RAW = [1, 7, 99, 6789, 12345, 7 + LIFETIME_OFFSET]


# seeds
def streams(s):
    """The four private RNG seeds of a control-arm run of seed index s."""
    stride = sim_run.SEED_STRIDE
    noise = sim_run.SEED_NOISE_BASE + stride * s
    return {"value": sim_run.SEED_V_BASE + stride * s,
            "noise": noise,
            "lifetime": noise + LIFETIME_OFFSET,
            "informed": sim_run.SEED_INF_BASE + stride * s}


def committed_streams():
    taken = {}
    for u in USED_SEEDS:
        for fam, v in streams(u).items():
            taken[v] = "%s(%d)" % (fam, u)
    for v in USED_JITTER:
        taken.setdefault(v, "jitter(%d)" % (v - JITTER_SEED_BASE))
    for v in USED_RAW:
        taken.setdefault(v, "raw literal %d" % v)
    return taken


def collisions(seeds, taken):
    """Streams in seeds that collide with a taken stream or each other. Returns [(this, that)]."""
    taken = dict(taken)
    bad = []
    for s in seeds:
        for fam, v in streams(s).items():
            me = "%s(%d)" % (fam, s)
            if v in taken:
                bad.append((me, taken[v]))
            else:
                taken[v] = me
    return bad


def admissible_seeds(start, n, taken):
    """(S): first n seeds >= start with no colliding stream."""
    taken = dict(taken)
    out = []
    s = start
    while len(out) < n:
        st = streams(s)
        vals = list(st.values())
        if len(set(vals)) == len(vals) and not any(v in taken for v in vals):
            out.append(s)
            for fam, v in st.items():
                taken[v] = "%s(%d)" % (fam, s)
        s += 1
    return out


# statistics
def strict_mean_se_n(values, what, n_expected):
    """extraction_diag.strict_mean_se with n as a parameter."""
    if len(values) != n_expected:
        raise ValueError("%s: %d values, expected %d" % (what, len(values),
                                                        n_expected))
    for v in values:
        _finite(v, what)
    m, se, n = mean_se(values)
    if n != n_expected:
        raise ValueError("%s: mean_se kept %d of %d" % (what, n, n_expected))
    _finite(m, "mean of " + what)
    if not _finite(se, "SE of " + what) > 0:
        raise ValueError("%s: SE is %r -- no interval, no verdict" % (what, se))
    return m, se


def call_at(m, se, c, se_c, factor, z):
    """extraction_diag.horizon_call with the critical value as a parameter."""
    mag = abs(m) * factor
    s = se * factor
    mag_hi = mag + z * s
    mag_lo = max(0.0, mag - z * s)
    c_hi = c + z * se_c
    c_lo = c - z * se_c
    for v, nm in ((mag_hi, "mag_hi"), (mag_lo, "mag_lo"), (c_hi, "c_hi"),
                  (c_lo, "c_lo")):
        _finite(v, nm)
    if mag_hi < c_lo:
        call = "BELOW"
    elif mag_lo > c_hi:
        call = "ABOVE"
    else:
        call = "STRADDLE"
    return call, mag, mag_lo, mag_hi, c_lo, c_hi


def tail_call(g, se_g, c, se_c, z):
    """Signed gain against the cost band (descriptive)."""
    lo, hi = g - z * se_g, g + z * se_g
    c_lo, c_hi = c - z * se_c, c + z * se_c
    for v, nm in ((lo, "gain lo"), (hi, "gain hi"), (c_lo, "c_lo"),
                  (c_hi, "c_hi")):
        _finite(v, nm)
    if lo > c_hi:
        return "ABOVE", lo, hi
    if hi < c_lo:
        return "BELOW", lo, hi
    return "STRADDLE", lo, hi


VERDICT_WORD = {"ABOVE": "STRATEGY AT 1h", "BELOW": "STRUCTURAL AT 1h",
                "STRADDLE": "INCONCLUSIVE"}


# per run
def seed_summary(samples, s):
    """One run's 60 s capture -> the numbers the verdict needs (quintiles as ED.tmb)."""
    costs = [c for (_q, _m, c) in samples.values()]
    row = {"seed": s, "cost": _finite(statistics.fmean(costs),
                                      "cost seed %d" % s),
           "n_samples": len(samples)}
    for h in (DESC_H, HORIZON):
        xs, ys = ED.windows(samples, h)
        n = len(xs)
        k = n // 5
        if k < 2:
            raise ValueError("seed %d h %d: quintile of %d windows" % (s, h, k))
        order = sorted(range(n), key=lambda i: xs[i])
        top = statistics.fmean([ys[i] for i in order[-k:]])
        bot = statistics.fmean([ys[i] for i in order[:k]])
        whole = ED.tmb(xs, ys, "seed %d h %d" % (s, h))
        if top - bot != whole:
            raise AssertionError("seed %d h %d: top - bot %.17g != ED.tmb %.17g"
                                 % (s, h, top - bot, whole))
        row[h] = {"tmb": _finite(whole, "TMB seed %d h %d" % (s, h)),
                  "top": _finite(top, "top seed %d h %d" % (s, h)),
                  "bot": _finite(bot, "bot seed %d h %d" % (s, h)),
                  "all": _finite(statistics.fmean(ys),
                                 "all-window mean seed %d h %d" % (s, h)),
                  "n": n}
    return row


def print_row(row, tag, elapsed):
    print("  %-7s seed %4d  cost %.6f  TMB900 %+10.6f  TMB3600 %+10.6f  "
          "top3600 %+10.6f  bot3600 %+10.6f  all3600 %+9.6f  n %d/%d  %4.0fs"
          % (tag, row["seed"], row["cost"], row[DESC_H]["tmb"],
             row[HORIZON]["tmb"], row[HORIZON]["top"], row[HORIZON]["bot"],
             row[HORIZON]["all"], row[DESC_H]["n"], row[HORIZON]["n"],
             elapsed), flush=True)


def run_seed(s):
    out, cap = ED.run_diag(s, False, ED.THETA, ED.HOLD, SIZE)
    ED.check_run(out, cap, "control", s, SIZE)
    row = seed_summary(cap["samples"], s)
    row["nan_sharpe"] = math.isnan(out["mm_sharpe"])
    return row


# anchor
def _tol(num_str):
    """Half the last printed digit of a number as printed"""
    if "." not in num_str:
        return 0.5
    return 0.5 * 10.0 ** (-len(num_str.split(".")[1]))


def parse_committed(lines):
    """Clip-0.120 block of extraction_diag_results.txt, up to its (P2) VERDICT line."""
    head = [i for i, ln in enumerate(lines)
            if ln.lstrip().startswith("maker size 0.120 -- measured round-trip "
                                      "cost")]
    if len(head) != 1:
        raise ValueError("expected exactly one 'maker size 0.120' cost line, "
                         "found %d" % len(head))
    i = head[0]
    mc = re.search(r"market order: (-?[0-9.]+) \+/- ([0-9.]+) bp", lines[i])
    if mc is None:
        raise ValueError("cost line does not parse: %r" % lines[i])
    out = {"cost": (float(mc.group(1)), _tol(mc.group(1)),
                    float(mc.group(2)), _tol(mc.group(2)))}
    j = i + 1
    while j < len(lines) and "(P2) VERDICT at 0.120" not in lines[j]:
        mr = re.match(r"^\s+(\d+)\s+(-?[0-9.]+) \+/- ([0-9.]+)\s", lines[j])
        if mr and int(mr.group(1)) in (DESC_H, HORIZON):
            h = int(mr.group(1))
            if h in out:
                raise ValueError("h=%d row appears twice in the 0.120 block" % h)
            out[h] = (float(mr.group(2)), _tol(mr.group(2)),
                      float(mr.group(3)), _tol(mr.group(3)))
        j += 1
    if j == len(lines):
        raise ValueError("0.120 block has no (P2) VERDICT line")
    for key in (DESC_H, HORIZON):
        if key not in out:
            raise ValueError("h=%d row not found in the 0.120 block" % key)
    return out


def anchor_check(rows, committed):
    """Compare the 8 anchor seeds with the committed values; returns failures."""
    got = {"cost": strict_mean_se_n([r["cost"] for r in rows], "anchor cost",
                                    len(ANCHOR))}
    for h in (DESC_H, HORIZON):
        got[h] = strict_mean_se_n([r[h]["tmb"] for r in rows],
                                  "anchor TMB h%d" % h, len(ANCHOR))
    fails = []
    for key, label in (("cost", "round-trip cost"), (HORIZON, "TMB h=3600"),
                       (DESC_H, "TMB h=900")):
        want_m, tol_m, want_se, tol_se = committed[key]
        for part, g, w, tol in (("mean", got[key][0], want_m, tol_m),
                                ("se", got[key][1], want_se, tol_se)):
            ok = abs(g - w) <= tol
            print("  %-16s %-4s  got %+.9f  committed %+.4f  tol %.0e  %s"
                  % (label, part, g, w, tol, "OK" if ok else "*** miss ***"))
            if not ok:
                fails.append("%s %s" % (label, part))
    return fails


# report
def report(rows, n_expected):
    """Pre-registered verdict and what is reported beside it."""
    seeds = [r["seed"] for r in rows]
    c, se_c = strict_mean_se_n([r["cost"] for r in rows], "cost", n_expected)
    res = {"cost": (c, se_c)}
    print("  measured round-trip cost of a 0.020 BTC market order, %d seeds: "
          "%.4f +/- %.4f bp" % (n_expected, c, se_c))
    print("  (cost 95%% band [%.4f, %.4f] bp at Z=%.2f; excludes permanent "
          "impact, L2)" % (c - Z * se_c, c + Z * se_c, Z))
    print("")

    for h in (HORIZON, DESC_H):
        tag = "SCORED" if h == HORIZON else "DESCRIPTIVE"
        vals = [r[h]["tmb"] for r in rows]
        m, se = strict_mean_se_n(vals, "TMB h%d" % h, n_expected)
        sign_ok = (m < 0) if ED.PREDICTED_SIGN < 0 else (m > 0)
        p3 = call_at(m, se, c, se_c, ED.COMPANION_FACTOR, Z)
        p2 = call_at(m, se, c, se_c, 1.0, Z)
        p3t = call_at(m, se, c, se_c, ED.COMPANION_FACTOR, T95)
        p2t = call_at(m, se, c, se_c, 1.0, T95)
        print("  h = %d s  [%s]" % (h, tag))
        print("    TMB %+.4f +/- %.4f bp  (|t| %.2f)   predicted sign "
              "negative: %s" % (m, se, _ratio(abs(m), se, "TMB t h%d" % h),
                                "yes" if sign_ok else "No; sign failure"))
        print("    (P3) |TMB|/2 band [%.4f, %.4f]  vs cost [%.4f, %.4f]  -> %-8s"
              "  at t(95)=%.3f: %s" % (p3[2], p3[3], p3[4], p3[5], p3[0], T95,
                                       p3t[0]))
        print("    (P2) |TMB|   band [%.4f, %.4f]  vs cost [%.4f, %.4f]  -> %-8s"
              "  at t(95)=%.3f: %s" % (p2[2], p2[3], p2[4], p2[5], p2[0], T95,
                                       p2t[0]))
        flips = [nm for nm, a, b in (("P3", p3, p3t), ("P2", p2, p2t))
                 if a[0] != b[0]]
        if flips:
            print("    *** flag: %s changes between Z=%.2f and t(95)=%.3f; "
                  "that call is decided in the gap ***"
                  % (" and ".join(flips), Z, T95))

        gt = [ED.PREDICTED_SIGN * r[h]["top"] for r in rows]
        gb = [-ED.PREDICTED_SIGN * r[h]["bot"] for r in rows]
        for r, a, b in zip(rows, gt, gb):
            if abs((a + b) - ED.PREDICTED_SIGN * r[h]["tmb"]) > \
                    1e-9 * max(1.0, abs(r[h]["tmb"])):
                raise AssertionError("tail gains do not sum to the TMB, seed "
                                     "%d h %d" % (r["seed"], h))
        mt, set_ = strict_mean_se_n(gt, "top-tail gain h%d" % h, n_expected)
        mb, seb = strict_mean_se_n(gb, "bottom-tail gain h%d" % h, n_expected)
        md, sed = strict_mean_se_n([a - b for a, b in zip(gt, gb)],
                                   "tail asymmetry h%d" % h, n_expected)
        ma, sea = strict_mean_se_n([r[h]["all"] for r in rows],
                                   "all-window mean h%d" % h, n_expected)
        ct = tail_call(mt, set_, c, se_c, Z)
        cb = tail_call(mb, seb, c, se_c, Z)
        ctt = tail_call(mt, set_, c, se_c, T95)
        cbt = tail_call(mb, seb, c, se_c, T95)
        print("    tails, sign-aligned gain of the trade each tail implies "
              "(descriptive):")
        print("      top quintile (maker long)  -> sell, gain -top  %+.4f +/- "
              "%.4f  band [%+.4f, %+.4f]  %-8s at t(95): %s"
              % (mt, set_, ct[1], ct[2], ct[0], ctt[0]))
        print("      bottom quintile (short)    -> buy,  gain +bot  %+.4f +/- "
              "%.4f  band [%+.4f, %+.4f]  %-8s at t(95): %s"
              % (mb, seb, cb[1], cb[2], cb[0], cbt[0]))
        print("      asymmetry gain_top - gain_bot, paired  %+.4f +/- %.4f "
              "(%.2f SE)" % (md, sed, _ratio(abs(md), sed, "asym t h%d" % h)))
        print("      all-window mean return (each seed's drift, L1)  %+.4f +/- "
              "%.4f" % (ma, sea))
        one_tail = [nm for nm, cc in (("TOP", ct), ("BOTTOM", cb))
                    if cc[0] == "ABOVE"]
        if one_tail and p3[0] != "ABOVE":
            print("    *** flag: the %s tail alone clears the cost while (P3) "
                  "does not; a one-tail attacker beats |TMB|/2 here ***"
                  % " and ".join(one_tail))

        i, share = dominance(vals)
        r2_ok = share <= DOM_BAR
        print("    R2 per-seed TMB: worst seed %d carries %.1f%% of the "
              "squared deviation (bar %.0f%%)  %s"
              % (seeds[i], 100.0 * share, 100.0 * DOM_BAR,
                 "ok" if r2_ok else "R2 FAIL: one-seed-dominated"))
        print("")
        res[h] = {"tmb": (m, se), "sign_ok": sign_ok, "p3": p3[0],
                  "p2": p2[0], "p3_t95": p3t[0], "p2_t95": p2t[0],
                  "flips": flips, "gain_top": (mt, set_),
                  "gain_bot": (mb, seb), "tail_top": ct[0],
                  "tail_bot": cb[0], "one_tail": one_tail,
                  "asym": (md, sed), "r2_ok": r2_ok,
                  "r2": (seeds[i], share)}

    s3 = res[HORIZON]
    res["verdict"] = VERDICT_WORD[s3["p3"]]
    res["p2_verdict"] = VERDICT_WORD[s3["p2"]]
    print("=" * 128)
    print("Verdict: (P3), per trade, h = 3600 s, %d scored seeds" % n_expected)
    print("=" * 128)
    print("  (P3) VERDICT: %s" % res["verdict"])
    print("  (P2) beside it (not the verdict): %s" % res["p2_verdict"])
    if not s3["sign_ok"]:
        print("  predicted sign FAILED at 3600: the TMB is positive.")
    if s3["flips"]:
        print("  Flag: %s decided between Z=1.96 and t(95)=1.985."
              % " and ".join(s3["flips"]))
    if s3["one_tail"] and s3["p3"] != "ABOVE":
        print("  flag: a one-tail attacker (%s) clears the cost although (P3) "
              "does not." % " and ".join(s3["one_tail"]))
    print("  R2 at 3600: %s (seed %d, %.1f%%)"
          % ("ok" if s3["r2_ok"] else "FAIL", s3["r2"][0], 100 * s3["r2"][1]))
    nan_sh = sum(1 for r in rows if r.get("nan_sharpe"))
    print("  NaN Sharpe seeds (counted, not halted on): %d" % nan_sh)
    return res


# self-tests
def t_ident():
    assert SIZE == ED.MIRROR_SIZE and SIZE in ED.MM_SIZES
    assert HORIZON in ED.HORIZONS and DESC_H in ED.HORIZONS
    assert ANCHOR == list(ED.SEEDS)
    assert ED.T == 604800.0 and abs(ED.GAMMA - 1.3392e-06) < 1e-9
    assert ED.THETA == 0.0 and ED.HOLD == 10800 and ED.SNIFF_SIZE == 0.02
    assert ED.PREDICTED_SIGN == -1 and ED.COMPANION_FACTOR == 0.5
    assert Z == 1.96 and T95 == 1.985 and DOM_BAR == 0.40
    assert len(SCORED) == N_SCORED == len(set(SCORED))
    assert not set(SCORED) & set(USED_SEEDS)
    print("  T-IDENT      PASS  clip 0.12, h 3600 scored / 900 descriptive, "
          "anchor = extraction_diag's 8 seeds;")
    print("                     T, gamma, theta, hold, sniffer size, sign and "
          "companion factor are extraction_diag's.")


def t_streams():
    """Each RNG make_world builds starts in random.Random(predicted seed)'s state."""
    for s in (0, 7, 113, 297):
        path, _vf, _eng, noise, informed = make_world(
            s, 600.0, ED.LAM, ED.P_MARKET, ED.LIFE, ED.DISP, "join")
        st = streams(s)
        assert noise.rng_lifetime.getstate() == \
            random.Random(st["lifetime"]).getstate(), ("lifetime", s)
        assert informed.rng.getstate() == \
            random.Random(st["informed"]).getstate(), ("informed", s)
        assert noise.rng.getstate() == \
            random.Random(st["noise"]).getstate(), ("noise", s)
        again = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE,
                                    1.0, 600, seed=st["value"])
        assert list(path) == list(again), ("value", s)
    print("  T-streams    PASS  for seeds 0, 7, 113, 297 the constructed "
          "world's noise, lifetime and informed RNGs are")
    print("                     in the state of random.Random(predicted seed), "
          "and the value path regenerates")
    print("                     from its predicted seed: the stream map is the "
          "code's, lifetime offset %d." % LIFETIME_OFFSET)


def t_seeds():
    taken = committed_streams()
    bad = collisions(list(range(100, 196)), taken)
    assert len(bad) == 20, len(bad)
    assert ("informed(100)", "lifetime(11)") in bad
    assert ("informed(189)", "lifetime(100)") in bad
    got = admissible_seeds(100, N_SCORED, taken)
    assert got == SCORED, (got[:3], got[-10:])
    assert collisions(SCORED, taken) == []
    allv = [v for s in list(ANCHOR) + SCORED for v in streams(s).values()]
    assert len(allv) == 4 * (len(ANCHOR) + N_SCORED) == len(set(allv))
    anchor_vals = {v for s in ANCHOR for v in streams(s).values()}
    scored_vals = {v for s in SCORED for v in streams(s).values()}
    assert not anchor_vals & scored_vals
    assert not scored_vals & set(taken)
    print("  T-seeds      PASS  the specified 100-195 has %d stream collisions "
          "(13 with committed seeds 11-23," % len(bad))
    print("                     7 inside the set) and was NOT used. The first "
          "96 admissible seeds >= 100 are")
    print("                     113..201 + 291..297 = SCORED, exactly; all %d "
          "anchor+scored streams are unique and"
          % len(allv))
    print("                     the 384 scored streams touch no stream any "
          "committed run used.")


def t_stats():
    xs = [1.0, 2.0, 4.0, 7.0, 3.0, 5.0, 6.0, 8.0]
    assert strict_mean_se_n(xs, "p", 8) == ED.strict_mean_se(xs, "p")
    for bad in ([1.0] * 7, [1.0] * 8, [float("nan")] + [1.0] * 7):
        try:
            strict_mean_se_n(bad, "planted", 8)
        except ValueError:
            continue
        raise AssertionError("strict_mean_se_n accepted %r" % bad)
    cases = [(0.1, 0.05, 1.0, 0.01), (5.0, 0.5, 1.0, 0.01),
             (1.0, 0.5, 1.0, 0.01), (-5.0, 0.5, 1.0, 0.01),
             (2.2, 0.2, 1.0, 0.01), (-5.2016, 2.3827, 1.4704, 0.0100),
             (-1.5313, 0.6218, 1.4704, 0.0100)]
    for m, se, cc, sc in cases:
        for f in (1.0, 0.5):
            assert call_at(m, se, cc, sc, f, Z) == \
                ED.horizon_call(m, se, cc, sc, f), (m, se, f)
    # committed 8-seed 3600 cell re-called: STRADDLE both ways
    assert call_at(-5.2016, 2.3827, 1.4704, 0.0100, 0.5, Z)[0] == "STRADDLE"
    assert call_at(-5.2016, 2.3827, 1.4704, 0.0100, 1.0, Z)[0] == "STRADDLE"
    # planted flip between 1.96 and 1.985
    assert call_at(-5.94, 2.0, 1.0, 0.001, 0.5, Z)[0] == "ABOVE"
    assert call_at(-5.94, 2.0, 1.0, 0.001, 0.5, T95)[0] == "STRADDLE"
    assert tail_call(3.0, 0.5, 1.0, 0.01, Z)[0] == "ABOVE"
    assert tail_call(-3.0, 0.5, 1.0, 0.01, Z)[0] == "BELOW"
    assert tail_call(1.2, 0.5, 1.0, 0.01, Z)[0] == "STRADDLE"
    print("  T-call       PASS  call_at == extraction_diag.horizon_call EXACTLY "
          "at Z on %d cases x 2 factors;" % len(cases))
    print("                     the committed 8-seed cell re-calls STRADDLE; a "
          "planted call flips 1.96 -> 1.985;")
    print("                     a tail is scored signed (a -3 bp gain is BELOW, "
          "not ABOVE); strict means refuse.")


def t_parse():
    with open(COMMITTED_FILE, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    got = parse_committed(lines)
    want = {"cost": (1.4704, 1.0e-2), HORIZON: (-5.2016, 2.3827),
            DESC_H: (-1.5313, 0.6218)}
    for k, (wm, ws) in want.items():
        gm, tm, gs, ts = got[k]
        assert abs(gm - wm) < 1e-12 and abs(gs - ws) < 1e-12, (k, got[k])
        assert abs(tm - 5e-5) < 1e-15 and abs(ts - 5e-5) < 1e-15, (k, tm, ts)
    # a 0.020 block placed first must not be read
    planted = [
        "  maker size 0.020 -- measured round-trip cost of a 0.020 BTC market "
        "order: 9.9999 +/- 0.9999 bp",
        "    900      -8.8888 +/- 0.8888    yes",
        "    3600     -7.7777 +/- 0.7777    yes",
        "    (P2) VERDICT at 0.020: INCONCLUSIVE",
        "  maker size 0.120 -- measured round-trip cost of a 0.020 BTC market "
        "order: 1.46 +/- 0.010 bp",
        "    900      -1.5 +/- 0.62    yes",
        "    3600     -5.20 +/- 2.3827    yes",
        "    10800    -0.6760 +/- 5.3301    yes",
        "    (P2) VERDICT at 0.120: INCONCLUSIVE",
        "    3600     -6.6666 +/- 0.6666    yes",
    ]
    p = parse_committed(planted)
    assert p["cost"][0] == 1.46 and abs(p["cost"][1] - 5e-3) < 1e-15
    assert abs(p["cost"][3] - 5e-4) < 1e-15
    assert p[DESC_H][0] == -1.5 and abs(p[DESC_H][1] - 0.05) < 1e-15
    assert p[HORIZON][0] == -5.20 and abs(p[HORIZON][1] - 5e-3) < 1e-15
    for broken in (planted[:4], planted[4:8]):
        try:
            parse_committed(broken)
        except ValueError:
            continue
        raise AssertionError("parse accepted a broken block")
    print("  T-parse      PASS  the committed 0.120 block parses to cost 1.4704 "
          "+/- 0.0100, TMB3600 -5.2016 +/-")
    print("                     2.3827, TMB900 -1.5313 +/- 0.6218, tol 5e-5 "
          "each; a planted 0.020 block placed")
    print("                     first is never read, rows after the 0.120 "
          "verdict are ignored, and tolerances")
    print("                     follow each number's own printed digits.")


def _spread(m, se, n):
    """n values (n even) with mean m and SE se exactly."""
    d = se * math.sqrt(n - 1)
    return [m + d if i % 2 == 0 else m - d for i in range(n)]


def _rows(n, seeds, cost, tmb3600, tmb900, top_share=0.5):
    """Planted seed_summary rows."""
    rows = []
    cs = _spread(*cost, n)
    t3 = _spread(*tmb3600, n)
    t9 = _spread(*tmb900, n)
    for i in range(n):
        r = {"seed": seeds[i], "cost": cs[i], "n_samples": 10080,
             "nan_sharpe": False}
        e = 0.05 * ((7 * i) % 5 - 2)
        for h, v in ((HORIZON, t3[i]), (DESC_H, t9[i])):
            top = top_share * v + e
            r[h] = {"tmb": v, "top": top, "bot": top - v, "all": 0.01 + e,
                    "n": 167 if h == HORIZON else 671}
        rows.append(r)
    return rows


def _quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a)
    return out, buf.getvalue()


def t_postsim():
    """seed_summary, anchor_check and report on planted data; output captured."""
    # seed_summary on a planted 7-day capture
    T = int(ED.T)
    # planted mid moves by -0.01 * q per 60 s, so TMB is negative
    samples = {}
    mid = 62000.0
    for t in range(ED.SAMPLE, T + 1, ED.SAMPLE):
        q = math.sin(t / 7919.0)
        samples[t] = (q, mid, 1.0 + 1e-6 * t)
        mid -= 0.01 * q
    row = seed_summary(samples, 999)
    assert row[HORIZON]["n"] == (T - ED.SAMPLE) // HORIZON == 167
    assert row[DESC_H]["n"] == (T - ED.SAMPLE) // DESC_H == 671
    for h in (HORIZON, DESC_H):
        xs, ys = ED.windows(samples, h)
        assert row[h]["tmb"] == ED.tmb(xs, ys, "p")
        assert row[h]["top"] - row[h]["bot"] == row[h]["tmb"]
        assert row[h]["tmb"] < 0
    assert abs(row["cost"] - statistics.fmean(
        [c for (_q, _m, c) in samples.values()])) == 0.0
    bad = dict(samples)
    bad[ED.SAMPLE * 10] = (0.1, 62000.0, float("nan"))
    try:
        seed_summary(bad, 999)
        raise AssertionError("a NaN cost was averaged")
    except ValueError:
        pass

    # anchor_check on the committed numbers
    with open(COMMITTED_FILE, encoding="utf-8") as fh:
        committed = parse_committed(fh.read().splitlines())
    good = _rows(8, ANCHOR, (1.4704, 0.0100), (-5.2016, 2.3827),
                 (-1.5313, 0.6218))
    fails, _o = _quiet(anchor_check, good, committed)
    assert fails == [], fails
    off = _rows(8, ANCHOR, (1.4704, 0.0100), (-5.2017, 2.3827),
                (-1.5313, 0.6218))
    fails, _o = _quiet(anchor_check, off, committed)
    assert fails == ["TMB h=3600 mean"], fails
    off = _rows(8, ANCHOR, (1.4705, 0.0100), (-5.2016, 2.3827),
                (-1.5313, 0.6219))
    fails, _o = _quiet(anchor_check, off, committed)
    assert fails == ["round-trip cost mean", "TMB h=900 se"], fails

    n = N_SCORED
    cost = (1.4704, 0.0029)
    # STRATEGY
    res, _o = _quiet(report, _rows(n, SCORED, cost, (-5.2016, 0.6878),
                                   (-1.5313, 0.18)), n)
    assert res["verdict"] == "STRATEGY AT 1h" and res["p2_verdict"] == \
        "STRATEGY AT 1h" and res[HORIZON]["r2_ok"]
    assert res[HORIZON]["sign_ok"] and res[HORIZON]["flips"] == []
    # INCONCLUSIVE: (P3) straddles, (P2) fires
    res, _o = _quiet(report, _rows(n, SCORED, cost, (-2.6008, 0.6878),
                                   (-1.5313, 0.18)), n)
    assert res["verdict"] == "INCONCLUSIVE", res["verdict"]
    assert res["p2_verdict"] == "INCONCLUSIVE"
    res, _o = _quiet(report, _rows(n, SCORED, cost, (-3.6, 0.6878),
                                   (-1.5313, 0.18)), n)
    assert res["verdict"] == "INCONCLUSIVE" and res["p2_verdict"] == \
        "STRATEGY AT 1h"
    # STRUCTURAL
    res, _o = _quiet(report, _rows(n, SCORED, cost, (-0.5, 0.3),
                                   (-0.2, 0.1)), n)
    assert res["verdict"] == "STRUCTURAL AT 1h"
    assert res[DESC_H]["p3"] == "BELOW"
    # sign failure
    res, _o = _quiet(report, _rows(n, SCORED, cost, (+5.2016, 0.6878),
                                   (-1.5313, 0.18)), n)
    assert not res[HORIZON]["sign_ok"] and res["verdict"] == "STRATEGY AT 1h"
    # asymmetric tails: top tail alone clears
    res, out = _quiet(report, _rows(n, SCORED, cost, (-2.6008, 0.6878),
                                    (-1.5313, 0.18), top_share=1.95), n)
    assert res["verdict"] == "INCONCLUSIVE"
    assert res[HORIZON]["one_tail"] == ["TOP"], res[HORIZON]["one_tail"]
    assert "a one-tail attacker" in out
    # flip between 1.96 and 1.985
    res, out = _quiet(report, _rows(n, SCORED, (1.0, 0.001), (-5.94, 2.0),
                                    (-1.5313, 0.18)), n)
    assert res["verdict"] == "STRATEGY AT 1h" and \
        res[HORIZON]["p3_t95"] == "STRADDLE" and "P3" in res[HORIZON]["flips"]
    # R2: one seed dominates
    rows = _rows(n, SCORED, cost, (-5.2016, 0.6878), (-1.5313, 0.18))
    rows[17][HORIZON]["tmb"] = -200.0
    rows[17][HORIZON]["bot"] = rows[17][HORIZON]["top"] + 200.0
    res, _o = _quiet(report, rows, n)
    assert not res[HORIZON]["r2_ok"] and res[HORIZON]["r2"][0] == SCORED[17]
    # missing seed and NaN refuse
    try:
        _quiet(report, rows[:-1], n)
        raise AssertionError("95 rows were accepted as 96")
    except ValueError:
        pass
    rows[3]["cost"] = float("nan")
    try:
        _quiet(report, rows, n)
        raise AssertionError("a NaN cost reached the verdict")
    except ValueError:
        pass
    print("  T-postsim    PASS  seed_summary on a planted 7-day capture: 167 / "
          "671 windows, TMB == ED.tmb, top-bot")
    print("                     == TMB, NaN cost raises. anchor_check passes "
          "planted committed values and fails")
    print("                     exactly the perturbed ones. report() returns "
          "STRATEGY / INCONCLUSIVE (P2 agreeing")
    print("                     and disagreeing) / STRUCTURAL, a sign failure, "
          "the one-tail flag, the 1.96/1.985")
    print("                     flip and an R2 failure; 95 rows and a NaN "
          "refuse. Output captured, not logged.")


def t_power():
    """Verdict probabilities at 96 seeds from the committed 8-seed numbers."""
    with open(COMMITTED_FILE, encoding="utf-8") as fh:
        committed = parse_committed(fh.read().splitlines())
    m8, _t, se8, _t2 = committed[HORIZON]
    c, _t3, sec8, _t4 = committed["cost"]
    k = math.sqrt(len(ANCHOR)) / math.sqrt(N_SCORED)
    s_half = 0.5 * se8 * k
    sc = sec8 * k
    c_hi, c_lo = c + Z * sc, c - Z * sc
    nd = NormalDist()
    print("  T-power      (no new data) per-seed SDs from the committed 8 seeds; "
          "SE(|TMB|/2) at 96 = %.4f," % s_half)
    print("               cost band at 96 = [%.4f, %.4f]:" % (c_lo, c_hi))
    for label, true in (("observed", abs(m8) / 2), ("half", abs(m8) / 4)):
        p_str = 1.0 - nd.cdf((c_hi + Z * s_half - true) / s_half)
        p_stc = nd.cdf((c_lo - Z * s_half - true) / s_half)
        print("               true TMB = %-8s (|TMB|/2 %.4f):  P(STRATEGY) "
              "%.3f  P(STRUCTURAL) %.3f  P(INCONCLUSIVE) %.3f"
              % (label, true, p_str, p_stc, 1.0 - p_str - p_stc))
    need = (Z * 0.5 * se8 * math.sqrt(len(ANCHOR))
            / (c - Z * sec8 * k - abs(m8) / 4)) ** 2
    print("               seeds for the half case's expected upper bound to "
          "fall below the cost: ~%.0f" % need)


def selftest():
    print("=" * 128)
    print("Self-tests: onehour_confirm.py. No scored seed has run.")
    print("=" * 128)
    print("  extraction_diag's own tests, imported unchanged:")
    ED.t_ident()
    ED.t_guards()
    ED.t_tmb()
    ED.t_windows()
    ED.t_walk()
    ED.t_verdict()
    print("  this script's tests:")
    t_ident()
    t_seeds()
    t_streams()
    t_stats()
    t_parse()
    t_postsim()
    t_power()
    print("  extraction_diag's mirror test, re-run first as registered:")
    ED.t_mirror()
    print("")
    print("  All self-tests PASSED.")
    return 0


def main():
    print("=" * 128)
    print("One-hour confirmation: cluster B, clip 0.12, control arm, 7 "
          "days, h = 3600 s scored")
    print("=" * 128)
    print("Working point lam=%.4f p_market=%.4f life=%.0fs disp=%.4f JOIN; "
          "gamma=%.4e k=%.5f; T=%.0fs; samples every %ds."
          % (ED.LAM, ED.P_MARKET, ED.LIFE, ED.DISP, ED.GAMMA, ED.K, ED.T,
             ED.SAMPLE))
    print("Anchor seeds %s (excluded from the verdict). Scored seeds: 113..201 "
          "+ 291..297 (%d), corrected" % (ANCHOR, len(SCORED)))
    print("before the run from the specified 100-195, which collides on 20 RNG "
          "streams (header (S)).")
    print("Pre-registered: (P3) per trade at Z=1.96 is the verdict; (P2) "
          "beside it; t(95)=1.985 flagged; tails")
    print("separately; R2 at 40%; 900 s descriptive. Estimated ~1 h 55 min.")
    print("")

    with open(COMMITTED_FILE, encoding="utf-8") as fh:
        committed = parse_committed(fh.read().splitlines())

    print("running the 8 anchor seeds ...", flush=True)
    t0 = time.time()
    anchor_rows = []
    for s in ANCHOR:
        t1 = time.time()
        row = run_seed(s)
        anchor_rows.append(row)
        print_row(row, "anchor", time.time() - t1)
    print("  anchor done in %.0fs" % (time.time() - t0))
    print("")
    print("=" * 128)
    print("Anchor: seeds 0-7 must reproduce "
          "extraction_diag_results.txt's clip-0.120 block at printed "
          "precision")
    print("=" * 128)
    fails = anchor_check(anchor_rows, committed)
    if fails:
        raise AssertionError("ANCHOR FAILED on %s -- this is not the committed "
                             "experiment. HALTING before any scored seed."
                             % ", ".join(fails))
    print("  Anchor HELD on all 6 checks.")
    print("")

    print("running the %d SCORED seeds ..." % len(SCORED), flush=True)
    t0 = time.time()
    rows = []
    for j, s in enumerate(SCORED):
        t1 = time.time()
        row = run_seed(s)
        rows.append(row)
        print_row(row, "scored", time.time() - t1)
        if (j + 1) % 12 == 0:
            el = time.time() - t0
            print("  -- %d/%d done, %.0fs elapsed, ~%.0fs to go"
                  % (j + 1, len(SCORED), el,
                     el / (j + 1) * (len(SCORED) - j - 1)), flush=True)
    print("  scored runs done in %.0fs" % (time.time() - t0))
    print("")
    print("=" * 128)
    print("Results: %d scored seeds" % len(SCORED))
    print("=" * 128)
    report(rows, N_SCORED)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
