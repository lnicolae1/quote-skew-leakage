# touch_geometry.py: touch geometry cited in the paper, measured (cluster B, control arm, JOIN, 8 seeds x 604800 s)
# - cited: "~1.46 bp from mid", "~0.156 bp market touch" (from a one-day controlled-quoter run),
#   "about nine times", "0.1244 BTC/side"
# - each two-sided second, bp of mid: MM half-spread, full-book and residual (non-MM) touch halves, ratio
# - per seed: median over seconds (primary) and mean; across seeds: mean +/- SE
# - T-MIRROR: seed 0 matches damage_qs12.run_ext to 1e-9
# - anchor: 8-seed dollar medians reproduce damage_qs12_results.txt (18.3620 / 2.4700) to half the
#   last printed digit, else halts
# pre-registered:
# - "1.46" survives iff 8-seed mean of per-seed median MM half-spread rounds to 1.46 (2 dp); else superseded
# - "0.156" survives iff same for residual touch half, rounded to 0.156 (3 dp); else superseded
# - "about nine times": 8-seed mean of per-seed median ratio, descriptive

import contextlib
import io
import math
import os
import re
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import damage_qs12 as QS
from clipped_placement import make_world
from clipped_mm_check import _resid_best, MM_ID
from damage_ab import (T, SEEDS, LAM, P_MARKET, LIFE, DISP, K, GAMMA, REF_MID,
                       mean_se)
from damage_qs12 import _finite, _ratio, THETA, HOLD
from market_maker import MarketMaker

MM_SIZE = 0.020
DEPTH_EVERY = 60
DEPTH_BAND_BP = 1.0
MIRROR_TOL = 1e-9
COMMITTED_FILE = os.path.join(_HERE, "damage_qs12_results.txt")
PROSE = {"mm_half": (1.46, 2), "touch_half": (0.156, 3), "ratio": 9.0,
         "near_btc": 0.1244}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def near_depth(eng, mid):
    """Non-MM resting size within DEPTH_BAND_BP of mid, side-averaged."""
    band = DEPTH_BAND_BP * 1e-4 * mid
    qb = qa = 0.0
    for o in eng.orders.values():
        if o.agent_id == MM_ID:
            continue
        if o.side == "buy" and mid - o.price <= band:
            qb += o.size
        elif o.side == "sell" and o.price - mid <= band:
            qa += o.size
    return 0.5 * (qb + qa)


def run_geometry(seed, mm_size=MM_SIZE):
    """run_ext control arm plus geometry capture."""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA, quote_size=mm_size)
    last_mid = REF_MID
    mm_spread, resid_touch = [], []
    mm_half, full_half, resid_half, ratio = [], [], [], []
    depth = []

    t = 0.0
    while t < T:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        last_mid = 0.5 * (bb + ba)
        full_half.append(0.5 * (ba - bb) / last_mid * 1e4)
        if int(t) % DEPTH_EVERY == 0:
            depth.append(near_depth(eng, last_mid))

        mmb = mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mmb = o.price if mmb is None else max(mmb, o.price)
            else:
                mma = o.price if mma is None else min(mma, o.price)
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if mmb is not None and mma is not None:
            mm_spread.append(mma - mmb)
            h = 0.5 * (mma - mmb) / last_mid * 1e4
            mm_half.append(h)
            if rb is not None and ra is not None and ra > rb:
                resid_touch.append(ra - rb)
                rh = 0.5 * (ra - rb) / (0.5 * (ra + rb)) * 1e4
                resid_half.append(rh)
                ratio.append(_ratio(h, rh, "MM / residual half-spread"))

    for series, nm in ((mm_half, "MM half"), (full_half, "full touch"),
                       (resid_half, "residual touch"), (depth, "depth")):
        if not series:
            raise ValueError("seed %d: empty %s series" % (seed, nm))
        for v in series:
            _finite(v, "%s seed %d" % (nm, seed))
    return {"seed": seed,
            "mm_spread_med": statistics.median(mm_spread),
            "resid_touch_med": statistics.median(resid_touch),
            "mm_pnl": mm.mark_to_market(last_mid),
            "mm_half_med": statistics.median(mm_half),
            "mm_half_mean": statistics.fmean(mm_half),
            "full_half_med": statistics.median(full_half),
            "full_half_mean": statistics.fmean(full_half),
            "resid_half_med": statistics.median(resid_half),
            "resid_half_mean": statistics.fmean(resid_half),
            "ratio_med": statistics.median(ratio),
            "near_btc": statistics.fmean(depth),
            "n_mm": len(mm_half), "n_full": len(full_half),
            "n_resid": len(resid_half)}


# --------------------------------------------------------------------------- #
# statistics, anchor, report
# --------------------------------------------------------------------------- #
def strict_mean_se(values, what, n_expected):
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
        raise ValueError("%s: SE is %r; no interval" % (what, se))
    return m, se


def _tol(s):
    return 0.5 * 10.0 ** (-len(s.split(".")[1])) if "." in s else 0.5


def parse_committed(lines):
    """Size-0.020 row of damage_qs12_results.txt: MM spread, residual touch, tolerances."""
    rows = [ln for ln in lines
            if re.match(r"^\s+0\.020\s+[0-9.]+%\s+-?[0-9.]+ \+/- ", ln)]
    if len(rows) != 1:
        raise ValueError("expected one size-0.020 caveat row, found %d"
                         % len(rows))
    toks = rows[0].split()
    spread, touch = toks[-2], toks[-1]
    return {"mm_spread": (float(spread), _tol(spread)),
            "resid_touch": (float(touch), _tol(touch))}


def anchor_check(rows, committed):
    n = len(SEEDS)
    got = {"mm_spread": strict_mean_se([r["mm_spread_med"] for r in rows],
                                       "MM spread $", n)[0],
           "resid_touch": strict_mean_se([r["resid_touch_med"] for r in rows],
                                         "residual touch $", n)[0]}
    fails = []
    for k in ("mm_spread", "resid_touch"):
        want, tol = committed[k]
        ok = abs(got[k] - want) <= tol
        print("  %-12s got %.6f  committed %.4f  tol %.0e  %s"
              % (k, got[k], want, tol, "OK" if ok else "*** MISS ***"))
        if not ok:
            fails.append(k)
    return fails


def report(rows):
    n = len(SEEDS)
    res = {}
    print("  per seed (median over seconds unless marked):")
    print("  %-5s %9s %9s %9s %9s %9s %9s %8s %9s" % (
        "seed", "MM half", "MM mean", "full tch", "resid tch", "resid mn",
        "ratio", "near BTC", "seconds"))
    for r in rows:
        print("  %-5d %9.4f %9.4f %9.4f %9.4f %9.4f %9.3f %8.4f %9d"
              % (r["seed"], r["mm_half_med"], r["mm_half_mean"],
                 r["full_half_med"], r["resid_half_med"],
                 r["resid_half_mean"], r["ratio_med"], r["near_btc"],
                 r["n_resid"]))
    print("")
    for key, label in (("mm_half_med", "MM half-spread, median (bp)"),
                       ("mm_half_mean", "MM half-spread, mean (bp)"),
                       ("full_half_med", "full-book touch half, median (bp)"),
                       ("full_half_mean", "full-book touch half, mean (bp)"),
                       ("resid_half_med", "residual touch half, median (bp)"),
                       ("resid_half_mean", "residual touch half, mean (bp)"),
                       ("ratio_med", "MM / residual touch, median ratio"),
                       ("near_btc", "near-touch non-MM BTC/side (<=1bp)")):
        m, se = strict_mean_se([r[key] for r in rows], label, n)
        res[key] = (m, se)
        print("  %-38s %9.4f +/- %.4f   (%d seeds)" % (label, m, se, n))
    print("")
    mm = res["mm_half_med"][0]
    tc = res["resid_half_med"][0]
    mm_ok = round(mm, PROSE["mm_half"][1]) == PROSE["mm_half"][0]
    tc_ok = round(tc, PROSE["touch_half"][1]) == PROSE["touch_half"][0]
    res["mm_ok"], res["tc_ok"] = mm_ok, tc_ok
    print("  '~1.46 bp from mid'     measured %.4f bp -> %s" % (
        mm, "SURVIVES" if mm_ok else "SUPERSEDED"))
    print("  '~0.156 bp market touch' measured %.4f bp (residual; full book "
          "%.4f) -> %s" % (tc, res["full_half_med"][0],
                           "SURVIVES" if tc_ok else
                           "SUPERSEDED"))
    print("  'about nine times'       measured %.2fx" % res["ratio_med"][0])
    print("  '0.1244 BTC/side'        measured %.4f BTC/side (descriptive; prose figure maker-absent)"
          % res["near_btc"][0])
    return res


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def t_parse():
    with open(COMMITTED_FILE, encoding="utf-8-sig") as f:
        c = parse_committed(f.read().splitlines())
    assert c["mm_spread"][0] == 18.3620 and c["resid_touch"][0] == 2.4700
    assert abs(c["mm_spread"][1] - 5e-5) < 1e-15
    planted = ["  0.120       4.5186%          -1058.55 +/- 388.57         347.7"
               "          18.3612           2.5625",
               "  0.020       4.5187%           -525.18 +/- 162.43         326.6"
               "          18.4               2.47"]
    p = parse_committed(planted)
    assert p["mm_spread"] == (18.4, 0.05) and abs(p["resid_touch"][1] - 5e-3) < 1e-15
    print("  T-PARSE     PASS  size-0.020 row -> 18.3620 / 2.4700, tol 5e-5; "
          "0.120 row ignored")


_PLANT_E = (-0.02, 0.02, -0.01, 0.01, -0.005, 0.005, 0.003, -0.003)  # sum 0


def _planted_rows(n=8):
    rows = []
    for i in range(n):
        e = _PLANT_E[i]
        rows.append({"seed": i, "mm_spread_med": 18.362 + e,
                     "resid_touch_med": 2.47 + e, "mm_pnl": 0.0,
                     "mm_half_med": 1.4808 + e, "mm_half_mean": 1.47 + e,
                     "full_half_med": 0.199 + e / 10,
                     "full_half_mean": 0.21 + e / 10,
                     "resid_half_med": 0.1992 + e / 10,
                     "resid_half_mean": 0.211 + e / 10,
                     "ratio_med": 7.4 + e, "near_btc": 0.12 + e / 10,
                     "n_mm": 1, "n_full": 1, "n_resid": 1})
    return rows


def t_postsim():
    with open(COMMITTED_FILE, encoding="utf-8-sig") as f:
        c = parse_committed(f.read().splitlines())
    rows = _planted_rows()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fails = anchor_check(rows, c)
        res = report(rows)
    assert fails == [], fails
    assert not res["mm_ok"] and not res["tc_ok"]
    rows2 = _planted_rows()
    for r in rows2:
        r["mm_half_med"] -= 0.0208
        r["resid_half_med"] -= 0.0432
    with contextlib.redirect_stdout(buf):
        res2 = report(rows2)
    assert res2["mm_ok"] and res2["tc_ok"], (res2["mm_half_med"],
                                             res2["resid_half_med"])
    off = _planted_rows()
    off[0]["mm_spread_med"] += 0.01
    with contextlib.redirect_stdout(buf):
        assert anchor_check(off, c) == ["mm_spread"]
    bad = _planted_rows()
    bad[3]["mm_half_med"] = float("nan")
    try:
        with contextlib.redirect_stdout(buf):
            report(bad)
        raise AssertionError("a NaN reached the report")
    except ValueError:
        pass
    print("  T-POSTSIM   PASS  planted rows: SUPERSEDED at 1.48/0.199, SURVIVES at "
          "1.46/0.156;")
    print("                    perturbed seed fails anchor; NaN raises")


def t_depth():
    from matching_engine import MatchingEngine
    eng = MatchingEngine()
    eng.add_limit_order("buy", 99.995, 0.3, "n1")    # 0.5 bp below 100
    eng.add_limit_order("buy", 99.98, 0.7, "n2")     # 2 bp below: outside
    eng.add_limit_order("sell", 100.009, 0.1, "n3")  # 0.9 bp above
    eng.add_limit_order("sell", 100.005, 0.2, MM_ID)  # the maker: excluded
    got = near_depth(eng, 100.0)
    assert abs(got - 0.5 * (0.3 + 0.1)) < 1e-12, got
    print("  T-DEPTH     PASS  near_depth counts non-MM size within 1 bp, "
          "side-averaged (%.2f)" % got)


def t_mirror():
    t0 = time.time()
    mine = run_geometry(0)
    theirs = QS.run_ext(0, False, THETA, HOLD, MM_SIZE)
    for k in ("mm_spread_med", "resid_touch_med", "mm_pnl"):
        a, b = mine[k], theirs[k]
        assert abs(a - b) <= MIRROR_TOL * max(1.0, abs(b)), \
            "mirror mismatch %s: %.12g vs %.12g" % (k, a, b)
    print("  T-MIRROR    PASS  seed 0: MM median spread, residual median touch, final PnL match")
    print("                    damage_qs12.run_ext's control arm to %.0e (%.0fs)."
          % (MIRROR_TOL, time.time() - t0))
    return mine


def selftest():
    print("=" * 110)
    print("Self-tests: touch_geometry.py")
    print("=" * 110)
    t_parse()
    t_depth()
    t_postsim()
    mine = t_mirror()
    print("")
    print("  self-tests PASS")
    return mine


def main(seed0=None):
    print("=" * 110)
    print("Touch geometry: cluster B, control arm, quote size %.3f, %d seeds x "
          "%.0fs" % (MM_SIZE, len(SEEDS), T))
    print("=" * 110)
    print("gamma=%.4e k=%.5f lam=%.3f p_market=%.4f life=%.0f disp=%.4f JOIN."
          % (GAMMA, K, LAM, P_MARKET, LIFE, DISP))
    print("Half-spreads in bp of mid; per-seed median over seconds primary.")
    print("")
    with open(COMMITTED_FILE, encoding="utf-8-sig") as f:
        committed = parse_committed(f.read().splitlines())
    rows = []
    t0 = time.time()
    for s in SEEDS:
        t1 = time.time()
        r = seed0 if (s == 0 and seed0 is not None) else run_geometry(s)
        rows.append(r)
        print("  seed %d  MM half %.4f  resid touch half %.4f  (%.0fs)"
              % (s, r["mm_half_med"], r["resid_half_med"], time.time() - t1),
              flush=True)
    print("  %d runs in %.0fs (seed 0 reused from T-MIRROR)" % (len(SEEDS),
                                                              time.time() - t0))
    print("")
    print("Anchor: 8-seed means of per-seed dollar medians vs "
          "damage_qs12_results.txt")
    fails = anchor_check(rows, committed)
    if fails:
        raise AssertionError("ANCHOR FAILED on %s; halting" % ", ".join(fails))
    print("  ANCHOR HELD.")
    print("")
    report(rows)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    s0 = selftest()
    print("")
    main(seed0=s0)
