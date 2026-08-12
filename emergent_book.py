# emergent_book.py: working point without sim_run._seed_book: are the targets met on a self-built book?

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sim_run
import clipped_depth_sweep as D
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from informed_traders import InformedTraderFlow, path_value
from clipped_placement import ClippedNoiseTraderFlow
from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from clipped_mm_check import MM_ID, _resid_best
from clipped_depth_sweep import run_measure, LAM
from clipped_window import verdict, margin_se, band, TARGETS, Z

P_MARKET = 0.0126
LIFE = 720.0
DISP = 0.0055
CLIP = "join"

WARMUP = 3600.0
SAMPLE_EVERY = 60
T_DAY = 86400.0
T_WEEK = 604800.0
SEEDS = list(range(15))
SMOKE_SEEDS = [0, 1]

K = 0.17763
GAMMA = 1.3391837316766703e-06

MKT_RATE = LAM * P_MARKET

SIM_SEEDED_NEAR_BTC = 0.1244
REAL_NEAR_BTC_BID = 0.0452
REAL_NEAR_BTC_ASK = 0.0247
REAL_NEAR_BTC_AVG = 0.5 * (REAL_NEAR_BTC_BID + REAL_NEAR_BTC_ASK)
SIM_SEEDED_NEAR_LVL = 3.1668
SIM_SEEDED_TOUCH_AGE = 395.93
REAL_NEAR_LVL_AVG = 0.5 * (2.15 + 1.95)

SNAP_TIMES = frozenset([600.0, 1800.0, 3600.0, 7200.0, 21600.0, 43200.0,
                        86400.0])


def make_world_emergent(seed, t_total, lam, p_market, life, disp, clip_mode,
                        seed_book=False):
    """Mirrors clipped_placement.make_world EXACTLY except for the seed book"""
    n = int(t_total)
    path = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE, 1.0, n,
                               seed=sim_run.SEED_V_BASE + 1000 * seed)
    assert len(path) == n + 1, (
        "value path length %d != n+1 = %d" % (len(path), n + 1))
    assert int(t_total / 1.0) < len(path), (
        "value path too short: vf(%r) would index %d into a path of length %d "
        "and CLAMP silently" % (t_total, int(t_total), len(path)))
    vf = path_value(path, 1.0)

    eng = MatchingEngine()
    if seed_book:
        sim_run._seed_book(eng)

    kw = dict(lam=lam, p_market=p_market, mean_lifetime=life, disp=disp,
              value_fn=vf, reference_price=sim_run.REFERENCE_PRICE,
              seed=sim_run.SEED_NOISE_BASE + 1000 * seed)
    noise = ClippedNoiseTraderFlow(clip_mode=clip_mode, **kw)
    informed = InformedTraderFlow(seed=sim_run.SEED_INF_BASE + 1000 * seed)
    return path, vf, eng, noise, informed


def classify(rec):
    """The channel of one flow record, as a printable (order_type, action)"""
    ot = getattr(rec, "order_type", None)
    if ot is not None:
        return "noise/%s" % ot
    return "informed/%s" % getattr(rec, "action", None)


class Observer:
    """Wraps noise.run_until and informed.run_until"""

    def __init__(self, eng, noise, informed, mm, warmup):
        self.eng = eng
        self.mm = mm
        self.warmup = warmup

        self.chan = {}
        self.n_by_chan = {}
        self.first_t = {}
        self.n_fill_events = 0
        self.first_fill_t = None
        self.two_sided_at = None
        self.n_fill_events_before_two_sided = 0

        self.snapshot = None
        self.snapshots = {}

        self.t_series = []
        self.w_series = []
        self.resid_series = []

        self._orig_noise = noise.run_until
        self._orig_inf = informed.run_until
        noise.run_until = self._noise_hook
        informed.run_until = self._inf_hook

    def _account(self, recs, t):
        for r in recs:
            ch = classify(r)
            self.n_by_chan[ch] = self.n_by_chan.get(ch, 0) + 1
            if ch not in self.first_t:
                self.first_t[ch] = t
            if getattr(r, "fills", None):
                self.n_fill_events += 1
                self.chan[ch] = self.chan.get(ch, 0) + 1
                if self.first_fill_t is None:
                    self.first_fill_t = t
                if self.two_sided_at is None:
                    self.n_fill_events_before_two_sided += 1

    def _book_fills(self, recs):
        """on_fills filters internally on counterparty_id == MM_ID"""
        for r in recs:
            if getattr(r, "fills", None):
                self.mm.on_fills(r.fills, r.side)

    def _noise_hook(self, engine, t_end):
        recs = self._orig_noise(engine, t_end)
        self._account(recs, t_end)
        if self.mm is not None and t_end > self.warmup:
            self._book_fills(recs)
        return recs

    def _inf_hook(self, engine, t_end, value_fn):
        recs = self._orig_inf(engine, t_end, value_fn)
        self._account(recs, t_end)
        if self.mm is not None and t_end > self.warmup:
            self._book_fills(recs)
            self.mm.requote(engine, t_end)

        bb, ba = engine.best_bid(), engine.best_ask()
        if bb is not None and ba is not None and self.two_sided_at is None:
            self.two_sided_at = t_end

        if t_end == self.warmup:
            self.snapshot = self._scan(bb, ba)
        if t_end in SNAP_TIMES:
            self.snapshots[t_end] = self._scan(bb, ba)

        if t_end > self.warmup and bb is not None and ba is not None:
            mid = 0.5 * (bb + ba)
            self.t_series.append(t_end)
            self.w_series.append(1e4 * (ba - bb) / mid)
            rb = _resid_best(engine.bids, bb, True)
            ra = _resid_best(engine.asks, ba, False)
            if rb is not None and ra is not None and ra > rb:
                self.resid_series.append(1e4 * (ra - rb) / (0.5 * (rb + ra)))
        return recs

    def _scan(self, bb, ba):
        """Book state at t=W. Same convention as run_measure's scan: levels"""
        eng = self.eng
        if bb is None or ba is None:
            return {"two_sided": False}
        mid = 0.5 * (bb + ba)
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        out = {"two_sided": True, "mid": mid,
               "width_bp": 1e4 * (ba - bb) / mid,
               "best_bid": bb, "best_ask": ba}
        for name, book, is_bid in (("bid", eng.bids, True),
                                   ("ask", eng.asks, False)):
            lv = nr_lv = 0
            sz = nr_sz = 0.0
            for p, q in book.items():
                live = list(q)
                if not live:
                    continue
                lv += 1
                s = sum(o.size for o in live)
                sz += s
                if (is_bid and p >= lo) or ((not is_bid) and p <= hi):
                    nr_lv += 1
                    nr_sz += s
            out["levels_" + name] = lv
            out["btc_" + name] = sz
            out["near_levels_" + name] = nr_lv
            out["near_btc_" + name] = nr_sz
        return out


AGG_KEYS = ("median_bp", "mean_bp", "shape", "levels_per_side",
            "orders_per_side", "agg_per_s", "ev_per_s", "near_btc",
            "near_lvl", "touch_age", "clipped_pct")


def per_seed_stats(x):
    """The per-seed reduction clipped_window.measure applies to run()'s dict"""
    sp = sorted(x["spreads_bp"])
    med = sp[len(sp) // 2]
    mean = statistics.mean(sp)
    out = {"median_bp": med, "mean_bp": mean,
           "shape": mean / med if med else float("nan")}
    for k in ("levels_per_side", "orders_per_side", "agg_per_s", "ev_per_s",
              "near_btc", "near_lvl", "touch_age", "clipped_pct"):
        out[k] = x[k]
    return out


def aggregate(rows):
    """Mean and SE across seeds, never across overlapping windows"""
    n = len(rows)
    out = {"n": n}
    for k in AGG_KEYS:
        v = [r[k] for r in rows]
        out[k] = statistics.mean(v)
        out[k + "_se"] = (statistics.stdev(v) / (n ** 0.5)) if n > 1 else 0.0
    return out


def day_blocks(ts, vals, warmup):
    """Split a measured series into blocks of one day by absolute timestamp"""
    out = {}
    for t, v in zip(ts, vals):
        out.setdefault(int((t - warmup - 1) // int(T_DAY)), []).append(v)
    return [out[k] for k in sorted(out)]


def run_one(seed, T, use_mm, warmup=WARMUP, seed_book=False, raw=False,
            keep_prefix=0):
    t_total = warmup + T
    _p, vf, eng, noise, informed = make_world_emergent(
        seed, t_total, LAM, P_MARKET, LIFE, DISP, CLIP, seed_book=seed_book)

    mm = None
    if use_mm:
        mm = MarketMaker(horizon=t_total, k=K, gamma=GAMMA,
                         quote_size=DEFAULT_QUOTE_SIZE)

    obs = Observer(eng, noise, informed, mm, warmup)
    t0 = time.time()
    res = run_measure(vf, eng, noise, informed, T, SAMPLE_EVERY, warmup=warmup)
    secs = time.time() - t0

    if raw:
        return res

    out = per_seed_stats(res)
    out["_seed"] = seed
    out["secs"] = secs
    out["two_sided_pct"] = res["two_sided_pct"]
    out["snapshot"] = obs.snapshot
    out["snapshots"] = dict(obs.snapshots)
    out["agg_counts"] = dict(obs.chan)
    out["chan_totals"] = dict(obs.n_by_chan)
    out["first_t"] = dict(obs.first_t)
    out["first_fill_t"] = obs.first_fill_t
    out["two_sided_at"] = obs.two_sided_at
    out["n_before_two_sided"] = obs.n_fill_events_before_two_sided
    out["series_matches_run_measure"] = (obs.w_series == res["spreads_bp"])
    out["day_medians"] = [sorted(b)[len(b) // 2]
                          for b in day_blocks(obs.t_series, obs.w_series, warmup)]
    out["resid_median"] = (statistics.median(obs.resid_series)
                           if obs.resid_series else float("nan"))
    out["n_resid"] = len(obs.resid_series)
    if keep_prefix:
        out["prefix_w"] = obs.w_series[:keep_prefix]
        out["prefix_t"] = obs.t_series[:keep_prefix]
    return out


def self_test():
    print("=" * 100)
    print("Self-test; every detector and counter shown firing on a "
          "constructed positive and negative case")
    print("=" * 100)
    ok = True

    _, _, e0, _, _ = make_world_emergent(0, 100.0, LAM, P_MARKET, LIFE, DISP,
                                         CLIP, seed_book=False)
    _, _, e1, _, _ = make_world_emergent(0, 100.0, LAM, P_MARKET, LIFE, DISP,
                                         CLIP, seed_book=True)
    print("  1. empty-book builder      : orders at t=0 = %d   (expect 0)"
          % len(e0.orders))
    print("     seeded builder (control): orders at t=0 = %d   (expect 10)"
          % len(e1.orders))
    ok &= (len(e0.orders) == 0 and len(e1.orders) == 10)

    fired = False
    try:
        short = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE,
                                    1.0, 50, seed=1)
        assert int(200.0 / 1.0) < len(short), "would clamp"
    except AssertionError:
        fired = True
    print("  2. path-length assertion   : fires on undersized path = %s   "
          "(expect True)" % fired)
    ok &= fired

    class _N(object):
        def __init__(self, ot, f):
            self.order_type, self.fills, self.side = ot, f, "buy"

    class _I(object):
        def __init__(self, ac, f):
            self.action, self.fills, self.side = ac, f, "sell"

    print("  3. classify(); channel only; noise/market and informed/take "
          "must stay separate channels")
    for rec, want_ch in [(_N("limit", []), "noise/limit"),
                         (_N("limit", [1]), "noise/limit"),
                         (_N("market", [1]), "noise/market"),
                         (_I("take", [1]), "informed/take"),
                         (_I("post", [1]), "informed/post"),
                         (_I(None, []), "informed/None")]:
        got_ch = classify(rec)
        good = (got_ch == want_ch)
        ok &= good
        print("     -> %-16s  expect %-16s  %s"
              % (got_ch, want_ch, "ok" if good else "FAIL"))

    class _Stub(object):
        def __init__(self):
            self.chan, self.n_by_chan, self.first_t = {}, {}, {}
            self.n_fill_events, self.first_fill_t = 0, None
            self.two_sided_at, self.n_fill_events_before_two_sided = None, 0
        _account = Observer._account

    s = _Stub()
    s._account([_N("limit", []), _N("market", []), _N("market", [1]),
                _I("take", [1])], 7.0)
    good = (s.n_by_chan == {"noise/limit": 1, "noise/market": 2,
                            "informed/take": 1}
            and s.chan == {"noise/market": 1, "informed/take": 1}
            and s.n_fill_events == 2 and s.first_fill_t == 7.0
            and s.first_t == {"noise/limit": 7.0, "noise/market": 7.0,
                              "informed/take": 7.0}
            and s.n_fill_events_before_two_sided == 2)
    print("  4. aggression counter      : totals=%r" % (s.n_by_chan,))
    print("     fill-bearing=%r n_fill=%d before_two_sided=%d  -> %s"
          % (s.chan, s.n_fill_events, s.n_fill_events_before_two_sided,
             "ok" if good else "FAIL"))
    print("     expect totals {'noise/limit':1,'noise/market':2,"
          "'informed/take':1}, fill-bearing {'noise/market':1,"
          "'informed/take':1}")
    ok &= good
    s2 = _Stub()
    s2._account([_N("limit", [])], 1.0)
    neg = (s2.chan == {} and s2.n_fill_events == 0
           and s2.n_by_chan == {"noise/limit": 1})
    print("     negative (one non-filling limit): fill-bearing=%r n_fill=%d "
          "totals=%r  -> %s"
          % (s2.chan, s2.n_fill_events, s2.n_by_chan, "ok" if neg else "FAIL"))
    ok &= neg

    eng = MatchingEngine()
    eng.add_limit_order("buy", 61999.00, 0.30, "x")
    eng.add_limit_order("buy", 61990.00, 0.70, "x")
    eng.add_limit_order("sell", 62001.00, 0.50, "x")
    obs = Observer.__new__(Observer)
    obs.eng = eng
    snap = Observer._scan(obs, eng.best_bid(), eng.best_ask())
    good = (snap["levels_bid"] == 2 and snap["levels_ask"] == 1
            and abs(snap["btc_bid"] - 1.0) < 1e-9
            and abs(snap["btc_ask"] - 0.5) < 1e-9
            and snap["near_levels_bid"] == 1 and snap["near_levels_ask"] == 1
            and abs(snap["near_btc_bid"] - 0.30) < 1e-9)
    print("  5. book scan               : levels %d/%d  btc %.2f/%.2f  "
          "near levels %d/%d  near btc %.2f/%.2f  -> %s"
          % (snap["levels_bid"], snap["levels_ask"], snap["btc_bid"],
             snap["btc_ask"], snap["near_levels_bid"], snap["near_levels_ask"],
             snap["near_btc_bid"], snap["near_btc_ask"],
             "ok" if good else "FAIL"))
    print("     expect levels 2/1, btc 1.00/0.50, near levels 1/1, near btc "
          "0.30/0.50  (61990 is $10 out, past the $6.20 1bp edge)")
    ok &= good

    same = ([1.0, 2.0, 3.0] == [1.0, 2.0, 3.0, 4.0][:3])
    diff = ([1.0, 2.0, 9.0] == [1.0, 2.0, 3.0, 4.0][:3])
    print("  6. prefix comparison       : identical -> %s (expect True), "
          "perturbed -> %s (expect False)" % (same, diff))
    ok &= (same and not diff)

    ts = [WARMUP + 1 + i for i in range(int(3 * T_DAY))]
    blocks = day_blocks(ts, [float(i) for i in range(len(ts))], WARMUP)
    print("  7. daily blocking          : %d blocks, sizes %s  (expect 3 x %d)"
          % (len(blocks), [len(b) for b in blocks], int(T_DAY)))
    ok &= (len(blocks) == 3 and all(len(b) == int(T_DAY) for b in blocks))

    mm = MarketMaker(horizon=WARMUP + T_DAY, k=K, gamma=GAMMA,
                     quote_size=DEFAULT_QUOTE_SIZE)
    sig = mm.sigma_absolute(sim_run.REFERENCE_PRICE)
    tau0 = mm.time_remaining_years(WARMUP + 1.0)
    tau1 = mm.time_remaining_years(WARMUP + T_DAY)
    print("  8. MM API probe            : sigma_absolute(%.0f)=%.4f  "
          "tau(first measured)=%.8f  tau(last measured)=%.8f"
          % (sim_run.REFERENCE_PRICE, sig, tau0, tau1))
    print("     C = gamma*sigma$^2*tau : first measured %.6f, last %.6f  "
          "(tau must run T->0 across the measured window)"
          % (GAMMA * sig * sig * tau0, GAMMA * sig * sig * tau1))
    ok &= (tau0 > 0.0 and abs(tau1) < 1e-12)

    one = aggregate([{k: 1.0 for k in AGG_KEYS}])
    two = aggregate([{k: 1.0 for k in AGG_KEYS}, {k: 3.0 for k in AGG_KEYS}])
    good = (one["median_bp_se"] == 0.0 and two["median_bp"] == 2.0
            and abs(two["median_bp_se"] - 1.0) < 1e-12)
    print("  9. aggregate() guard       : n=1 se=%.4f (expect 0.0), "
          "n=2 mean=%.4f se=%.4f (expect 2.0, 1.0)  -> %s"
          % (one["median_bp_se"], two["median_bp"], two["median_bp_se"],
             "ok" if good else "FAIL"))
    ok &= good

    print("\n  self-test: %s" % ("all PASS" if ok else "failure; stopping"))
    return ok


def harness_validation(seeds):
    """Run the MM-absent 1-day path at warmup=0 with a seed book through this"""
    print("=" * 100)
    print("Harness validation; this module's builder + Observer + "
          "run_measure must reproduce committed run() on all %d seeds"
          % len(seeds))
    print("=" * 100)
    all_ok = True
    rows = []
    for s in seeds:
        want = D.run(s, LAM, P_MARKET, LIFE, DISP)
        got = run_one(s, T_DAY, use_mm=False, warmup=0.0, seed_book=True,
                      raw=True)
        same = (want == got)
        all_ok &= same
        bad = [k for k in want if want[k] != got.get(k)]
        sp = sorted(got["spreads_bp"])
        print("  seed %2d  dict equal = %-5s  median=%.10f  %s"
              % (s, same, sp[len(sp) // 2],
                 "" if same else "differing keys: %r" % bad))
        rows.append(per_seed_stats(got))
    agg = aggregate(rows)
    print("\n  aggregation check; must reproduce Gate B:")
    for name, key, tgt, tol in TARGETS:
        print("    %-20s %12.5f +/- %-10.5f" % (name, agg[key], agg[key + "_se"]))
    if len(seeds) == 15:
        print("    expect (15 seeds)      0.28879 +/- 0.01276 | "
              "551.23755 +/- 2.53988 | 0.04464 +/- 0.00073 | 2.49631 +/- 0.08231")
    else:
        print("    (%d seeds; NOT comparable to the 15-seed Gate B figures; "
              "the per-seed dict equality above is the check here)" % len(seeds))
    print("\n  seeded-book BASELINE for the not-scored diagnostics "
          "(%d seeds, 1 day, MM-absent, warmup=0):" % len(seeds))
    print("    near_btc  %10.5f +/- %-9.5f   (clipped_mm_check.ABSENT "
          "records %.4f)" % (agg["near_btc"], agg["near_btc_se"],
                             SIM_SEEDED_NEAR_BTC))
    print("    near_lvl  %10.4f +/- %-9.4f   (records %.4f)"
          % (agg["near_lvl"], agg["near_lvl_se"], SIM_SEEDED_NEAR_LVL))
    print("    touch_age %10.2f +/- %-9.2f   (records %.2f)"
          % (agg["touch_age"], agg["touch_age_se"], SIM_SEEDED_TOUCH_AGE))
    print("\n  harness validation: %s"
          % ("all %d cells equal" % len(seeds) if all_ok
             else "failure; stopping"))
    return all_ok


def report_warmup(tag, r, T):
    print("  [%s] warm-up verification; single seed (seed %d), not an "
          "aggregate" % (tag, r["_seed"]))
    print("    first fill-bearing record at t = %r s" % (r["first_fill_t"],))
    n_nm = r["chan_totals"].get("noise/market", 0)
    exp_nm = MKT_RATE * (WARMUP + T)
    sd_nm = exp_nm ** 0.5
    print("    first noise market order at t  = %r s   (analytic mean gap "
          "1/%.7f = %.1f s)"
          % (r["first_t"].get("noise/market"), MKT_RATE, 1.0 / MKT_RATE))
    print("    noise market orders total      = %d   vs analytic %.0f over "
          "%.0fs  (%+.1f%%, %+.2f sigma)"
          % (n_nm, exp_nm, WARMUP + T,
             100.0 * (n_nm - exp_nm) / exp_nm if exp_nm else float("nan"),
             (n_nm - exp_nm) / sd_nm if sd_nm else float("nan")))
    print("    informed takes total           = %d   (separate stream, NOT "
          "covered by lam*p_market)"
          % r["chan_totals"].get("informed/take", 0))
    print("    book first two-sided at t      = %r s" % (r["two_sided_at"],))
    print("    fill-bearing records before that = %d" % r["n_before_two_sided"])
    print("      note: reported, not asserted. A marketable order can cross a "
          "one-sided book, so a nonzero")
    print("      count here is possible in principle and is informative "
          "rather than automatically a bug.")
    print("    observer series == run_measure spreads_bp : %s   (proves the "
          "daily blocking uses the committed metric at the committed "
          "instants, not a parallel reimplementation)"
          % r["series_matches_run_measure"])
    print("    records by (order_type, action); all / fill-bearing:")
    for ch in sorted(r["chan_totals"]):
        print("      %-18s %8d / %-8d  first at t=%r"
              % (ch, r["chan_totals"][ch], r["agg_counts"].get(ch, 0),
                 r["first_t"].get(ch)))
    s = r["snapshot"]
    print("    snapshot at t=W=%.0f; single instant, single seed. Same book "
          "convention as the measurement" % WARMUP)
    print("    scans, reported per side. An instantaneous book statistic is A "
          "draw, NOT A state: the near-touch")
    print("    region holds only one or two orders per side, so a single "
          "arrival swings these by orders of")
    print("    magnitude. Measured across 3 seeds x 7 times, the ask/bid "
          "near-touch ratio ranged 0.0 to 429")
    print("    in both directions with no consistent side. Read near_btc in "
          "the verdict block, not this.")
    if not s or not s.get("two_sided"):
        print("      Book not two-sided at W; the warm-up did not build a "
              "book")
        return
    print("      touch width            %.4f bp   (bid %.2f / ask %.2f)"
          % (s["width_bp"], s["best_bid"], s["best_ask"]))
    print("      levels    bid/ask      %d / %d" % (s["levels_bid"], s["levels_ask"]))
    print("      levels within 1bp      %d / %d"
          % (s["near_levels_bid"], s["near_levels_ask"]))
    print("      resting BTC bid/ask    %.5f / %.5f" % (s["btc_bid"], s["btc_ask"]))
    print("      near-touch BTC bid/ask %.5f / %.5f   (real Binance.US 1bp "
          "bucket: bid %.4f / ask %.4f)"
          % (s["near_btc_bid"], s["near_btc_ask"], REAL_NEAR_BTC_BID,
             REAL_NEAR_BTC_ASK))


def report_verdicts(tag, agg):
    print("  [%s] verdicts; committed clipped_window.verdict, Z=%.2f, "
          "full-book width, %d seeds" % (tag, Z, agg["n"]))
    met = 0
    for name, key, tgt, tol in TARGETS:
        pt, se = agg[key], agg[key + "_se"]
        lo_b, hi_b = band(tgt, tol)
        v = verdict(pt, se, tgt, tol)
        met += (v == "MET")
        direction = ""
        if v == "MISSED":
            direction = ("  BELOW floor, ratio %.4f" % (pt / lo_b) if pt < lo_b
                         else "  ABOVE ceiling, ratio %.4f" % (pt / hi_b))
        print("    %-20s %12.5f +/- %-10.5f  95%% [%11.5f, %11.5f]  band "
              "[%11.5f, %11.5f]  %-10s margin %6.2f SE%s"
              % (name, pt, se, pt - Z * se, pt + Z * se, lo_b, hi_b, v,
                 margin_se(pt, se, tgt, tol), direction))
    print("    -> %d/4 MET" % met)
    print("    diagnostic, not scored (no band was ever fitted to these):")
    print("      near_btc  (side-avg BTC within 1bp)  %10.5f +/- %-9.5f"
          % (agg["near_btc"], agg["near_btc_se"]))
    print("        vs real Binance.US side-avg %.5f  (bid %.4f / ask %.4f, "
          "spread_depth_results 0-1bp)  ratio %.2fx"
          % (REAL_NEAR_BTC_AVG, REAL_NEAR_BTC_BID, REAL_NEAR_BTC_ASK,
             agg["near_btc"] / REAL_NEAR_BTC_AVG))
    print("        vs sim seeded book %.5f  (clipped_mm_check.ABSENT, 847a7fa "
          "-- a sim number, NOT a target)  ratio %.2fx"
          % (SIM_SEEDED_NEAR_BTC, agg["near_btc"] / SIM_SEEDED_NEAR_BTC))
    print("        note: near_btc excludes seed orders, so the seeded book's "
          "0.10 BTC/side never entered")
    print("        the 0.1244 either. Any move here is second-order through "
          "the dynamics, not direct removal.")
    print("      near_lvl  (side-avg levels within 1bp) %8.4f +/- %-9.4f"
          "  vs real %.2f (ratio %.2fx), vs sim seeded %.4f (ratio %.2fx)"
          % (agg["near_lvl"], agg["near_lvl_se"], REAL_NEAR_LVL_AVG,
             agg["near_lvl"] / REAL_NEAR_LVL_AVG, SIM_SEEDED_NEAR_LVL,
             agg["near_lvl"] / SIM_SEEDED_NEAR_LVL))
    print("      touch_age (median s at the touch)    %10.2f +/- %-9.2f"
          "  vs sim seeded %.2f (ratio %.2fx)"
          % (agg["touch_age"], agg["touch_age_se"], SIM_SEEDED_TOUCH_AGE,
             agg["touch_age"] / SIM_SEEDED_TOUCH_AGE))
    return met


def report_snapshots(seeds=(0, 1, 2)):
    """Multi-time book probe, MM-absent, 1 day, at absolute times spanning the"""
    print("=" * 100)
    print("Multi-time book PROBE; MM-ABSENT, 1 day, %d seeds. Is the "
          "warm-up asymmetry transient, structural, or seed 0?" % len(seeds))
    print("=" * 100)
    print("  W=%.0f, so t<=3600 is warm-up and t>=7200 is inside the measured "
          "window." % WARMUP)
    print("")
    times = sorted(SNAP_TIMES)
    for s in seeds:
        r = run_one(s, T_DAY, use_mm=False)
        print("  Seed %d   (%.1fs)" % (s, r["secs"]))
        print("    %8s %10s %14s %14s %12s %12s %10s"
              % ("t", "width bp", "levels b/a", "lvl<1bp b/a", "near BTC b",
                 "near BTC a", "ratio a/b"))
        for t in times:
            snap = r["snapshots"].get(t)
            if not snap or not snap.get("two_sided"):
                print("    %8.0f  book not two-sided" % t)
                continue
            nb, na = snap["near_btc_bid"], snap["near_btc_ask"]
            ratio = (na / nb) if nb > 0 else float("inf")
            print("    %8.0f %10.4f %14s %14s %12.5f %12.5f %10.1f"
                  % (t, snap["width_bp"],
                     "%d/%d" % (snap["levels_bid"], snap["levels_ask"]),
                     "%d/%d" % (snap["near_levels_bid"],
                                snap["near_levels_ask"]),
                     nb, na, ratio))
        print("")
    print("  real Binance.US near-touch resting size within 1bp: bid %.4f / "
          "ask %.4f BTC (side-avg %.4f)."
          % (REAL_NEAR_BTC_BID, REAL_NEAR_BTC_ASK, REAL_NEAR_BTC_AVG))
    print("  A ratio near 1 is symmetric. Seed 0 at t=3600 gave 429.")
    return 0


def main(smoke=False, snapshots=False):
    print("=" * 100)
    print("Emergent book; the working point without sim_run._seed_book")
    print("=" * 100)
    print("lam=%.4f p_market=%.4f mean_lifetime=%.0fs disp=%.4f %s clipping. "
          "Nothing retuned." % (LAM, P_MARKET, LIFE, DISP, CLIP.upper()))
    print("W=%.0fs warm-up, measured window starts after it, all rates over T."
          % WARMUP)
    print("MM arm: k=%.5f gamma=%.6g, horizon=W+T so tau runs T->0 across the "
          "measured window." % (K, GAMMA))
    print("")
    print("!! 847a7fa's bands were fitted MM-ABSENT. The MM-present arm below "
          "is scored against bands never")
    print("!! fitted to it. Pre-existing (8ddf8e8 did the same); stated here "
          "so it is on the page.")
    print("!! k=0.17763 and clipped_mm_check.ABSENT were both measured through "
          "make_world on a seeded book")
    print("!! and are used as-is, not re-derived here.")
    print("")

    if not self_test():
        return 1
    print("")
    if snapshots:
        return report_snapshots()
    if not harness_validation(SMOKE_SEEDS if smoke else SEEDS):
        return 1
    print("")

    if smoke:
        print("=" * 100)
        print("Smoke; seed 0, 1 day, both arms. Plumbing only, NOT "
              "reportable.")
        print("=" * 100)
        for use_mm, tag in ((False, "MM-ABSENT"), (True, "MM-PRESENT")):
            r = run_one(0, T_DAY, use_mm)
            print("  %s  %.1fs  median=%.5f  levels=%.3f  agg/s=%.5f  "
                  "shape=%.4f  two_sided=%.2f%%  resid n=%d"
                  % (tag, r["secs"], r["median_bp"], r["levels_per_side"],
                     r["agg_per_s"], r["shape"], r["two_sided_pct"],
                     r["n_resid"]))
            report_warmup(tag, r, T_DAY)
            print("")
        return 0

    store = {}
    for T, tlab in ((T_DAY, "1d"), (T_WEEK, "7d")):
        for use_mm, mlab in ((False, "MM-ABSENT"), (True, "MM-PRESENT")):
            kp = int(T_DAY) if not use_mm else 0
            rows = []
            t0 = time.time()
            for s in SEEDS:
                rows.append(run_one(s, T, use_mm, keep_prefix=kp))
                print("  ran %s %s seed %d (%.0fs elapsed)"
                      % (tlab, mlab, s, time.time() - t0), flush=True)
            store[(tlab, mlab)] = rows
    print("")

    for tlab in ("1d", "7d"):
        for mlab in ("MM-ABSENT", "MM-PRESENT"):
            rows = store[(tlab, mlab)]
            print("=" * 100)
            print("%s  %s" % (tlab, mlab))
            print("=" * 100)
            report_warmup("%s %s" % (tlab, mlab), rows[0],
                          T_DAY if tlab == "1d" else T_WEEK)
            bad = [r["_seed"] for r in rows
                   if not r["series_matches_run_measure"]]
            print("    observer/run_measure series agreement across all %d "
                  "seeds: %s" % (len(rows), "all match" if not bad
                                 else "mismatch on seeds %r" % bad))
            report_verdicts("%s %s" % (tlab, mlab), aggregate(rows))
            resid = [r["resid_median"] for r in rows if r["n_resid"]]
            if resid:
                print("    secondary (never scored): median resid_bp "
                      "(MM-excluded) = %.5f across %d seeds"
                      % (statistics.mean(resid), len(resid)))
            print("")

    print("=" * 100)
    print("T-independence gate; MM-ABSENT only (tau genuinely differs with "
          "the MM, so it does not apply there)")
    print("=" * 100)
    r1 = {r["_seed"]: r for r in store[("1d", "MM-ABSENT")]}
    r7 = {r["_seed"]: r for r in store[("7d", "MM-ABSENT")]}
    all_ok = True
    for s in SEEDS:
        a_w, a_t = r1[s]["prefix_w"], r1[s]["prefix_t"]
        b_w, b_t = r7[s]["prefix_w"], r7[s]["prefix_t"]
        same = (a_w == b_w[:len(a_w)] and a_t == b_t[:len(a_t)])
        all_ok &= same
        print("  seed %2d  1d reproduces day 1 of 7d exactly: %-5s  "
              "(n=%d vs prefix %d)" % (s, same, len(a_w), len(b_w[:len(a_w)])))
    print("  gate: %s" % ("PASS" if all_ok else "FAIL"))

    s0 = SEEDS[0]
    a_w = r1[s0]["prefix_w"]
    b_bad = list(r7[s0]["prefix_w"])
    if a_w and len(b_bad) >= len(a_w):
        i = len(a_w) // 2
        b_bad[i] = b_bad[i] + 1.0
        neg = (a_w == b_bad[:len(a_w)])
        print("  Negative control: element %d of the 7d prefix perturbed by "
              "+1.0 -> comparison returns %s (expect False)" % (i, neg))
        all_ok &= (neg is False)
    else:
        print("  negative control: skipped; empty or short prefix, which is "
              "itself a failure")
        all_ok = False
    print("  gate including negative control: %s"
          % ("PASS" if all_ok else "FAIL"))
    print("")

    print("=" * 100)
    print("Convergence; primary: per-day blocks within the 7-day run")
    print("=" * 100)
    for mlab in ("MM-ABSENT", "MM-PRESENT"):
        rows = store[("7d", mlab)]
        print("  %s" % mlab)
        for d in range(7):
            v = [r["day_medians"][d] for r in rows if len(r["day_medians"]) > d]
            if v:
                se = statistics.stdev(v) / (len(v) ** 0.5) if len(v) > 1 else 0.0
                print("    day %d  median width %.5f +/- %.5f bp  (%d seeds)"
                      % (d + 1, statistics.mean(v), se, len(v)))
        print("")
    print("  Seeded-book context for this drift: 0.2798bp at 1 day against "
          "0.3835bp at 7 days (78701e4 header).")
    print("  Origin unidentified; no committed file computes those numbers, "
          "and reconstruction gave 0.3267")
    print("  and 0.4195 with the ratio wrong too, so they are quoted as "
          "context and NOT as a comparison target.")
    print("")

    print("=" * 100)
    print("Convergence; secondary: separate 1d vs 7d runs, in SE units")
    print("=" * 100)
    for mlab in ("MM-ABSENT", "MM-PRESENT"):
        a1 = aggregate(store[("1d", mlab)])
        a7 = aggregate(store[("7d", mlab)])
        print("  %s" % mlab)
        for name, key, tgt, tol in TARGETS:
            d = a7[key] - a1[key]
            pooled = (a1[key + "_se"] ** 2 + a7[key + "_se"] ** 2) ** 0.5
            print("    %-20s 1d %12.5f   7d %12.5f   diff %+11.5f  "
                  "= %+6.2f pooled SE"
                  % (name, a1[key], a7[key], d,
                     d / pooled if pooled else float("nan")))
        print("")

    print("  No committed default changed. Nothing retuned. No sniffer.")
    return 0


if __name__ == "__main__":
    sys.exit(main(smoke=("--smoke" in sys.argv),
                  snapshots=("--snapshots" in sys.argv)))
