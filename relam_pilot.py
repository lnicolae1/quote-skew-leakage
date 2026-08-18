# relam_pilot.py: calibration re-solve, step 1: feasibility pilot

import argparse
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import clipped_depth_sweep as CDS
import clipped_lam_sweep as CLS
import clipped_mm_check as CMC
import clipped_placement as CP
import clipped_window as CW
import damage_ab as DA
import emergent_book as EB
import lifetime_sweep as LS
import market_maker as MMOD
import residual_fit as RF

from clipped_lam_sweep import PRODUCT
from clipped_placement import make_world, TARGET
from clipped_depth_sweep import (run_measure, SAMPLE_EVERY, REAL_MEDIAN_BP,
                                 REAL_LEVELS, TOL_WIDTH, TOL_LEVELS, TOL_RATE)
from clipped_window import band, verdict, margin_se, Z, REAL_SHAPE, TOL_SHAPE, \
    TARGETS
from clipped_mm_check import MM_ID
from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from damage_ab import K, GAMMA, REF_MID, mean_se
from emergent_book import per_seed_stats, aggregate
from residual_fit import wmean

NAN = float("nan")

LAM_ANCHOR = 1.8040
LIFE_ANCHOR = 720.0
DISP = 0.0055
CLIP = "join"

LIVES = (720.0, 360.0, 180.0, 120.0, 60.0)

T_PILOT = 86400.0
SEEDS = (0, 1, 2)

ADD_RATE_R1 = 6.3235
ADD_RATE_R2 = 7.1913

P_PART_FILL = 5.0
P_PART_NEAR = 15.0
P_EDGE_SE = 2.0

REF_FILL_SHARE_K17763 = 8.63
REF_NEAR_SHARE_K17763 = 31.29

SELFTEST_SEED = 0
SELFTEST_SHORT = 86400.0


def family(life):
    """(lam, p_market) for one lifetime"""
    lam = LAM_ANCHOR * (LIFE_ANCHOR / life)
    p_market = PRODUCT / lam
    return lam, p_market


class PilotObserver:
    """Wraps noise.run_until and informed.run_until"""

    def __init__(self, eng, noise, informed, mm, vf, seed_ids, requote_int):
        self.eng = eng
        self.mm = mm
        self.vf = vf
        self.seed_ids = seed_ids
        self.requote_int = requote_int

        self.last = REF_MID
        self.v_now = NAN
        self.fills = []

        self.n_fill = 0
        self.n_mm_fill = 0
        self.vol_total = 0.0
        self.vol_mm = 0.0
        self.mm_passive_share = []
        self.mm_near_share = []

        self.n_sec = 0
        self.seed_touch = 0

        self._orig_noise = noise.run_until
        self._orig_inf = informed.run_until
        noise.run_until = self._noise_hook
        informed.run_until = self._inf_hook

    def _book(self, recs, t_end):
        """Fill accounting for one flow's records"""
        ti = int(t_end)
        for r in recs:
            fl = getattr(r, "fills", None)
            if not fl:
                continue
            self.n_fill += len(fl)
            for f in fl:
                self.vol_total += f.size
                if f.counterparty_id != MM_ID:
                    continue
                self.n_mm_fill += 1
                self.vol_mm += f.size
                ps = -1.0 if r.side == "buy" else 1.0
                self.fills.append((ti, ps, f.size, self.last,
                                   self.v_now - self.last))
            if self.mm is not None:
                self.mm.on_fills(fl, r.side)

    def _noise_hook(self, engine, t_end):
        self.v_now = self.vf(t_end)
        recs = self._orig_noise(engine, t_end)
        self._book(recs, t_end)
        return recs

    def _inf_hook(self, engine, t_end, value_fn):
        recs = self._orig_inf(engine, t_end, value_fn)
        self._book(recs, t_end)

        if self.mm is not None:
            self.mm.requote(engine,
                            int(t_end) if self.requote_int else t_end)

        bb, ba = engine.best_bid(), engine.best_ask()
        if bb is None or ba is None:
            return recs

        self.last = 0.5 * (bb + ba)
        self.n_sec += 1

        qb, qa = engine.bids[bb], engine.asks[ba]
        if (any(o.id in self.seed_ids for o in qb)
                or any(o.id in self.seed_ids for o in qa)):
            self.seed_touch += 1

        if t_end % SAMPLE_EVERY:
            return recs
        self._participation_scan(0.5 * (bb + ba))
        return recs

    def _participation_scan(self, mid):
        """MM share of resting size, all levels and within 1bp"""
        eng = self.eng
        sz_all = sz_mm = 0.0
        near_mm = near_all = 0.0
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        for book, is_bid in ((eng.bids, True), (eng.asks, False)):
            for p, q in book.items():
                for o in q:
                    if o.id in self.seed_ids:
                        continue
                    sz_all += o.size
                    if o.agent_id == MM_ID:
                        sz_mm += o.size
                near = (p >= lo) if is_bid else (p <= hi)
                if near:
                    near_all += sum(o.size for o in q
                                    if o.id not in self.seed_ids)
                    near_mm += sum(o.size for o in q if o.agent_id == MM_ID)
        if sz_all > 0:
            self.mm_passive_share.append(100.0 * sz_mm / sz_all)
        if near_all > 0:
            self.mm_near_share.append(100.0 * near_mm / near_all)


def run_cell(seed, lam, p_market, life, disp=DISP, horizon=T_PILOT,
             mm_kw=None, requote_int=True):
    """One seed of one cell"""
    _p, vf, eng, noise, informed = make_world(seed, horizon, lam, p_market,
                                              life, disp, CLIP)
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=horizon, **mm_kw) if mm_kw is not None else None
    obs = PilotObserver(eng, noise, informed, mm, vf, seed_ids, requote_int)

    t0 = time.time()
    res = run_measure(vf, eng, noise, informed, horizon, SAMPLE_EVERY,
                      warmup=0.0)
    secs = time.time() - t0

    out = per_seed_stats(res)
    out["raw"] = res
    out["secs"] = secs
    out["seed"] = seed
    out["n_fills"] = float(len(obs.fills))
    out["fill_records"] = obs.fills

    out["mm_fill_share"] = (100.0 * obs.n_mm_fill / obs.n_fill
                            if obs.n_fill else NAN)
    out["mm_vol_share"] = (100.0 * obs.vol_mm / obs.vol_total
                           if obs.vol_total else NAN)
    out["mm_passive_share"] = (statistics.mean(obs.mm_passive_share)
                               if obs.mm_passive_share else NAN)
    out["mm_near_share"] = (statistics.mean(obs.mm_near_share)
                            if obs.mm_near_share else NAN)

    out["seed_touch_pct"] = (100.0 * obs.seed_touch / obs.n_sec
                             if obs.n_sec else NAN)

    if mm is None:
        for k in ("Ed_w", "h_med", "sd_q", "pnl"):
            out[k] = NAN
        return out

    out["Ed_w"] = (wmean([(-f[1] * f[4], f[2]) for f in obs.fills])
                   if obs.fills else NAN)
    out["Ed_u"] = (statistics.mean([-f[1] * f[4] for f in obs.fills])
                   if obs.fills else NAN)

    halfs, qs = [], []
    for row in mm.log.private_view():
        qs.append(row.true_inventory_q)
        b, a = row.quoted_bid, row.quoted_ask
        if b is None or a is None:
            continue
        halfs.append(0.5 * (a - b))
    out["h_med"] = statistics.median(halfs) if halfs else NAN
    out["sd_q"] = statistics.stdev(qs) if len(qs) > 1 else NAN
    out["n_quotes"] = float(len(halfs))

    out["pnl"] = mm.mark_to_market(obs.last)
    return out


def run_cell_seeds(lam, p_market, life, seeds=SEEDS, horizon=T_PILOT,
                   mm_kw=None, requote_int=True, echo=False):
    rows = []
    for s in seeds:
        r = run_cell(s, lam, p_market, life, horizon=horizon, mm_kw=mm_kw,
                     requote_int=requote_int)
        rows.append(r)
        if echo:
            print("      seed %d done (%.0fs, %d MM fills)"
                  % (s, r["secs"], int(r["n_fills"])), flush=True)
    return rows


def pilot_mm_kw():
    """The committed maker"""
    return dict(k=K, gamma=GAMMA, quote_size=DEFAULT_QUOTE_SIZE)


EXTRA_KEYS = ("Ed_w", "Ed_u", "h_med", "sd_q", "pnl", "n_fills",
              "mm_fill_share", "mm_vol_share", "mm_passive_share",
              "mm_near_share", "seed_touch_pct")


def summarise(rows):
    """Mean and SE across seeds"""
    out = aggregate(rows)
    for k in EXTRA_KEYS:
        m, se, n = mean_se([r[k] for r in rows])
        out[k], out[k + "_se"], out[k + "_n"] = m, se, n
    gap = [r["h_med"] - r["Ed_w"] for r in rows]
    m, se, n = mean_se(gap)
    out["edge_gap"], out["edge_gap_se"], out["edge_gap_n"] = m, se, n
    out["Ed_over_h"] = (out["Ed_w"] / out["h_med"]
                        if out["h_med"] not in (0.0, NAN) else NAN)
    return out


def score_cal(agg):
    """The four scored targets through clipped_window's own band()/verdict()"""
    rows = []
    for name, key, tgt, tol in TARGETS:
        pt, se = agg[key], agg[key + "_se"]
        v = verdict(pt, se, tgt, tol)
        lo_b, hi_b = band(tgt, tol)
        rows.append((name, pt, se, lo_b, hi_b, v,
                     margin_se(pt, se, tgt, tol)))
    return rows


def decide(agg, cal_rows):
    """P-cal, P-edge, P-part"""
    p_cal = all(r[5] == "MET" for r in cal_rows)
    gap, gse = agg["edge_gap"], agg["edge_gap_se"]
    p_edge = (gap > 0.0) and (gse > 0.0) and (gap > P_EDGE_SE * gse)
    p_part = (agg["mm_fill_share"] >= P_PART_FILL
              and agg["mm_near_share"] >= P_PART_NEAR)
    return p_cal, p_edge, p_part


def _fail(tag, msg):
    raise AssertionError("%s FAILED: %s" % (tag, msg))


def t_id():
    """T-ID: every imported name is the same object as its source's"""
    pairs = [
        ("PRODUCT", PRODUCT, CLS.PRODUCT),
        ("make_world", make_world, CP.make_world),
        ("TARGET", TARGET, CP.TARGET),
        ("run_measure", run_measure, CDS.run_measure),
        ("SAMPLE_EVERY", SAMPLE_EVERY, CDS.SAMPLE_EVERY),
        ("REAL_MEDIAN_BP", REAL_MEDIAN_BP, CDS.REAL_MEDIAN_BP),
        ("REAL_LEVELS", REAL_LEVELS, CDS.REAL_LEVELS),
        ("TOL_WIDTH", TOL_WIDTH, CDS.TOL_WIDTH),
        ("TOL_LEVELS", TOL_LEVELS, CDS.TOL_LEVELS),
        ("TOL_RATE", TOL_RATE, CDS.TOL_RATE),
        ("band", band, CW.band),
        ("verdict", verdict, CW.verdict),
        ("margin_se", margin_se, CW.margin_se),
        ("Z", Z, CW.Z),
        ("REAL_SHAPE", REAL_SHAPE, CW.REAL_SHAPE),
        ("TOL_SHAPE", TOL_SHAPE, CW.TOL_SHAPE),
        ("TARGETS", TARGETS, CW.TARGETS),
        ("MM_ID", MM_ID, CMC.MM_ID),
        ("MarketMaker", MarketMaker, MMOD.MarketMaker),
        ("DEFAULT_QUOTE_SIZE", DEFAULT_QUOTE_SIZE, MMOD.DEFAULT_QUOTE_SIZE),
        ("K", K, DA.K),
        ("GAMMA", GAMMA, DA.GAMMA),
        ("REF_MID", REF_MID, DA.REF_MID),
        ("mean_se", mean_se, DA.mean_se),
        ("per_seed_stats", per_seed_stats, EB.per_seed_stats),
        ("aggregate", aggregate, EB.aggregate),
        ("wmean", wmean, RF.wmean),
    ]
    for name, here, there in pairs:
        if here is not there:
            _fail("T-ID", "%s is not the same object as its source" % name)
    print("  T-ID    PASS; %d imported names are is-identical to their "
          "source" % len(pairs))


def t_family():
    """T-family: the family reduces to the committed anchor at life=720"""
    lam, pm = family(LIFE_ANCHOR)
    if abs(lam - 1.8040) > 1e-12:
        _fail("T-FAMILY", "lam at the anchor is %.17g, not 1.8040" % lam)
    if abs(pm - 0.0100) > 1e-12:
        _fail("T-FAMILY", "p_market at the anchor is %.17g, not 0.0100" % pm)
    for life in LIVES:
        lam, pm = family(life)
        if abs(lam * pm - PRODUCT) > 1e-12:
            _fail("T-FAMILY", "lam*p_market at life=%.0f is %.17g, not %.17g"
                  % (life, lam * pm, PRODUCT))
        if abs(lam * life - LAM_ANCHOR * LIFE_ANCHOR) > 1e-9:
            _fail("T-FAMILY", "lam*life at life=%.0f is %.17g, not %.17g"
                  % (life, lam * life, LAM_ANCHOR * LIFE_ANCHOR))
    print("  T-family PASS; anchor gives lam=1.8040 p_market=0.0100; both "
          "constraints hold at all %d lifetimes" % len(LIVES))


def t_verdict():
    """T-VERDICT: band()/verdict() on hand-computed MET, MISSED, UNRESOLVED"""
    lo_b, hi_b = band(100.0, 0.20)
    if (lo_b, hi_b) != (80.0, 120.0):
        _fail("T-VERDICT", "band(100.0, 0.20) = %r, expected (80.0, 120.0)"
              % ((lo_b, hi_b),))
    cases = [(100.0, 1.0, "MET"), (200.0, 1.0, "MISSED"),
             (118.0, 5.0, "UNRESOLVED")]
    for pt, se, want in cases:
        got = verdict(pt, se, 100.0, 0.20)
        if got != want:
            _fail("T-VERDICT", "verdict(%.1f, %.1f) = %s, expected %s"
                  % (pt, se, got, want))
    if abs((118.0 + Z * 5.0) - 127.8) > 1e-9:
        _fail("T-VERDICT", "Z is not 1.96: upper is %.6f" % (118.0 + Z * 5.0))
    print("  T-VERDICT PASS; band [80.0, 120.0]; MET / MISSED / UNRESOLVED "
          "all reproduce")


def t_obs():
    """T-OBS: the observer is non-invasive"""
    lam, pm = family(LIFE_ANCHOR)
    ref = CDS.run(SELFTEST_SEED, lam, pm, LIFE_ANCHOR, DISP)
    mine = run_cell(SELFTEST_SEED, lam, pm, LIFE_ANCHOR,
                    horizon=T_PILOT, mm_kw=None)["raw"]
    if set(ref.keys()) != set(mine.keys()):
        _fail("T-OBS", "key sets differ: %r vs %r"
              % (sorted(ref.keys()), sorted(mine.keys())))
    for k in sorted(ref.keys()):
        if ref[k] != mine[k]:
            _fail("T-OBS", "key %r differs: %r vs %r" % (k, ref[k], mine[k]))
    n = len(ref["spreads_bp"])
    if n < 1000:
        _fail("T-OBS", "only %d spread samples -- too few to be a check" % n)
    print("  T-OBS   PASS; full dict equality against clipped_depth_sweep."
          "run, including all %d spreads_bp" % n)


def t_fill():
    """T-fill: the mirrored fill record reproduces residual_fit.run_arm"""
    ref_recs, _mid, _extras = RF.run_arm(SELFTEST_SEED, LIFE_ANCHOR,
                                         SELFTEST_SHORT)
    mine = run_cell(SELFTEST_SEED, LS.LAM, LS.P_MARKET, LIFE_ANCHOR,
                    disp=LS.DISP, horizon=SELFTEST_SHORT,
                    mm_kw=pilot_mm_kw(), requote_int=True)
    got = mine["fill_records"]
    if not ref_recs:
        _fail("T-FILL", "the REFERENCE produced zero fills -- the check would "
                        "be vacuous")
    if len(got) != len(ref_recs):
        _fail("T-FILL", "fill counts differ: %d mine vs %d run_arm"
              % (len(got), len(ref_recs)))
    for i, (a, b) in enumerate(zip(got, ref_recs)):
        want = (b[0], b[1], b[2], b[3], b[6])
        if a != want:
            _fail("T-FILL", "record %d differs: %r vs %r" % (i, a, want))
    ed_mine = wmean([(-f[1] * f[4], f[2]) for f in got])
    ed_ref = wmean([(-b[1] * b[6], b[2]) for b in ref_recs])
    if ed_mine != ed_ref:
        _fail("T-FILL", "E[d] differs: %.17g vs %.17g" % (ed_mine, ed_ref))
    print("  T-fill  PASS; %d fill records identical to residual_fit.run_arm;"
          " E[d] %.6f matches exactly" % (len(got), ed_mine))
    return len(got)


def t_part():
    """T-part: the mirrored participation counters reproduce"""
    ref = CMC.run(SELFTEST_SEED, K)
    mine = run_cell(SELFTEST_SEED, CDS.LAM, CMC.P_MARKET, CMC.LIFE,
                    disp=CMC.DISP, horizon=SELFTEST_SHORT,
                    mm_kw=dict(k=K), requote_int=False)
    checks = [("mm_fill_share", mine["mm_fill_share"], ref["mm_fill_share"]),
              ("mm_vol_share", mine["mm_vol_share"], ref["mm_vol_share"]),
              ("mm_passive_share", mine["mm_passive_share"],
               ref["mm_passive_share"]),
              ("mm_near_share", mine["mm_near_share"], ref["mm_near_share"]),
              ("agg_per_s", mine["agg_per_s"], ref["agg_per_s"])]
    for name, a, b in checks:
        if a != b:
            _fail("T-PART", "%s differs: %.17g vs %.17g" % (name, a, b))
    if mine["mm_fill_share"] <= 0.0:
        _fail("T-PART", "MM fill share is zero -- the check would be vacuous")
    print("  T-part  PASS; %d participation/rate figures identical to "
          "clipped_mm_check.run (fill share %.4f pct, near share %.4f pct)"
          % (len(checks), mine["mm_fill_share"], mine["mm_near_share"]))


def t_det(n_fills_expected):
    """T-det: determinism, at a horizon that actually produces fills"""
    lam, pm = family(LIFE_ANCHOR)
    a = run_cell(SELFTEST_SEED, lam, pm, LIFE_ANCHOR,
                 horizon=SELFTEST_SHORT, mm_kw=pilot_mm_kw())
    b = run_cell(SELFTEST_SEED, lam, pm, LIFE_ANCHOR,
                 horizon=SELFTEST_SHORT, mm_kw=pilot_mm_kw())
    if a["n_fills"] <= 0.0:
        _fail("T-DET", "ZERO MM fills at horizon %.0f -- the determinism check "
                       "would compare two empty sets and pass vacuously"
              % SELFTEST_SHORT)
    keys = ["median_bp", "mean_bp", "shape", "levels_per_side", "agg_per_s",
            "n_fills", "Ed_w", "Ed_u", "h_med", "sd_q", "pnl",
            "mm_fill_share", "mm_vol_share", "mm_passive_share",
            "mm_near_share", "seed_touch_pct"]
    for k in keys:
        if a[k] != b[k]:
            _fail("T-DET", "key %r differs across identical runs: %.17g vs "
                           "%.17g" % (k, a[k], b[k]))
    if a["fill_records"] != b["fill_records"]:
        _fail("T-DET", "fill records differ across identical runs")
    print("  T-det   PASS; %d statistics reproduce across two identical runs,"
          " on %d MM fills" % (len(keys), int(a["n_fills"])))
    return a


def t_nonempty(a):
    """T-nonempty: the fill set, the quote log and the scan samples are all"""
    if a["n_fills"] <= 0.0:
        _fail("T-NONEMPTY", "zero MM fills")
    if a["n_quotes"] <= 0.0:
        _fail("T-NONEMPTY", "zero quotes in the maker's log")
    if not (a["mm_near_share"] == a["mm_near_share"]):
        _fail("T-NONEMPTY", "mm_near_share is NaN -- no near-touch scan sample "
                            "ever had non-seed size")
    if not (a["Ed_w"] == a["Ed_w"]):
        _fail("T-NONEMPTY", "E[d] is NaN")
    print("  T-nonempty PASS; %d fills, %d quotes, E[d] and participation "
          "both finite" % (int(a["n_fills"]), int(a["n_quotes"])))


def selftest():
    print("=" * 110)
    print("relam_pilot.py self-tests")
    print("=" * 110)
    t0 = time.time()
    t_id()
    t_family()
    t_verdict()
    print("  (the three run-based gates each build a full 1-day world; "
          "expect ~60s)", flush=True)
    t_obs()
    nf = t_fill()
    t_part()
    a = t_det(nf)
    t_nonempty(a)
    print("")
    print("  all self-tests PASS  (%.0fs)" % (time.time() - t0))
    print("")
    print("  Not covered by any gate, and said plainly:")
    print("    - seed_touch_pct is transcribed from mm_on_off.py:83-89 and has")
    print("      no committed reference to check against. Diagnostic only.")
    print("    - the family arithmetic is checked against its own anchor, not")
    print("      against an external source; product is imported, not restated.")
    print("    - nothing here tests whether the levels anchor holds. That is")
    print("      the measurement the pilot exists to make.")


def header():
    print("=" * 118)
    print("The calibration re-solve; pilot. Can the book hold density while "
          "shortening memory?")
    print("=" * 118)
    print("Feasibility pilot: %.0fs x %d seeds per cell. Not the headline; "
          "survivors go to a 7-day arm next." % (T_PILOT, len(SEEDS)))
    print("MM present (k=%.5f, gamma=%.6g, quote_size=%.3f). JOIN clipping. "
          "Seed book in. disp=%.4f held." % (K, GAMMA, DEFAULT_QUOTE_SIZE,
                                             DISP))
    print("")
    print("The family: lam*p_market = %.5f (rate) and lam*life = %.1f "
          "(levels), so life is the only free parameter."
          % (PRODUCT, LAM_ANCHOR * LIFE_ANCHOR))
    hdr = ("  %-10s %10s %12s %14s %14s"
           % ("life (s)", "lam /s", "p_market", "pct of 7.1913", "pct of 6.3235"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for life in LIVES:
        lam, pm = family(life)
        print("  %-10.0f %10.4f %12.6f %13.0f%% %13.0f%%"
              % (life, lam, pm, 100.0 * lam / ADD_RATE_R2,
                 100.0 * lam / ADD_RATE_R1))
    print("")
    print("  The last two columns are lam against the measured real addition "
          "rate, Route 2 and Route 1")
    print("  (lam_ceiling_arithmetic.py:110-111). Any cell over 100% QUOTES an "
          "Arrival rate the real venue")
    print("  is not measured to sustain. Those cells are still run; a "
          "mechanism that fails even there is")
    print("  dead without the ceiling argument; but survival at such a cell "
          "must be read with that attached.")
    print("")
    print("The levels anchor is a model, NOT A measurement "
          "(lam_ceiling_arithmetic.py:314, caveat at :318-328).")
    print("  Measured levels/side are scored per cell like any other target. "
          "Systematically off-target levels")
    print("  mean the anchor is wrong, which is a finding about the model and "
          "not a failure of the cell.")
    print("")
    print("The four bands were fitted MM-ABSENT and this arm is MM-present "
          "(emergent_book.py:46-49).")
    print("  And the committed point itself fails them: at 7d MM-present, "
          "width 0.3976bp against a 0.34476")
    print("  ceiling, and lifetime_sweep_results.txt:62 records 'Levels "
          "passing all three: none'. So a cell")
    print("  That merely matches the committed point has PASSED nothing.")
    print("")
    print("  The life=720 cell is the family's anchor, not a control: "
          "Product = 1.8040 * 0.0100 passes through")
    print("  Phase 3's committed-rate point, NOT the 847a7fa working point "
          "(p_market 0.0126, +26%). It will not")
    print("  reproduce 847a7fa's numbers and is not meant to.")
    print("")


def pass_conditions():
    print("=" * 118)
    print("Pre-registered PASS conditions; fixed before any cell ran")
    print("=" * 118)
    print("  P-cal   all four scored targets MET, using clipped_window's own "
          "band()/verdict(). MET requires the")
    print("          whole 95% interval inside the band; straddling an edge is "
          "UNRESOLVED and is NOT a hit.")
    print("          At %d seeds this is a hard bar and UNRESOLVED is a likely "
          "outcome. It is not weakened." % len(SEEDS))
    print("  P-edge  E[d] < h with the gap clearing %.1f SE, paired per seed "
          "(the seeds share the V path)." % P_EDGE_SE)
    print("  P-part  mm_fill_share >= %.1f%% AND mm_near_share >= %.1f%%."
          % (P_PART_FILL, P_PART_NEAR))
    print("          These are a judgement call, at roughly half the committed "
          "point's own k=0.17763 values")
    print("          (%.2f%% of fills, %.2f%% of near-touch size). Half, because "
          "a re-solve moving lam up to 12x"
          % (REF_FILL_SHARE_K17763, REF_NEAR_SHARE_K17763))
    print("          should not have to leave participation untouched; and not "
          "lower, because a maker under")
    print("          5% of fills and 15% of near-touch size is not the venue's "
          "liquidity provider in any sense")
    print("          that makes the leakage question meaningful. A bar a ghost "
          "could clear would let a cell pass")
    print("          by making the maker irrelevant, which is exactly what this "
          "condition exists to catch.")
    print("")
    print("  A cell survives only if all three hold. Survivors are not ranked "
          "And no winner is picked.")
    print("  If no cell survives, that is a bounded negative result: the "
          "(lam, mean_lifetime) lever is exhausted.")
    print("")
    print("  Burden asymmetry, flagged not fixed: P-cal and P-edge are interval "
          "tests; P-part is a point-estimate")
    print("  test, because its thresholds were pre-registered as bare "
          "percentages. The 95% interval is printed")
    print("  beside each participation figure so the reader can see whether the "
          "point estimate is resolved.")
    print("")


def report_cell(life, lam, pm, agg, cal_rows, p_cal, p_edge, p_part):
    print("-" * 118)
    print("Cell life=%.0fs  lam=%.4f  p_market=%.6f  (lam is %.0f%% of the "
          "measured Route-2 addition rate)"
          % (life, lam, pm, 100.0 * lam / ADD_RATE_R2))
    print("-" * 118)

    print("  A. The four scored targets  (95% interval vs band; MET requires "
          "the whole interval inside)")
    for name, pt, se, lo_b, hi_b, v, m in cal_rows:
        print("     %-20s %11.5f +/- %-9.5f  95%% [%10.5f, %10.5f]  band "
              "[%10.5f, %10.5f]  %-10s %7.2f SE"
              % (name, pt, se, pt - Z * se, pt + Z * se, lo_b, hi_b, v, m))
    n_met = sum(1 for r in cal_rows if r[5] == "MET")
    n_missed = sum(1 for r in cal_rows if r[5] == "MISSED")
    n_unres = sum(1 for r in cal_rows if r[5] == "UNRESOLVED")
    print("     -> %d/4 MET, %d MISSED, %d UNRESOLVED"
          % (n_met, n_missed, n_unres))
    print("     measured levels/side %.1f +/- %.1f against the anchor's "
          "nominal %.1f; the anchor is a model"
          % (agg["levels_per_side"], agg["levels_per_side_se"], REAL_LEVELS))

    print("")
    print("  B. The edge  (E[d] size-weighted over all fills; h from the "
          "maker's own quote log)")
    print("     E[d] size-weighted  %10.4f +/- %-9.4f  $/BTC"
          % (agg["Ed_w"], agg["Ed_w_se"]))
    print("     E[d] unweighted     %10.4f +/- %-9.4f  $/BTC"
          % (agg["Ed_u"], agg["Ed_u_se"]))
    print("     h (median half-spread) %7.4f +/- %-9.4f  $"
          % (agg["h_med"], agg["h_med_se"]))
    print("     E[d] / h            %10.4f" % agg["Ed_over_h"])
    print("     paired gap h - E[d] %10.4f +/- %-9.4f   needs > %.4f"
          % (agg["edge_gap"], agg["edge_gap_se"],
             P_EDGE_SE * agg["edge_gap_se"]))
    print("     MM fills per seed   %10.1f +/- %-9.1f"
          % (agg["n_fills"], agg["n_fills_se"]))

    print("")
    print("  C. Participation  (point estimate is what P-part scores; the "
          "interval is printed, not scored)")
    for label, key, thr in (("fill-count share (%)", "mm_fill_share",
                             P_PART_FILL),
                            ("near-touch size share (%)", "mm_near_share",
                             P_PART_NEAR),
                            ("volume share (%)", "mm_vol_share", None),
                            ("passive share, all (%)", "mm_passive_share",
                             None)):
        pt, se = agg[key], agg[key + "_se"]
        thr_s = ("threshold %.1f" % thr) if thr is not None else "not scored"
        print("     %-26s %9.4f +/- %-8.4f  95%% [%9.4f, %9.4f]  %s"
              % (label, pt, se, pt - Z * se, pt + Z * se, thr_s))
    print("     %-26s %9.6f +/- %-8.6f  BTC"
          % ("sd(q) over the run", agg["sd_q"], agg["sd_q_se"]))

    print("")
    print("  D. PnL; reported, not scored")
    print("     MM PnL              %10.2f +/- %-9.2f  $"
          % (agg["pnl"], agg["pnl_se"]))
    print("     this is not A PASS condition. The committed 7-day life=60 "
          "figure is +14.91 +/- 193.95, i.e.")
    print("     0.08 SE from zero and indistinguishable from both zero and "
          "-525.18. At 1 day and %d seeds" % len(SEEDS))
    print("     this pilot has strictly less power than that, so no PnL "
          "threshold could resolve.")

    print("")
    print("  E. Seed-book influence; diagnostic, gates nothing")
    print("     seed order at the touch %7.4f +/- %-8.4f  %% of two-sided "
          "seconds  (committed reference 7.4692)"
          % (agg["seed_touch_pct"], agg["seed_touch_pct_se"]))

    print("")
    print("  Verdict for this cell")
    print("     P-cal   %-5s  (%d/4 MET)" % (str(p_cal), n_met))
    print("     P-edge  %-5s  (gap %.4f +/- %.4f, needs > %.4f and > 0)"
          % (str(p_edge), agg["edge_gap"], agg["edge_gap_se"],
             P_EDGE_SE * agg["edge_gap_se"]))
    print("     P-part  %-5s  (fills %.4f vs %.1f, near %.4f vs %.1f)"
          % (str(p_part), agg["mm_fill_share"], P_PART_FILL,
             agg["mm_near_share"], P_PART_NEAR))
    print("     survives: %s" % str(p_cal and p_edge and p_part))
    print("")


def main(argv=None):
    ap = argparse.ArgumentParser(description="The calibration re-solve pilot.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the self-tests ONLY and exit; no cells")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    selftest()
    print("")
    header()
    pass_conditions()

    t0 = time.time()
    survivors = []
    for life in LIVES:
        lam, pm = family(life)
        print("  running life=%.0f (lam=%.4f p_market=%.6f) ..."
              % (life, lam, pm), flush=True)
        rows = run_cell_seeds(lam, pm, life, mm_kw=pilot_mm_kw(), echo=True)
        agg = summarise(rows)
        cal_rows = score_cal(agg)
        p_cal, p_edge, p_part = decide(agg, cal_rows)
        report_cell(life, lam, pm, agg, cal_rows, p_cal, p_edge, p_part)
        if p_cal and p_edge and p_part:
            survivors.append(life)

    print("=" * 118)
    print("Result")
    print("=" * 118)
    if not survivors:
        print("  No cell survives. Every cell fails at least one of P-cal, "
              "P-edge, P-part.")
        print("  That is A BOUNDED negative result: holding density while "
              "shortening the book's memory does not")
        print("  produce a maker that clears its own edge and stays a "
              "participant, anywhere on this family.")
        print("  The (lam, mean_lifetime) lever is exhausted at pilot power. "
              "No estimate is quoted and no cell")
        print("  Is carried forward.")
    else:
        print("  Surviving cells: %s"
              % ", ".join("life=%.0f" % v for v in survivors))
        print("  these are not ranked and no winner is picked. They go to a "
              "7-day arm next session, which is")
        print("  where any headline comes from. A 1-day 3-seed survival is a "
              "feasibility signal, not a result.")
    print("")
    print("  total wall clock %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
