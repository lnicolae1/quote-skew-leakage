# stacking_source.py: two follow-ups to resting_size.py; measurement only (no fix, retune or sweep)
# - A: near-touch size p95 1.69x vs trade_sizes.bin: broken draw (a) or survivorship (b)?
#   sizes at 4 stages: every limit drawn -> drawn size of those rested -> rested size -> near-touch scan
# - B: share of near-touch stacking due to JOIN clipping
#   - per-order clip flag is not retained (NoiseOrder has no clip field; flow keeps only n_clipped)
#   - recovered by pairing the n_clipped delta across noise.limit_price() with the order_id of the
#     next add_limit_order(); adjacent in _submit, and run_measure drains noise before informed
#   - instance wrappers only; pairing tested in self_test on forced-on/forced-off books
# - scope: MM-absent emergent book, W=3600s, 1 day, 15 seeds; lam=1.804, p_market=0.0126,
#   mean_lifetime=720, disp=0.0055, JOIN; driven by clipped_depth_sweep.run_measure
# results:
# - A: stages 1-3 identical (n = 2,406,478): draw_size() is a faithful bootstrap, (a) ruled out
#   - departure only at 3->4, in the tail: p95 1.69x, mean 1.13x, p25/median 1.00-1.01
#   - under JOIN a crossing limit rests at its own near touch: noise limits never trade aggressively
# - B: orders/level unclipped median 1, mean 1.00216, p95 1; clipped median 8, mean 21.3, p95 87
#   - clipped share 12.42% at submission -> 91.64% of near-touch resting observations (7.4x)
#   - caveats: clipped orders are a selected (aggressively priced) population; stacking from clipping is
#     near-definitional (the magnitude is the finding); submission counts include warm-up, scans do not
# - C: size selection sits in the unclipped 8.36%: near-touch p95/mean vs source
#   clipped 1.24x/1.07x, unclipped 6.04x/1.85x (small orders consumed whole; large ones keep a remainder)

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_depth_sweep import run_measure
from emergent_book import (make_world_emergent, WARMUP, SAMPLE_EVERY, T_DAY,
                           SEEDS, LAM, P_MARKET, LIFE, DISP, CLIP)
from resting_size import describe, safe_ratio, BAND, load_source_sizes


# --------------------------------------------------------------------------- #
# observer: submission sizes (A), recovered clip flags (B)
# --------------------------------------------------------------------------- #
class Tracker:
    """Instance-level hooks:
    - noise.limit_price: flags whether this draw clipped (n_clipped delta)
    - eng.add_limit_order: attaches the pending flag to the returned order_id
    - informed.run_until: runs the scan after both flows act
    """

    def __init__(self, eng, noise, informed, warmup):
        self.eng = eng
        self.noise = noise
        self.warmup = warmup
        # hardcoded prefix could silently empty every population
        self.prefix = getattr(noise, "agent_id_prefix", "noise")

        # A: submission-stage populations
        self.sub_all = []          # every noise limit order's drawn size
        self.sub_rested = []       # drawn size, rested orders
        self.sub_resting = []      # rested size

        # B: clip recovery (side kept for self-test and per-side split)
        self.pending_clip = None
        self.clip_by_id = {}       # order_id -> (clipped, side), if it rested
        self.n_limit = 0
        self.n_clipped_sub = 0
        self.n_limit_side = {"buy": 0, "sell": 0}
        self.n_clip_side = {"buy": 0, "sell": 0}

        # scan populations, keyed by (clipped, side)
        self.near_sizes = {}       # remaining sizes
        self.near_opl = {}         # counts per level
        for c in (True, False):
            for sd in ("buy", "sell"):
                self.near_sizes[(c, sd)] = []
                self.near_opl[(c, sd)] = []
        self.level_total_opl = []
        self.n_scans = 0
        self.n_near_unknown = 0    # resting orders with no recovered flag
        # recorded side vs scanned book; nonzero = misattached tag
        self.n_side_mismatch = 0

        self._orig_lp = noise.limit_price
        self._orig_add = eng.add_limit_order
        self._orig_inf = informed.run_until
        noise.limit_price = self._lp_hook
        eng.add_limit_order = self._add_hook
        informed.run_until = self._inf_hook

    # ---------------- B: recover the clipped flag ---------------- #
    def _lp_hook(self, engine, side):
        before = getattr(self.noise, "n_clipped", 0)
        price = self._orig_lp(engine, side)
        self.pending_clip = (getattr(self.noise, "n_clipped", 0) > before)
        return price

    def _add_hook(self, side, price, size, agent_id):
        res = self._orig_add(side, price, size, agent_id)
        # informed posts skip noise.limit_price; guard stops them eating a stale flag
        if (self.pending_clip is not None
                and str(agent_id).startswith(self.prefix)):
            self.n_limit += 1
            self.n_limit_side[side] = self.n_limit_side.get(side, 0) + 1
            if self.pending_clip:
                self.n_clipped_sub += 1
                self.n_clip_side[side] = self.n_clip_side.get(side, 0) + 1
            self.sub_all.append(size)
            if res.resting_size > 0:
                self.clip_by_id[res.order_id] = (self.pending_clip, side)
                self.sub_rested.append(size)
                self.sub_resting.append(res.resting_size)
            self.pending_clip = None
        return res

    # ---------------- the periodic near-touch scan ---------------- #
    def scan(self, engine):
        bb, ba = engine.best_bid(), engine.best_ask()
        if bb is None or ba is None:
            return False
        mid = 0.5 * (bb + ba)
        lo, hi = mid * (1 - BAND), mid * (1 + BAND)
        self.n_scans += 1
        for book, is_bid in ((engine.bids, True), (engine.asks, False)):
            book_side = "buy" if is_bid else "sell"
            for p, q in book.items():
                if not ((p >= lo) if is_bid else (p <= hi)):
                    continue
                live = list(q)
                if not live:
                    continue
                self.level_total_opl.append(len(live))
                n_by = {True: 0, False: 0}
                for o in live:
                    rec = self.clip_by_id.get(o.id)
                    if rec is None:
                        # informed posts etc.
                        self.n_near_unknown += 1
                        continue
                    c, rec_side = rec
                    if rec_side != book_side:
                        self.n_side_mismatch += 1
                    self.near_sizes[(c, book_side)].append(o.size)
                    n_by[c] += 1
                for c in (True, False):
                    if n_by[c]:
                        self.near_opl[(c, book_side)].append(n_by[c])
        return True

    def _inf_hook(self, engine, t_end, value_fn):
        recs = self._orig_inf(engine, t_end, value_fn)
        if t_end > self.warmup and not (t_end % SAMPLE_EVERY):
            self.scan(engine)
        return recs


def run_seed(seed):
    t_total = WARMUP + T_DAY
    _p, vf, eng, noise, informed = make_world_emergent(
        seed, t_total, LAM, P_MARKET, LIFE, DISP, CLIP, seed_book=False)
    trk = Tracker(eng, noise, informed, WARMUP)
    run_measure(vf, eng, noise, informed, T_DAY, SAMPLE_EVERY, warmup=WARMUP)
    return trk


def _blank_tracker(eng):
    """Tracker with scan state only, no hooks; for the self-test."""
    t = Tracker.__new__(Tracker)
    t.eng = eng
    t.prefix = "noise"
    t.clip_by_id = {}
    # same (clipped, side) keys as __init__, else scan() KeyErrors on a tagged order
    t.near_sizes = {}
    t.near_opl = {}
    for c in (True, False):
        for sd in ("buy", "sell"):
            t.near_sizes[(c, sd)] = []
            t.near_opl[(c, sd)] = []
    t.level_total_opl = []
    t.n_scans = 0
    t.n_near_unknown = 0
    t.n_side_mismatch = 0
    return t


# --------------------------------------------------------------------------- #
# self-test: clip-flag pairing on books with clipping forced on / off
# --------------------------------------------------------------------------- #
def self_test():
    from matching_engine import MatchingEngine
    from clipped_placement import ClippedNoiseTraderFlow
    from informed_traders import InformedTraderFlow
    import sim_run

    print("=" * 100)
    print("Self-test: clip-flag pairing, clipping forced on and off")
    print("=" * 100)
    ok = True

    def build(best_bid, best_ask):
        eng = MatchingEngine()
        eng.add_limit_order("buy", best_bid, 5.0, "wall")
        eng.add_limit_order("sell", best_ask, 5.0, "wall")
        noise = ClippedNoiseTraderFlow(
            clip_mode="join", lam=LAM, p_market=0.0, mean_lifetime=LIFE,
            disp=DISP, value_fn=lambda t: sim_run.REFERENCE_PRICE,
            reference_price=sim_run.REFERENCE_PRICE, seed=7)
        inf = InformedTraderFlow(seed=7)
        trk = Tracker(eng, noise, inf, 0.0)
        return eng, noise, trk

    # 0. agent-id guard matches a real noise id, rejects others
    _e, _n, _t = build(61999.00, 62001.00)
    real_id = "%s_%d" % (_n.agent_id_prefix, 1)
    print("  0. agent-id guard          : prefix=%r, a real id %r matches -> "
          "%s ; 'informed' matches -> %s"
          % (_t.prefix, real_id, real_id.startswith(_t.prefix),
             "informed".startswith(_t.prefix)))
    ok &= (real_id.startswith(_t.prefix)
           and not "informed".startswith(_t.prefix))

    # 1. forced on: ask wall 61,001; buys drawn around 62,000 (sd ~ $341) all cross
    eng, noise, trk = build(61000.00, 61001.00)
    for _ in range(60):
        noise.limit_price(eng, "buy")
    print("  1. clipping FORCED ON      : n_clipped after 60 buy draws = %d "
          "(expect ~60)" % noise.n_clipped)
    ok &= (noise.n_clipped >= 55)

    # 2. forced off: ask wall 70,000 above any buy draw
    eng2, noise2, trk2 = build(61999.00, 70000.00)
    for _ in range(60):
        noise2.limit_price(eng2, "buy")
    print("  2. clipping FORCED OFF     : n_clipped after 60 buy draws = %d "
          "(expect 0)" % noise2.n_clipped)
    ok &= (noise2.n_clipped == 0)

    # 3. pairing via the real _submit path, side-conditional:
    #    ask far below the draws -> every buy clipped, every sell unclipped
    #    (both sides clipping needs best_bid > best_ask, which the engine forbids)
    #    count match alone passes with flags on the wrong orders; both checked
    eng3, noise3, trk3 = build(61000.00, 61001.00)
    for _ in range(40):
        noise3.step(eng3)
    buys = [(c, sd) for c, sd in trk3.clip_by_id.values() if sd == "buy"]
    sells = [(c, sd) for c, sd in trk3.clip_by_id.values() if sd == "sell"]
    buy_clip = sum(1 for c, _ in buys if c)
    sell_clip = sum(1 for c, _ in sells if c)
    print("  3. pairing, forced on      : buys n=%d clipped=%d (expect all) ; "
          "sells n=%d clipped=%d (expect none)"
          % (len(buys), buy_clip, len(sells), sell_clip))
    print("     flow counter n_clipped=%d, tracker n_clipped_sub=%d"
          % (noise3.n_clipped, trk3.n_clipped_sub))
    good = (trk3.n_clipped_sub == noise3.n_clipped
            and len(buys) > 0 and buy_clip == len(buys)
            and len(sells) > 0 and sell_clip == 0)
    print("     every buy clipped, every sell unclipped, "
          "tracker == flow counter -> %s" % ("ok" if good else "FAIL"))
    ok &= good

    eng4, noise4, trk4 = build(61999.00, 70000.00)
    for _ in range(40):
        noise4.step(eng4)
    t4 = sum(1 for c, _sd in trk4.clip_by_id.values() if c)
    f4 = sum(1 for c, _sd in trk4.clip_by_id.values() if not c)
    print("  4. pairing, forced off     : recovered clipped=%d unclipped=%d, "
          "flow counter n_clipped=%d" % (t4, f4, noise4.n_clipped))
    good = (t4 == 0 and f4 > 0 and trk4.n_clipped_sub == 0)
    print("     no order flagged clipped -> %s"
          % ("ok" if good else "FAIL"))
    ok &= good

    # 5. an informed post must not consume a pending noise flag
    eng5, noise5, trk5 = build(61000.00, 61001.00)
    noise5.limit_price(eng5, "buy")            # sets pending_clip
    pend_before = trk5.pending_clip
    eng5.add_limit_order("sell", 62500.00, 0.001, "informed")
    good = (trk5.pending_clip == pend_before and not trk5.clip_by_id)
    print("  5. informed post isolation : pending flag before=%r after=%r "
          "(must be unchanged), tags written=%d -> %s"
          % (pend_before, trk5.pending_clip, len(trk5.clip_by_id),
             "ok" if good else "FAIL"))
    ok &= good

    # 6. submission collectors fire; rested <= drawn
    good = (trk3.n_limit == len(trk3.sub_all) and trk3.n_limit > 0
            and len(trk3.sub_rested) <= trk3.n_limit)
    print("  6. submission collectors   : forced-on run drew %d limit orders, "
          "%d rested, %d sizes recorded -> %s"
          % (trk3.n_limit, len(trk3.sub_rested), len(trk3.sub_all),
             "ok" if good else "FAIL"))
    ok &= good

    # 7. scan reports nothing on an empty book
    empty = MatchingEngine()
    trk_e = _blank_tracker(empty)
    empty_result = trk_e.scan(empty)
    print("  7. empty book              : scan returned %r (expect False), "
          "n_scans=%d (expect 0) -> %s"
          % (empty_result, trk_e.n_scans,
             "ok" if (empty_result is False and trk_e.n_scans == 0)
             else "FAIL"))
    ok &= (empty_result is False and trk_e.n_scans == 0)

    # 7b. scan fires on a populated book; untagged orders counted as unknown
    pop = MatchingEngine()
    pop.add_limit_order("buy", 61999.00, 0.30, "noise_1")
    pop.add_limit_order("sell", 62001.00, 0.20, "informed")
    trk_p = _blank_tracker(pop)
    fired = trk_p.scan(pop)
    print("     populated book          : scan returned %r, n_scans=%d, "
          "untagged counted as unknown=%d (expect 2) -> %s"
          % (fired, trk_p.n_scans, trk_p.n_near_unknown,
             "ok" if (fired is True and trk_p.n_near_unknown == 2) else "FAIL"))
    ok &= (fired is True and trk_p.n_near_unknown == 2)

    # 8. source file is the Phase 3 population
    d = describe(load_source_sizes())
    print("  8. trade_sizes.bin         : n=%d median=%.5f mean=%.5f "
          "(Phase 3: 39184 / 0.00076 / 0.00347) -> %s"
          % (d["n"], d["median"], d["mean"],
             "ok" if d["n"] == 39184 else "FAIL"))
    ok &= (d["n"] == 39184)

    print("\n  self-test: %s" % ("PASS" if ok else "FAIL; stopping"))
    return ok


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
HDR = ("  %-38s %9s %10s %10s %10s %10s %10s"
       % ("", "n", "p25", "median", "p75", "p95", "mean"))


def row(label, d):
    return ("  %-38s %9d %10.5f %10.5f %10.5f %10.5f %10.5f"
            % (label, d["n"], d["p25"], d["median"], d["p75"], d["p95"],
               d["mean"]))


def ratio_row(label, num, den):
    return ("  %-38s %10.2f %10.2f %10.2f %10.2f %10.2f"
            % (label, safe_ratio(num["p25"], den["p25"]),
               safe_ratio(num["median"], den["median"]),
               safe_ratio(num["p75"], den["p75"]),
               safe_ratio(num["p95"], den["p95"]),
               safe_ratio(num["mean"], den["mean"])))


def main():
    print("=" * 100)
    print("Stacking and source: follow-ups to resting_size.py")
    print("=" * 100)
    print("Emergent book (no seed book), W=%.0fs, MM-ABSENT, 1 day, %d seeds, "
          "scan every %ds" % (WARMUP, len(SEEDS), SAMPLE_EVERY))
    print("lam=%.4f p_market=%.4f mean_lifetime=%.0fs disp=%.4f %s clipping "
          "(rule unchanged)"
          % (LAM, P_MARKET, LIFE, DISP, CLIP.upper()))
    print("")
    print("Clip flag recovered by pairing the n_clipped delta across "
          "noise.limit_price() with the")
    print("order_id of the add_limit_order() that immediately follows it; "
          "pairing tested in the self-test.")
    print("")

    if not self_test():
        return 1
    print("")

    t0 = time.time()
    sub_all, sub_rested, sub_resting = [], [], []
    near = {}
    opl = {}
    for c in (True, False):
        for sd in ("buy", "sell"):
            near[(c, sd)] = []
            opl[(c, sd)] = []
    tot_opl = []
    n_limit = n_clip_sub = n_unknown = n_mismatch = 0
    n_lim_side = {"buy": 0, "sell": 0}
    n_clip_side = {"buy": 0, "sell": 0}
    for s in SEEDS:
        trk = run_seed(s)
        sub_all.extend(trk.sub_all)
        sub_rested.extend(trk.sub_rested)
        sub_resting.extend(trk.sub_resting)
        for c in (True, False):
            for sd in ("buy", "sell"):
                near[(c, sd)].extend(trk.near_sizes[(c, sd)])
                opl[(c, sd)].extend(trk.near_opl[(c, sd)])
        tot_opl.extend(trk.level_total_opl)
        n_limit += trk.n_limit
        n_clip_sub += trk.n_clipped_sub
        n_unknown += trk.n_near_unknown
        n_mismatch += trk.n_side_mismatch
        for sd in ("buy", "sell"):
            n_lim_side[sd] += trk.n_limit_side[sd]
            n_clip_side[sd] += trk.n_clip_side[sd]
        print("  ran seed %d (%d limit orders, %d clipped, %d near-touch "
              "tagged, %.0fs)"
              % (s, trk.n_limit, trk.n_clipped_sub,
                 sum(len(trk.near_sizes[k]) for k in trk.near_sizes),
                 time.time() - t0), flush=True)
    print("")

    def pool(c):
        """Both sides of one clip category."""
        return near[(c, "buy")] + near[(c, "sell")]

    def pool_opl(c):
        return opl[(c, "buy")] + opl[(c, "sell")]

    d_src = describe(load_source_sizes())
    d_all = describe(sub_all)
    d_rest = describe(sub_rested)
    d_resting = describe(sub_resting)
    d_near = describe(pool(True) + pool(False))

    print("=" * 100)
    print("A. Broken draw, or size-selected near-touch sample?")
    print("=" * 100)
    print(HDR)
    print(row("trade_sizes.bin (39,184 TAKER)", d_src))
    print(row("1. SUBMITTED: every limit drawn", d_all))
    print(row("2. SUBMITTED: those that rested", d_rest))
    print(row("3. the size that actually RESTED", d_resting))
    print(row("4. NEAR-TOUCH resting, periodic scan", d_near))
    print("")
    print("  %-38s %10s %10s %10s %10s %10s"
          % ("ratios vs source", "p25", "median", "p75", "p95", "mean"))
    print(ratio_row("1. submitted / source", d_all, d_src))
    print(ratio_row("2. rested, drawn size / source", d_rest, d_src))
    print(ratio_row("3. rested size / source", d_resting, d_src))
    print(ratio_row("4. near-touch scan / source", d_near, d_src))
    print("")
    print("  stage 1 ~1.00 => faithful draw, excess is survivorship (b); "
          "else draw_size() bug (a)")
    print("  first departure from 1.00: 1->2 rest-vs-execute, 2->3 partial fill on "
          "arrival, 3->4 survival")
    print("")
    same_n = (d_all["n"] == d_rest["n"] == d_resting["n"])
    print("  stages 1/2/3 sample counts: %d / %d / %d; identical: %s"
          % (d_all["n"], d_rest["n"], d_resting["n"], same_n))
    if same_n:
        print("  under JOIN a crossing limit rests at its own near touch; "
              "the only aggressive noise flow is")
        print("  the p_market branch, so nothing happens between submission "
              "and resting: departure is all 3->4")
    print("")

    print("=" * 100)
    print("B. Stacking due to JOIN clipping")
    print("=" * 100)
    print("  limit orders submitted            %9d" % n_limit)
    print("  of those, CLIPPED at submission   %9d   (%.2f%%)"
          % (n_clip_sub, 100.0 * n_clip_sub / max(1, n_limit)))
    n_c, n_u = len(pool(True)), len(pool(False))
    print("  near-touch RESTING observations   %9d   (clipped %d = %.2f%%, "
          "unclipped %d = %.2f%%)"
          % (n_c + n_u, n_c, 100.0 * n_c / max(1, n_c + n_u),
             n_u, 100.0 * n_u / max(1, n_c + n_u)))
    print("  untagged near-touch orders        %9d   (informed posts etc.; "
          "excluded from the split)" % n_unknown)
    print("  side-tag mismatches               %9d   (buy resting on ask book or vice versa; must be 0)" % n_mismatch)
    print("")

    # --- by side: clipping is directional ---
    print("  by side (clipping is directional):")
    print("  %-22s %12s %12s %12s %14s %14s"
          % ("", "submitted", "clipped", "clip %", "near-touch clp",
             "near-touch unc"))
    for sd, book in (("buy", "bid"), ("sell", "ask")):
        nc_s, nu_s = len(near[(True, sd)]), len(near[(False, sd)])
        print("  %-22s %12d %12d %11.2f%% %14d %14d"
              % ("%s (rests on %s)" % (sd, book), n_lim_side[sd],
                 n_clip_side[sd],
                 100.0 * n_clip_side[sd] / max(1, n_lim_side[sd]),
                 nc_s, nu_s))
    print("")
    print("  %-22s %12s %12s %12s"
          % ("orders/level BY SIDE", "n", "median", "mean"))
    for sd, book in (("buy", "bid"), ("sell", "ask")):
        for lbl, c in (("CLIPPED", True), ("UNCLIPPED", False)):
            dd = describe([float(x) for x in opl[(c, sd)]])
            print("  %-22s %12d %12.5f %12.5f"
                  % ("%s %s" % (book, lbl), dd["n"], dd["median"], dd["mean"]))
    print("")
    print("  near-touch clipped share >> submission share => clipped orders "
          "concentrate at the touch")
    print("")
    print(HDR)
    print(row("near-touch size, CLIPPED orders", describe(pool(True))))
    print(row("near-touch size, UNCLIPPED orders", describe(pool(False))))
    print("")
    print(HDR)
    print(row("orders/level, ALL (resting_size C)",
              describe([float(x) for x in tot_opl])))
    print(row("orders/level, CLIPPED only",
              describe([float(x) for x in pool_opl(True)])))
    print(row("orders/level, UNCLIPPED only",
              describe([float(x) for x in pool_opl(False)])))
    print("")
    print("  clipped/unclipped counts are per level and category (sum to the total "
          "only on levels with both); total also includes untagged orders")
    print("")
    print("  caveats:")
    print("   - clipped orders (aggressively priced draws) are a selected "
          "population, not a random half")
    print("   - clipping -> stacking is near-definitional; the finding is the "
          "magnitude, and that unclipped never stack")
    print("   - 12.42% vs 91.64%: submission counters include the warm-up hour, "
          "scans do not (clip rate stationary)")
    print("")

    print("=" * 100)
    print("C. Population carrying the 1.69x")
    print("=" * 100)
    d_c, d_u = describe(pool(True)), describe(pool(False))
    print("  %-38s %10s %10s %10s %10s %10s"
          % ("ratio vs source", "p25", "median", "p75", "p95", "mean"))
    print(ratio_row("near-touch CLIPPED / source", d_c, d_src))
    print(ratio_row("near-touch UNCLIPPED / source", d_u, d_src))
    print(ratio_row("near-touch POOLED / source", d_near, d_src))
    print("")
    print("  unclipped share of near-touch observations: %.2f%%"
          % (100.0 * n_u / max(1, n_c + n_u)))
    print("")
    print("  size selection sits in the UNCLIPPED minority, via survival:")
    print("  CLIPPED orders are replaced by a constant stream of fresh arrivals; "
          "sizes stay near the draw")
    print("  UNCLIPPED orders stay only if they survive: small ones consumed "
          "whole, large ones keep a remainder")
    print("")
    print("  wall clock %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
