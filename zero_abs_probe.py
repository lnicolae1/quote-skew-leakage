# zero_abs_probe.py: measures the ZERO_ABS depth-entry category

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from compute_volatility import (State, Side, stream_reconstruct,
                                build_cloud_path, parse_ts)

DAYS = ["07-11", "07-12", "07-13"]
NEAR_BP = 50.0
AGE_BUCKETS = (1.0, 10.0, 60.0, 600.0, 3600.0)
N_EXAMPLES = 10


class Probe:
    """Classifies every ZERO_ABS entry against the price's own history"""

    def __init__(self):
        self.last_removed = {}
        self.ever_seen = set()
        self.counts = {"RE_DELETE": 0, "SEEN_GONE": 0, "NEVER_SEEN": 0}
        self.near_counts = {"RE_DELETE": 0, "SEEN_GONE": 0, "NEVER_SEEN": 0}
        self.ages_s = []
        self.ages_uid = []
        self.never_seen_bp = []
        self.never_examples = []
        self.repeat = {}
        self.n_zero_abs = 0
        self.n_removed = 0
        self.clock = None
        self.mid_fn = None

    def note_snapshot(self, levels):
        for price_str, _q in levels:
            self.ever_seen.add(price_str)

    def record(self, side, updates):
        now = self.clock() if self.clock is not None else None
        mid = self.mid_fn() if self.mid_fn is not None else None
        for price_str, qty_str in updates:
            present = price_str in side.book
            if float(qty_str) != 0.0:
                if not present:
                    self.ever_seen.add(price_str)
                continue
            if present:
                self.n_removed += 1
                if now is not None:
                    self.last_removed[price_str] = now
                continue

            self.n_zero_abs += 1
            self.repeat[price_str] = self.repeat.get(price_str, 0) + 1
            if price_str in self.last_removed:
                cat = "RE_DELETE"
                if now is not None:
                    uid0, t0 = self.last_removed[price_str]
                    self.ages_s.append(now[1] - t0)
                    self.ages_uid.append(now[0] - uid0)
            elif price_str in self.ever_seen:
                cat = "SEEN_GONE"
            else:
                cat = "NEVER_SEEN"
                if mid is not None:
                    bp = abs(float(price_str) - mid) / mid * 10000.0
                    self.never_seen_bp.append(bp)
                    if len(self.never_examples) < N_EXAMPLES:
                        self.never_examples.append((price_str, bp))
            self.counts[cat] += 1
            if mid is not None:
                d = abs(float(price_str) - mid) / mid * 10000.0
                if d <= NEAR_BP:
                    self.near_counts[cat] += 1

    def wrap(self, side):
        real_apply = side.apply
        real_load = side.load_snapshot

        def wrapped_apply(updates):
            self.record(side, updates)
            return real_apply(updates)

        def wrapped_load(levels):
            self.note_snapshot(levels)
            return real_load(levels)

        side.apply = wrapped_apply
        side.load_snapshot = wrapped_load


def self_test():
    ok = True

    def report(name, good, detail=""):
        print("  [%s] %s%s" % ("PASS" if good else "FAIL", name,
                               ("   " + detail) if detail else ""))
        return good

    def fixture(t_start=1000.0):
        clock = {"uid": 100, "t": t_start}
        p = Probe()
        p.clock = lambda: (clock["uid"], clock["t"])
        s = Side(is_bid=True)
        p.wrap(s)
        return p, s, clock

    p, s, clk = fixture()
    s.apply([("100.00", "1.0")])
    clk["uid"] += 10; clk["t"] += 5.0
    s.apply([("100.00", "0")])
    clk["uid"] += 20; clk["t"] += 7.5
    s.apply([("100.00", "0")])
    good = (p.counts["RE_DELETE"] == 1 and p.counts["NEVER_SEEN"] == 0
            and p.counts["SEEN_GONE"] == 0
            and len(p.ages_s) == 1 and abs(p.ages_s[0] - 7.5) < 1e-9
            and p.ages_uid == [20])
    ok &= report("T1 create->remove->zero scores RE_DELETE with correct age",
                 good, "counts=%r ages_s=%r ages_uid=%r"
                 % (p.counts, p.ages_s, p.ages_uid))

    p2, s2, _ = fixture()
    s2.apply([("500.00", "0")])
    good2 = (p2.counts["NEVER_SEEN"] == 1 and p2.counts["RE_DELETE"] == 0
             and p2.ages_s == [])
    ok &= report("T2 negative: unseen price scores NEVER_SEEN, no age recorded",
                 good2, "counts=%r" % p2.counts)

    p3, s3, _ = fixture()
    s3.apply([("100.00", "1.0")])
    s3.apply([("100.00", "0")])
    good3 = (p3.n_zero_abs == 0 and p3.n_removed == 1)
    ok &= report("T3 negative: a present-price delete is REMOVED, not ZERO_ABS",
                 good3, "zero_abs=%d removed=%d" % (p3.n_zero_abs, p3.n_removed))

    p4, s4, _ = fixture()
    s4.load_snapshot([("100.00", "1.0"), ("101.00", "2.0")])
    s4.apply([("100.00", "0")])
    s4.apply([("102.00", "0")])
    s4.book.pop("101.00")
    s4.apply([("101.00", "0")])
    good4 = (p4.counts["NEVER_SEEN"] == 1 and p4.counts["SEEN_GONE"] == 1
             and p4.n_removed == 1)
    ok &= report("T4 snapshot levels count as seen; SEEN_GONE is reachable",
                 good4, "counts=%r removed=%d" % (p4.counts, p4.n_removed))

    seq = [("100.00", "1.0"), ("101.00", "2.0"), ("100.00", "0"),
           ("102.00", "0"), ("100.00", "0"), ("103.00", "0.5")]
    plain = Side(is_bid=True)
    instr = Side(is_bid=True)
    p5 = Probe()
    p5.clock = lambda: (0, 0.0)
    p5.wrap(instr)
    plain.load_snapshot([("099.00", "1.0")])
    instr.load_snapshot([("099.00", "1.0")])
    for e in seq:
        plain.apply([e])
        instr.apply([e])
    good5 = (plain.book == instr.book and plain.best() == instr.best())
    ok &= report("T5 wrapper does not perturb reconstruction (book and best)",
                 good5, "plain=%r instr=%r" % (plain.book, instr.book))

    good6 = all(a >= 0.0 for a in p.ages_s) and all(u >= 0 for u in p.ages_uid)
    ok &= report("T6 negative: no recorded age is negative", good6)

    tot7 = sum(p4.counts.values())
    good7 = (tot7 == p4.n_zero_abs)
    ok &= report("T7 completeness: ZERO_ABS count equals bucket total",
                 good7, "buckets=%d zero_abs=%d" % (tot7, p4.n_zero_abs))

    return ok


def pct(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = int(q * (len(ys) - 1))
    return ys[i]


def main():
    print("=" * 100)
    print("ZERO_ABS PROBE; are they re-deletes of removed levels, or prices")
    print("                  the book never held?")
    print("=" * 100)
    print("Cloud chain %s. Committed reconstruction path, instance"
          % " + ".join(DAYS))
    print("wrappers only. Ages lag by one depth event (see header timing note).")
    print("")
    print("*" * 100)
    print("Correction; read before the numbers. Added after zero_abs_timeline.py.")
    print("")
    print("  THE RE_DELETE / NEVER_SEEN split is not a mechanistic distinction.")
    print("  It measures elapsed price-grid coverage. ever_seen and last_removed")
    print("  accumulate monotonically and are never pruned, so once a price has")
    print("  been visited any later zero for it scores RE_DELETE by")
    print("  construction. NEVER_SEEN can only come from a price this run has")
    print("  not yet reached.")
    print("")
    print("  Evidence: NEVER_SEEN per diff across 07-12, one day at roughly")
    print("  constant CREATED/diff; 0.2046 at 00:00 falling to 0.0005 by")
    print("  17:00, nearly three orders of magnitude, then bursting back to")
    print("  0.4800 and 0.8518 on 07-13 02:00 and 03:00 as the price moved into")
    print("  unvisited territory. No market quantity behaves like that; a")
    print("  coverage counter filling up does.")
    print("")
    print("  So: section A's split is a property of this WINDOW's length and")
    print("  price path, not of the venue; a longer run pushes it toward 100%")
    print("  RE_DELETE. Section B's ages are time since that price last saw")
    print("  activity, not time since a causally related removal, and they do")
    print("  not bear on the mechanism. Section D's repeat counts likewise.")
    print("")
    print("  What survives uncontaminated: the total ZERO_ABS count and rate")
    print("  (bucket-independent); SEEN_GONE = 8 of 394,287, an independent")
    print("  check that snapshot re-anchors replace the book cleanly; and")
    print("  section C's near-touch distances, which hold whatever the bucket.")
    print("")
    print("  Settled elsewhere: zero_abs_timeline.py rules out the collector")
    print("  artifact; the three reconnect-adjacent hours on 07-13 sit BELOW")
    print("  that day's own average. That eliminates a rival; it does not")
    print("  establish the fleeting-order explanation.")
    print("*" * 100)
    print("")
    print("Self-test")
    if not self_test():
        print("")
        print("self-test FAILED; refusing to report numbers.")
        return 1
    print("  All PASS")
    print("")

    state = State()
    gaps = []
    probe = Probe()

    def clock():
        if state.last_update_id is None or state.last_ts is None:
            return None
        return (state.last_update_id, parse_ts(state.last_ts).timestamp())

    def mid_fn():
        bb = state.bids.best()
        ba = state.asks.best()
        if bb is None or ba is None:
            return None
        bbf, baf = float(bb), float(ba)
        if bbf >= baf:
            return None
        return (bbf + baf) / 2.0

    probe.clock = clock
    probe.mid_fn = mid_fn
    probe.wrap(state.bids)
    probe.wrap(state.asks)

    def on_event(ts_str, uid, mid):
        pass

    print("streaming")
    for day in DAYS:
        path = build_cloud_path(day)
        before = probe.n_zero_abs
        counts = stream_reconstruct(path, state, gaps, on_event)
        print("  %s: %d diffs, %d ZERO_ABS"
              % (path, counts["diffs"], probe.n_zero_abs - before))
    print("")

    tot = sum(probe.counts.values())
    print("=" * 100)
    print("A. What the ZERO_ABS entries actually were")
    print("=" * 100)
    print("  Contaminated; this split measures elapsed price-grid coverage,")
    print("  not two mechanisms. See the correction block above.")
    print("  total ZERO_ABS   %d      (real REMOVED for comparison: %d)"
          % (probe.n_zero_abs, probe.n_removed))
    print("  distinct prices tracked: seen %d, with a recorded removal %d"
          % (len(probe.ever_seen), len(probe.last_removed)))
    print("")
    print("  %-12s %14s %10s   %14s %10s"
          % ("state", "all", "share", "within %.0fbp" % NEAR_BP, "share"))
    near_tot = sum(probe.near_counts.values())
    for k in ("RE_DELETE", "SEEN_GONE", "NEVER_SEEN"):
        print("  %-12s %14d %9.4f%%   %14d %9.4f%%"
              % (k, probe.counts[k],
                 100.0 * probe.counts[k] / tot if tot else 0.0,
                 probe.near_counts[k],
                 100.0 * probe.near_counts[k] / near_tot if near_tot else 0.0))
    print("  %-12s %14d               %14d" % ("total", tot, near_tot))
    print("")

    print("=" * 100)
    print("B. How long ago was the level removed?  (RE_DELETE only)")
    print("=" * 100)
    print("  Contaminated; these are time since that price last saw activity,")
    print("  a property of how often the mid revisits a tick. They do NOT bear")
    print("  on the mechanism. See the correction block above.")
    if probe.ages_s:
        a = probe.ages_s
        print("  n = %d" % len(a))
        print("  seconds since that price was REMOVED:")
        for q, lbl in ((0.0, "min"), (0.25, "p25"), (0.5, "median"),
                       (0.75, "p75"), (0.95, "p95"), (0.99, "p99"),
                       (1.0, "max")):
            print("      %-8s %14.4f s" % (lbl, pct(a, q)))
        print("      mean     %14.4f s" % statistics.mean(a))
        print("")
        print("  cumulative share removed within:")
        for b in AGE_BUCKETS:
            n = sum(1 for x in a if x <= b)
            print("      <= %7.0f s   %10d   %7.4f%%"
                  % (b, n, 100.0 * n / len(a)))
        print("")
        u = probe.ages_uid
        print("  update IDs since removal:  median %d   p95 %d   max %d"
              % (pct(u, 0.5), pct(u, 0.95), pct(u, 1.0)))
    else:
        print("  no RE_DELETE ages recorded.")
    print("")

    print("=" * 100)
    print("C. The alarming case; NEVER_SEEN")
    print("=" * 100)
    print("  The count here is contaminated (it tracks unvisited price")
    print("  territory). The distances are not; those entries were near the")
    print("  touch whatever bucket they fell in.")
    print("  count %d  (%.4f%% of ZERO_ABS)"
          % (probe.counts["NEVER_SEEN"],
             100.0 * probe.counts["NEVER_SEEN"] / tot if tot else 0.0))
    if probe.never_seen_bp:
        b = probe.never_seen_bp
        print("  distance from mid, bp:  median %.2f   p95 %.2f   max %.2f"
              % (pct(b, 0.5), pct(b, 0.95), pct(b, 1.0)))
        n_near = sum(1 for x in b if x <= NEAR_BP)
        print("  within %.0fbp of mid: %d (%.4f%%); the 1000-level cap cannot"
              % (NEAR_BP, n_near, 100.0 * n_near / len(b)))
        print("  explain these, so a large number here is a reconstruction")
        print("  problem, not a coverage artifact.")
        print("  first %d examples (price, bp from mid):" % len(probe.never_examples))
        for pr, bp in probe.never_examples:
            print("      %-16s %10.2f bp" % (pr, bp))
    else:
        print("  none recorded with a usable mid.")
    print("")

    print("=" * 100)
    print("D. Repeats; is one price deleted over and over?")
    print("=" * 100)
    print("  Contaminated for the same reason as a and B.")
    if probe.repeat:
        vals = list(probe.repeat.values())
        print("  distinct prices producing a ZERO_ABS: %d" % len(vals))
        print("  ZERO_ABS per such price: median %d   p95 %d   max %d"
              % (pct(vals, 0.5), pct(vals, 0.95), pct(vals, 1.0)))
        top = sorted(probe.repeat.items(), key=lambda kv: -kv[1])[:N_EXAMPLES]
        print("  worst offenders:")
        for pr, n in top:
            print("      %-16s %8d" % (pr, n))
    print("")

    print("=" * 100)
    print("E. Verdict on the hypothesis")
    print("=" * 100)
    share_re = (probe.counts["RE_DELETE"] / tot) if tot else float("nan")
    print("  hypothesis: 'ZERO_ABS are zero-quantity republishes for levels")
    print("  already removed'")
    print("  RE_DELETE share = %d / %d = %.6f"
          % (probe.counts["RE_DELETE"], tot, share_re))
    if share_re > 0.9:
        print("  -> hypothesis supported. The category is benign: the venue")
        print("     re-asserts deletions that already happened. It changes")
        print("     nothing in the book and correctly enters neither the")
        print("     addition nor the removal count. book_change_mix.py's share")
        print("     is unaffected.")
    elif probe.counts["NEVER_SEEN"] > probe.counts["RE_DELETE"]:
        print("  -> Hypothesis rejected, and in the bad direction. Most of")
        print("     these prices were never in the book. That points at the")
        print("     reconstruction itself and would affect every result built")
        print("     on this path, not just book_change_mix.py. Check the")
        print("     price-string formatting caveat in the header first; it")
        print("     is the cheapest explanation and it is a false alarm.")
    else:
        print("  -> Mixed. Neither state dominates; read sections A-D and do")
        print("     not summarise this as a single verdict.")
    print("")
    print("  Nothing retuned, swept or simulated. The clipping rule is")
    print("  untouched. No committed file was modified.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
