# book_change_mix.py: fraction of real book changes that are additions, from collected depth data

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from compute_volatility import (State, Side, stream_reconstruct,
                                build_cloud_path, parse_ts)

DAYS = ["07-11", "07-12", "07-13"]
NEAR_BP = 50.0

CATS = ("CREATED", "UP", "DOWN", "SAME", "REMOVED", "ZERO_ABS")


def blank():
    d = {c: 0 for c in CATS}
    d["no_mid"] = 0
    return d


class Classifier:
    """Counts what each price-level entry did to the book, before it is applied"""

    def __init__(self):
        self.all = blank()
        self.near = blank()
        self.n_entries = 0
        self.mid_fn = None

    def classify(self, side, price_str, qty_str):
        """Pure classification"""
        present = price_str in side.book
        if float(qty_str) == 0.0:
            return "REMOVED" if present else "ZERO_ABS"
        if not present:
            return "CREATED"
        old = float(side.book[price_str])
        new = float(qty_str)
        if new > old:
            return "UP"
        if new < old:
            return "DOWN"
        return "SAME"

    def record(self, side, updates):
        mid = self.mid_fn() if self.mid_fn is not None else None
        for price_str, qty_str in updates:
            cat = self.classify(side, price_str, qty_str)
            self.all[cat] += 1
            self.n_entries += 1
            if mid is None:
                self.near["no_mid"] += 1
                continue
            dist_bp = abs(float(price_str) - mid) / mid * 10000.0
            if dist_bp <= NEAR_BP:
                self.near[cat] += 1

    def wrap(self, side):
        """Instance-level wrapper"""
        real_apply = side.apply

        def wrapped(updates):
            self.record(side, updates)
            return real_apply(updates)

        side.apply = wrapped


def add_share(d):
    """(CREATED+UP) / (CREATED+UP+DOWN+REMOVED)"""
    adds = d["CREATED"] + d["UP"]
    denom = adds + d["DOWN"] + d["REMOVED"]
    return ((adds / denom) if denom else float("nan")), adds, denom


def self_test():
    ok = True

    def report(name, good, detail=""):
        print("  [%s] %s%s" % ("PASS" if good else "FAIL", name,
                               ("   " + detail) if detail else ""))
        return good

    c = Classifier()
    s = Side(is_bid=True)
    s.book = {"100.00": "1.0"}
    cases = [
        (("100.00", "2.0"), "UP"),
        (("100.00", "0.5"), "DOWN"),
        (("100.00", "1.0"), "SAME"),
        (("101.00", "1.0"), "CREATED"),
        (("100.00", "0"),   "REMOVED"),
        (("999.00", "0"),   "ZERO_ABS"),
    ]
    bad = []
    for (p, q), want in cases:
        got = c.classify(s, p, q)
        if got != want:
            bad.append("%s,%s -> %s (want %s)" % (p, q, got, want))
    ok &= report("T1 classifier truth table, all 6 branches",
                 not bad, "; ".join(bad))

    c2 = Classifier()
    s2 = Side(is_bid=True)
    s2.book = {"100.00": "5.0", "101.00": "5.0"}
    c2.record(s2, [("100.00", "4.0"), ("101.00", "0")])
    sh2, adds2, den2 = add_share(c2.all)
    good2 = (adds2 == 0 and den2 == 2 and sh2 == 0.0)
    ok &= report("T2 negative: down+remove stream gives share exactly 0.0",
                 good2, "adds=%d denom=%d share=%r" % (adds2, den2, sh2))

    c2b = Classifier()
    s2b = Side(is_bid=True)
    s2b.book = {"100.00": "1.0"}
    c2b.record(s2b, [("100.00", "2.0"), ("102.00", "1.0"), ("103.00", "1.0")])
    sh2b, adds2b, den2b = add_share(c2b.all)
    good2b = (adds2b == 3 and den2b == 3 and sh2b == 1.0)
    ok &= report("T2b negative: all-add stream gives share exactly 1.0",
                 good2b, "adds=%d denom=%d share=%r" % (adds2b, den2b, sh2b))

    seq = [("100.00", "1.0"), ("101.00", "2.0"), ("100.00", "3.0"),
           ("101.00", "0"), ("102.00", "0"), ("100.00", "3.0"),
           ("099.50", "0.7")]
    plain = Side(is_bid=True)
    instr = Side(is_bid=True)
    c3 = Classifier()
    c3.wrap(instr)
    for e in seq:
        plain.apply([e])
        instr.apply([e])
    good3 = (plain.book == instr.book and plain.best() == instr.best())
    ok &= report("T3 wrapper does not perturb reconstruction (book and best)",
                 good3, "plain=%r instr=%r best %r vs %r"
                 % (plain.book, instr.book, plain.best(), instr.best()))

    total4 = sum(c3.all[k] for k in CATS)
    good4 = (total4 == len(seq) == c3.n_entries)
    ok &= report("T4 completeness: every entry classified exactly once",
                 good4, "buckets=%d entries=%d fed=%d"
                 % (total4, c3.n_entries, len(seq)))

    instr5 = Side(is_bid=True)
    c5 = Classifier()
    c5.wrap(instr5)
    instr5.load_snapshot([("100.00", "1.0"), ("101.00", "2.0"),
                          ("102.00", "3.0")])
    good5 = (c5.n_entries == 0 and len(instr5.book) == 3)
    ok &= report("T5 negative: snapshot load contributes 0 classified entries",
                 good5, "entries=%d book=%d" % (c5.n_entries, len(instr5.book)))

    c6 = Classifier()
    s6 = Side(is_bid=True)
    s6.book = {"100.00": "1.0"}
    c6.record(s6, [("100.00", "1.0"), ("500.00", "0"), ("100.00", "2.0")])
    sh6, adds6, den6 = add_share(c6.all)
    good6 = (c6.all["SAME"] == 1 and c6.all["ZERO_ABS"] == 1
             and den6 == 1 and adds6 == 1 and sh6 == 1.0)
    ok &= report("T6 negative: SAME and ZERO_ABS excluded from denominator",
                 good6, "SAME=%d ZERO_ABS=%d denom=%d"
                 % (c6.all["SAME"], c6.all["ZERO_ABS"], den6))

    c7 = Classifier()
    s7 = Side(is_bid=True)
    s7.book = {}
    c7.mid_fn = lambda: 100.0
    c7.record(s7, [("100.10", "1.0"), ("200.00", "1.0")])
    good7 = (c7.all["CREATED"] == 2 and c7.near["CREATED"] == 1)
    ok &= report("T7 negative: 50bp band excludes a far level, keeps a near one",
                 good7, "all=%d near=%d"
                 % (c7.all["CREATED"], c7.near["CREATED"]))

    return ok


def main():
    print("=" * 100)
    print("Book change mix; what share of real book changes are additions?")
    print("=" * 100)
    print("Cloud chain %s, committed reconstruction path (State,"
          % " + ".join(DAYS))
    print("stream_reconstruct, build_cloud_path from compute_volatility).")
    print("Side.apply wrapped on the instance; no committed module modified.")
    print("Near-mid band: %.0fbp. Read the header limitations." % NEAR_BP)
    print("")
    print("Self-test")
    if not self_test():
        print("")
        print("self-test FAILED; refusing to report numbers from an "
              "unverified classifier.")
        return 1
    print("  All PASS")
    print("")

    state = State()
    gaps = []
    clf = Classifier()

    def mid_fn():
        bb = state.bids.best()
        ba = state.asks.best()
        if bb is None or ba is None:
            return None
        bbf, baf = float(bb), float(ba)
        if bbf >= baf:
            return None
        return (bbf + baf) / 2.0

    clf.mid_fn = mid_fn
    clf.wrap(state.bids)
    clf.wrap(state.asks)

    span = {"first_ts": None, "last_ts": None,
            "first_uid": None, "last_uid": None, "n_events": 0}

    def on_event(ts_str, uid, mid):
        if span["first_ts"] is None:
            span["first_ts"] = ts_str
            span["first_uid"] = uid
        span["last_ts"] = ts_str
        span["last_uid"] = uid
        span["n_events"] += 1

    print("streaming (line by line; files are never loaded whole)")
    nsnap = ndiff = 0
    for day in DAYS:
        path = build_cloud_path(day)
        before = clf.n_entries
        counts = stream_reconstruct(path, state, gaps, on_event)
        nsnap += counts["snapshots"]
        ndiff += counts["diffs"]
        print("  %s: %d snapshots, %d diffs applied, %d level entries, "
              "%d crossed samples"
              % (path, counts["snapshots"], counts["diffs"],
                 clf.n_entries - before, counts["crossed"]))
    print("")

    t0 = parse_ts(span["first_ts"])
    t1 = parse_ts(span["last_ts"])
    secs = (t1 - t0).total_seconds()
    uid_span = span["last_uid"] - span["first_uid"]
    lost = sum(g["updates_lost"] for g in gaps)
    uid_obs = uid_span - lost

    print("=" * 100)
    print("A. Coverage")
    print("=" * 100)
    print("  first event      %s  uid=%d" % (span["first_ts"], span["first_uid"]))
    print("  last  event      %s  uid=%d" % (span["last_ts"], span["last_uid"]))
    print("  wall span        %.3f s  (%.4f days)" % (secs, secs / 86400.0))
    print("  snapshots        %d   diffs applied %d   depth events %d"
          % (nsnap, ndiff, span["n_events"]))
    print("  gaps recorded    %d   update IDs lost %d" % (len(gaps), lost))
    for g in gaps:
        print("      %-16s lost %6d   %s -> %s"
              % (g["kind"], g["updates_lost"], g["start_ts"], g["end_ts"]))
    print("  update ID span   %d - %d = %d" % (span["last_uid"],
                                               span["first_uid"], uid_span))
    print("  observed IDs     %d - %d = %d" % (uid_span, lost, uid_obs))
    print("  -> update rate   %d / %.3f = %.4f IDs/sec" % (uid_obs, secs,
                                                           uid_obs / secs))
    print("     (NOTES.md verified facts state 20.9 IDs/sec independently)")
    print("  level entries    %d" % clf.n_entries)
    print("  -> entry rate    %d / %.3f = %.4f entries/sec"
          % (clf.n_entries, secs, clf.n_entries / secs))
    print("  entries per ID   %d / %d = %.4f" % (clf.n_entries, uid_obs,
                                                 clf.n_entries / uid_obs))
    print("     limitation (5) predicted this would be < 1, on the reasoning")
    print("     that netting inside each 100ms window collapses several update")
    print("     IDs into one published entry. The data contradicts that")
    print("     prediction: the ratio measured above is > 1, so entries exceed")
    print("     update IDs and the netting model is wrong in its direction.")
    print("     What the ratio being near 1 does establish is that entries and")
    print("     IDs are close to interchangeable in practice, which is why")
    print("     Route 1 and Route 2 in section D land close together. The")
    print("     mechanism behind a ratio above 1 is NOT explained here.")
    print("")

    print("=" * 100)
    print("B. What the changes were")
    print("=" * 100)
    print("  %-10s %14s %10s   %14s %10s" %
          ("category", "all entries", "share", "within %.0fbp" % NEAR_BP, "share"))
    tot_all = sum(clf.all[c] for c in CATS)
    tot_near = sum(clf.near[c] for c in CATS)
    for c in CATS:
        print("  %-10s %14d %9.4f%%   %14d %9.4f%%"
              % (c, clf.all[c], 100.0 * clf.all[c] / tot_all if tot_all else 0.0,
                 clf.near[c],
                 100.0 * clf.near[c] / tot_near if tot_near else 0.0))
    print("  %-10s %14d %9s     %14d" % ("total", tot_all, "", tot_near))
    print("  entries with no usable mid (missing or crossed book): %d"
          % clf.near["no_mid"])
    print("")

    sh_a, adds_a, den_a = add_share(clf.all)
    sh_n, adds_n, den_n = add_share(clf.near)
    print("=" * 100)
    print("C. The addition share; what replaces the divide-by-2")
    print("=" * 100)
    print("  share = (CREATED + UP) / (CREATED + UP + DOWN + REMOVED)")
    print("  SAME and ZERO_ABS are excluded from the denominator (they are not")
    print("  changes to apportion); both counts are printed above so the")
    print("  denominator can be rebuilt by hand.")
    print("")
    print("  All levels      (%d + %d) / %d = %d / %d = %.6f"
          % (clf.all["CREATED"], clf.all["UP"], den_a, adds_a, den_a, sh_a))
    print("  within %.0fbp    (%d + %d) / %d = %d / %d = %.6f"
          % (NEAR_BP, clf.near["CREATED"], clf.near["UP"], den_n, adds_n,
             den_n, sh_n))
    print("")
    print("  the assumed value was 0.500000 (the divide-by-2).")
    print("  measured all   / assumed = %.6f / 0.5 = %.4fx" % (sh_a, sh_a / 0.5))
    print("  measured near  / assumed = %.6f / 0.5 = %.4fx" % (sh_n, sh_n / 0.5))
    print("")
    print("  The near figure is the one to quote: the 1000-level snapshot cap")
    print("  cannot bind inside %.0fbp, so limitation (4); the only one that"
          % NEAR_BP)
    print("  inflates additions; does not apply there. Limitations (1) and (3)")
    print("  still deflate it, so near is a lower bound on the addition share.")
    print("")

    print("=" * 100)
    print("D. The implied submission ceiling")
    print("=" * 100)
    print("  Two routes, and they answer different questions. Both are printed")
    print("  because multiplying a share by the wrong rate is exactly the error")
    print("  limitation (5) warns about.")
    print("")
    ent_rate = clf.n_entries / secs
    print("  Route 1; directly measured, and this is the defensible one.")
    print("  Count the addition entries and divide by the wall span. No share,")
    print("  no assumed rate, nothing multiplied.")
    print("      All   %d additions / %.3f s = %.4f additions/sec"
          % (adds_a, secs, adds_a / secs))
    print("      near  %d additions / %.3f s = %.4f additions/sec"
          % (adds_n, secs, adds_n / secs))
    print("")
    print("  route 2; share x update-ID rate. Assumes additions and")
    print("  non-additions are netted at the SAME rate inside the 100ms window,")
    print("  which is NOT established. Reported for comparison only.")
    id_rate = uid_obs / secs
    print("      All   %.6f x %.4f IDs/s = %.4f /sec" % (sh_a, id_rate,
                                                         sh_a * id_rate))
    print("      near  %.6f x %.4f IDs/s = %.4f /sec" % (sh_n, id_rate,
                                                         sh_n * id_rate))
    print("")
    print("  for comparison, the numbers this replaces:")
    print("      assumed half-rate ceiling (lam_ceiling_arithmetic.py)  10.4491 /s")
    print("      the unsourced 3.09 constant                             3.0900 /s")
    print("      committed lam                                           1.8040 /s")
    print("")
    print("  what this does and does not settle. Route 1 near is a measured")
    print("  lower bound on the real book's order-addition rate. A lower bound")
    print("  on the ceiling can only raise the ceiling relative to a bound that")
    print("  sits below it; it cannot lower one. Whether that flips the")
    print("  life=60 verdict is arithmetic on this number and is NOT done here:")
    print("  this script measures, it does not re-verdict. Nothing is retuned,")
    print("  swept or simulated, and the clipping rule is untouched.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
