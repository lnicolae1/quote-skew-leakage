# zero_abs_timeline.py: ZERO_ABS over time: market regime or collector artifact?

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from compute_volatility import (State, Side, stream_reconstruct,
                                build_cloud_path, parse_ts)

DAYS = ["07-11", "07-12", "07-13"]
ADJACENT_HOURS = 1


def hour_key(ts_str):
    """'2026-07-13T22:31:05.214118+00:00' -> '07-13 22'"""
    d, t = ts_str.split("T")
    return d[5:] + " " + t[:2]


def blank_bucket():
    return {"diffs": 0, "RE_DELETE": 0, "SEEN_GONE": 0, "NEVER_SEEN": 0,
            "CREATED": 0, "REMOVED": 0}


class Timeline:
    def __init__(self):
        self.last_removed = {}
        self.ever_seen = set()
        self.buckets = {}
        self.order = []
        self.snapshot_hours = []
        self.n_zero_abs = 0
        self.now_hour = None

    def bucket(self):
        h = self.now_hour() if self.now_hour is not None else None
        if h is None:
            h = "PRE-ANCHOR"
        if h not in self.buckets:
            self.buckets[h] = blank_bucket()
            self.order.append(h)
        return self.buckets[h]

    def note_snapshot(self, levels):
        h = self.now_hour() if self.now_hour is not None else None
        self.snapshot_hours.append(h if h is not None else "CHAIN START")
        for price_str, _q in levels:
            self.ever_seen.add(price_str)

    def record(self, side, updates):
        b = self.bucket()
        if side.is_bid:
            b["diffs"] += 1
        for price_str, qty_str in updates:
            present = price_str in side.book
            if float(qty_str) != 0.0:
                if not present:
                    b["CREATED"] += 1
                    self.ever_seen.add(price_str)
                continue
            if present:
                b["REMOVED"] += 1
                self.last_removed[price_str] = True
                continue
            self.n_zero_abs += 1
            if price_str in self.last_removed:
                b["RE_DELETE"] += 1
            elif price_str in self.ever_seen:
                b["SEEN_GONE"] += 1
            else:
                b["NEVER_SEEN"] += 1

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

    good1 = (hour_key("2026-07-13T22:31:05.214118+00:00") == "07-13 22"
             and hour_key("2026-07-11T00:00:00.000001+00:00") == "07-11 00")
    ok &= report("T1 hour_key extracts day and hour", good1,
                 hour_key("2026-07-13T22:31:05.214118+00:00"))

    a = hour_key("2026-07-12T23:59:59.999999+00:00")
    b = hour_key("2026-07-13T00:00:00.000001+00:00")
    ok &= report("T2 negative: 23:59:59 and 00:00:01 land in different buckets",
                 a != b, "%s vs %s" % (a, b))

    def fixture(hour="07-13 22"):
        tl = Timeline()
        tl.now_hour = lambda: hour
        s_b = Side(is_bid=True)
        s_a = Side(is_bid=False)
        tl.wrap(s_b)
        tl.wrap(s_a)
        return tl, s_b, s_a

    tl, sb, sa = fixture()
    for _ in range(5):
        sb.apply([("100.00", "1.0")])
        sa.apply([("101.00", "1.0")])
    good3 = (tl.buckets["07-13 22"]["diffs"] == 5)
    ok &= report("T3 bid+ask apply pair counts as one diff", good3,
                 "diffs=%d (want 5)" % tl.buckets["07-13 22"]["diffs"])

    tl4, sb4, _ = fixture()
    sb4.load_snapshot([("100.00", "1.0"), ("101.00", "2.0")])
    n4 = tl4.buckets["07-13 22"]["diffs"] if "07-13 22" in tl4.buckets else 0
    good4 = (n4 == 0 and len(tl4.snapshot_hours) == 1)
    ok &= report("T4 negative: snapshot load adds 0 diffs, marks 1 snapshot",
                 good4, "diffs=%d snaps=%d" % (n4, len(tl4.snapshot_hours)))

    seq = [("100.00", "1.0"), ("101.00", "2.0"), ("100.00", "0"),
           ("102.00", "0"), ("103.00", "0.5")]
    plain = Side(is_bid=True)
    instr = Side(is_bid=True)
    tl5 = Timeline()
    tl5.now_hour = lambda: "07-13 22"
    tl5.wrap(instr)
    plain.load_snapshot([("099.00", "1.0")])
    instr.load_snapshot([("099.00", "1.0")])
    for e in seq:
        plain.apply([e])
        instr.apply([e])
    good5 = (plain.book == instr.book and plain.best() == instr.best())
    ok &= report("T5 wrapper does not perturb reconstruction (book and best)",
                 good5, "plain=%r instr=%r" % (plain.book, instr.book))

    tl6 = Timeline()
    cur = {"h": "07-13 21"}
    tl6.now_hour = lambda: cur["h"]
    s6 = Side(is_bid=True)
    tl6.wrap(s6)
    s6.apply([("100.00", "1.0")])
    cur["h"] = "07-13 22"
    s6.apply([("101.00", "1.0")])
    good6 = (tl6.buckets["07-13 21"]["diffs"] == 1
             and tl6.buckets["07-13 22"]["diffs"] == 1
             and len(tl6.buckets) == 2)
    ok &= report("T6 negative: entries follow the clock into separate buckets",
                 good6, "buckets=%r" % sorted(tl6.buckets))

    tl7, sb7, _ = fixture()
    sb7.apply([("100.00", "1.0")])
    sb7.apply([("100.00", "0")])
    sb7.apply([("100.00", "0")])
    sb7.apply([("500.00", "0")])
    bb = tl7.buckets["07-13 22"]
    tot7 = bb["RE_DELETE"] + bb["SEEN_GONE"] + bb["NEVER_SEEN"]
    good7 = (tot7 == tl7.n_zero_abs == 2 and bb["CREATED"] == 1
             and bb["REMOVED"] == 1)
    ok &= report("T7 completeness: bucket categories reconcile with total",
                 good7, "tot=%d n_zero_abs=%d" % (tot7, tl7.n_zero_abs))

    return ok


def rate(n, d):
    return (n / d) if d else float("nan")


def main():
    print("=" * 118)
    print("ZERO_ABS timeline; market regime, or collector artifact?")
    print("=" * 118)
    print("Cloud chain %s, hourly buckets. Committed reconstruction"
          % " + ".join(DAYS))
    print("path, instance wrappers only. Bucketing lags by one depth event.")
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
    tl = Timeline()
    tl.now_hour = lambda: (hour_key(state.last_ts)
                           if state.last_ts is not None else None)
    tl.wrap(state.bids)
    tl.wrap(state.asks)

    def on_event(ts_str, uid, mid):
        pass

    print("streaming")
    for day in DAYS:
        path = build_cloud_path(day)
        counts = stream_reconstruct(path, state, gaps, on_event)
        print("  %s: %d diffs, %d snapshots" % (path, counts["diffs"],
                                                counts["snapshots"]))
    print("")

    gap_hours = set()
    for g in gaps:
        for key in ("start_ts", "end_ts"):
            if g.get(key):
                gap_hours.add(hour_key(g[key]))
    snap_hours = set(h for h in tl.snapshot_hours
                     if h not in (None, "CHAIN START"))

    marked = gap_hours | snap_hours
    ordered = [h for h in tl.order if h != "PRE-ANCHOR"]
    idx = {h: i for i, h in enumerate(ordered)}
    adjacent = set()
    for h in marked:
        if h in idx:
            i = idx[h]
            for j in range(i - ADJACENT_HOURS, i + ADJACENT_HOURS + 1):
                if 0 <= j < len(ordered):
                    adjacent.add(ordered[j])

    print("=" * 118)
    print("A. The hourly timeline")
    print("=" * 118)
    print("  marks:  G = hour contains a recorded gap    S = hour contains a "
          "snapshot re-anchor    * = reconnect-adjacent")
    print("")
    print("  ZA/CREAT is the normalised column and the one to read: CREATED/diff")
    print("  itself moves with general book activity, so ZERO_ABS per diff")
    print("  confounds 'more ZERO_ABS' with 'more of everything'.")
    print("")
    print("  %-9s %8s %9s %9s %9s   %10s %10s %10s   %9s %9s  %s"
          % ("hour UTC", "diffs", "ZERO_ABS", "per diff", "ZA/CREAT",
             "RE_DEL/df", "never/df", "SEEN_GONE", "CREAT/df", "REMOV/df",
             "mark"))
    for h in tl.order:
        b = tl.buckets[h]
        mk = ""
        if h in gap_hours:
            mk += "G"
        if h in snap_hours:
            mk += "S"
        if h in adjacent:
            mk += "*"
        za = b["RE_DELETE"] + b["SEEN_GONE"] + b["NEVER_SEEN"]
        print("  %-9s %8d %9d %9.4f %9.4f   %10.4f %10.4f %10d   %9.4f %9.4f  %s"
              % (h, b["diffs"], za, rate(za, b["diffs"]),
                 rate(za, b["CREATED"]),
                 rate(b["RE_DELETE"], b["diffs"]),
                 rate(b["NEVER_SEEN"], b["diffs"]), b["SEEN_GONE"],
                 rate(b["CREATED"], b["diffs"]),
                 rate(b["REMOVED"], b["diffs"]), mk))
    print("")
    print("  recorded gaps: %d" % len(gaps))
    for g in gaps:
        print("      %-16s lost %6d   %s -> %s"
              % (g["kind"], g["updates_lost"], g["start_ts"], g["end_ts"]))
    sh = tl.snapshot_hours
    uniq = []
    for h in sh:
        if not uniq or uniq[-1] != h:
            uniq.append(h)
    print("  snapshot re-anchors: %d load calls = %d re-anchors (load_snapshot"
          % (len(sh), len(sh) // 2))
    print("  fires once per side). Hours: %s" % ", ".join(str(h) for h in uniq))
    print("")

    def agg(hours, field):
        n = sum(tl.buckets[h][field] for h in hours)
        d = sum(tl.buckets[h]["diffs"] for h in hours)
        return n, d, rate(n, d)

    adj = [h for h in ordered if h in adjacent]
    oth = [h for h in ordered if h not in adjacent]

    print("=" * 118)
    print("B. Reconnect-adjacent hours versus all other hours")
    print("   (the split was fixed before the run: +/-%d hour of a gap or a "
          "snapshot)" % ADJACENT_HOURS)
    print("=" * 118)
    print("  adjacent hours (%d): %s" % (len(adj), ", ".join(adj)))
    print("  other hours    (%d)" % len(oth))
    print("")
    print("  %-14s %12s %12s %12s %12s"
          % ("field", "adjacent", "other", "ratio", "verdict-relevant"))
    for field in ("NEVER_SEEN", "RE_DELETE", "CREATED", "REMOVED"):
        na, da, ra = agg(adj, field)
        no, do, ro = agg(oth, field)
        r = (ra / ro) if ro else float("nan")
        print("  %-14s %12.4f %12.4f %12.4fx %s"
              % (field + "/diff", ra, ro, r,
                 "<-- story B predicts this alone rises"
                 if field == "NEVER_SEEN" else ""))
    print("")

    print("=" * 118)
    print("C. Is 07-13 elevated even away from the reconnects?")
    print("   This is the discriminator. Story A (fleeting orders, a market")
    print("   regime) needs 07-13's quiet hours to be elevated too.")
    print("=" * 118)
    d13 = [h for h in oth if h.startswith("07-13")]
    dpre = [h for h in oth if not h.startswith("07-13")]
    for label, hs in (("07-11/07-12, non-adjacent", dpre),
                      ("07-13 ONLY, non-adjacent", d13)):
        if not hs:
            print("  %-28s (no hours)" % label)
            continue
        za_n = sum(tl.buckets[h]["RE_DELETE"] + tl.buckets[h]["SEEN_GONE"]
                   + tl.buckets[h]["NEVER_SEEN"] for h in hs)
        dd = sum(tl.buckets[h]["diffs"] for h in hs)
        cc = sum(tl.buckets[h]["CREATED"] for h in hs)
        _, _, rn = agg(hs, "NEVER_SEEN")
        _, _, rr = agg(hs, "RE_DELETE")
        _, _, rc = agg(hs, "CREATED")
        print("  %-28s hours=%3d diffs=%8d  ZERO_ABS/diff=%.4f"
              % (label, len(hs), dd, rate(za_n, dd)))
        print("  %-28s NEVER_SEEN/diff=%.4f  RE_DELETE/diff=%.4f  "
              "CREATED/diff=%.4f" % ("", rn, rr, rc))
        print("  %-28s normalised  ZERO_ABS/CREATED = %d / %d = %.4f"
              % ("", za_n, cc, rate(za_n, cc)))
    print("")
    print("  the normalised row is the honest comparison. CREATED/diff itself")
    print("  rose between the two periods, so part of the ZERO_ABS/diff rise is")
    print("  simply more of everything. Dividing by CREATED removes that and")
    print("  leaves the part that actually needs explaining.")
    zq = sum(tl.buckets[h]["RE_DELETE"] + tl.buckets[h]["SEEN_GONE"]
             + tl.buckets[h]["NEVER_SEEN"] for h in d13)
    cq = sum(tl.buckets[h]["CREATED"] for h in d13)
    zp = sum(tl.buckets[h]["RE_DELETE"] + tl.buckets[h]["SEEN_GONE"]
             + tl.buckets[h]["NEVER_SEEN"] for h in dpre)
    cp = sum(tl.buckets[h]["CREATED"] for h in dpre)
    dq = sum(tl.buckets[h]["diffs"] for h in d13)
    dp = sum(tl.buckets[h]["diffs"] for h in dpre)
    if cp and cq and dp and dq:
        raw_x = rate(zq, dq) / rate(zp, dp)
        nrm_x = rate(zq, cq) / rate(zp, cp)
        act_x = rate(cq, dq) / rate(cp, dp)
        print("")
        print("      07-13 quiet vs 07-11/12 quiet, as multiples:")
        print("        ZERO_ABS per diff       %.4f / %.4f = %.4fx"
              % (rate(zq, dq), rate(zp, dp), raw_x))
        print("        CREATED  per diff       %.4f / %.4f = %.4fx  <- general"
              % (rate(cq, dq), rate(cp, dp), act_x))
        print("        ZERO_ABS per CREATED    %.4f / %.4f = %.4fx  <- residual"
              % (rate(zq, cq), rate(zp, cp), nrm_x))
        print("      General activity explains a factor of %.4fx of the %.4fx."
              % (act_x, raw_x))
    print("")

    print("=" * 118)
    print("D. Which story survives")
    print("=" * 118)
    print("  *** The verdict does NOT REST on NEVER_SEEN. An earlier version of")
    print("  this block concluded from NEVER_SEEN/diff, and that quantity is")
    print("  history-contaminated: ever_seen accumulates monotonically, so once")
    print("  a price has been visited any later zero for it scores RE_DELETE")
    print("  instead. NEVER_SEEN therefore measures new price territory; how")
    print("  much of the price grid this run has not yet covered; and it")
    print("  decays as the run proceeds regardless of what the market does.")
    print("  The 07-12 column shows the decay directly: 0.2046 at 00:00 falling")
    print("  to 0.0005 by 17:00, three orders of magnitude in one day, with no")
    print("  corresponding move in CREATED/diff. See zero_abs_probe.py's")
    print("  correction block. ***")
    print("")
    print("  the verdict rests on ZERO_ABS/diff and ZERO_ABS/CREATED, which do")
    print("  not depend on bucket assignment. A ZERO_ABS is a ZERO_ABS whether")
    print("  it lands in RE_DELETE or NEVER_SEEN.")
    print("")

    def za_of(hours):
        z = sum(tl.buckets[h]["RE_DELETE"] + tl.buckets[h]["SEEN_GONE"]
                + tl.buckets[h]["NEVER_SEEN"] for h in hours)
        d = sum(tl.buckets[h]["diffs"] for h in hours)
        c = sum(tl.buckets[h]["CREATED"] for h in hours)
        return z, d, c

    all13 = [h for h in ordered if h.startswith("07-13")]
    z13, d13d, c13 = za_of(all13)
    zq, dq, cq = za_of(d13)
    zp, dp, cp = za_of(dpre)
    za13 = rate(z13, d13d)

    print("  07-13 as a whole:   ZERO_ABS/diff = %.4f   ZERO_ABS/CREATED = %.4f"
          % (za13, rate(z13, c13)))
    print("")
    print("  the reconnect hours, each against 07-13's own average:")
    n_below = 0
    for h in adj:
        b = tl.buckets[h]
        z = b["RE_DELETE"] + b["SEEN_GONE"] + b["NEVER_SEEN"]
        r = rate(z, b["diffs"])
        below = r < za13
        n_below += 1 if below else 0
        print("      %-9s ZERO_ABS/diff = %.4f   ZERO_ABS/CREATED = %.4f   %s"
              % (h, r, rate(z, b["CREATED"]),
                 "BELOW the 07-13 average" if below else "ABOVE"))
    print("")
    print("  07-13 quiet hours   %.4f  vs  07-11/12 quiet hours  %.4f"
          % (rate(zq, dq), rate(zp, dp)))
    print("  normalised          %.4f  vs                        %.4f"
          % (rate(zq, cq), rate(zp, cp)))
    print("")

    b_ruled_out = (n_below == len(adj) and len(adj) > 0)
    rise_survives_norm = (cp and cq and rate(zq, cq) / rate(zp, cp) > 1.2)

    if b_ruled_out:
        print("  -> story B (collector artifact) is ruled out. All %d"
              % len(adj))
        print("     reconnect-adjacent hours sit BELOW 07-13's own average on")
        print("     ZERO_ABS/diff. The reboot hours are among the quietest of")
        print("     the day for the very quantity the collector story needed to")
        print("     be loudest. Reconstruction was not disturbed by the reboot.")
    else:
        print("  -> Story B survives: reconnect-adjacent hours are NOT below")
        print("     07-13's average, so a collector cause cannot be dismissed.")
    print("")
    if rise_survives_norm:
        print("  And the rise is real after normalising. ZERO_ABS/CREATED is")
        print("  still %.4fx higher in 07-13's quiet hours, so general activity"
              % (rate(zq, cq) / rate(zp, cp)))
        print("  does not account for it. Something specific to ZERO_ABS moved.")
    else:
        print("  And the rise largely disappears after normalising: once")
        print("  ZERO_ABS is divided by CREATED the two periods are close, so")
        print("  the day-level rise was mostly general book activity, not")
        print("  anything specific to ZERO_ABS.")
    print("")
    print("  What is and is not established. Ruling out the collector does NOT")
    print("  establish fleeting orders. A timeline can eliminate a cause; it")
    print("  cannot demonstrate one. Story a remains the surviving candidate,")
    print("  untested. Testing it needs a different measurement; the aggTrade")
    print("  cross-reference already noted as deferred is the obvious route.")
    print("")
    print("  Nothing retuned, swept or simulated. The clipping rule is")
    print("  untouched. No committed file was modified. Nothing is recorded in")
    print("  book_change_mix.py's header on the strength of this run.")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    sys.exit(main())
