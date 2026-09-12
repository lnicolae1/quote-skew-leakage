# touch_compare.py: touch comparison deciding whether the recovered days are usable

import json
import os
import sys

from compute_volatility import State

DATA_DIR = "market_data_droplet"

CHAIN_DAYS = ["08-15", "08-16", "08-17", "08-18", "08-19", "08-20", "08-21"]
TARGET_SNAPSHOT_UID = 4547092681

TICK = 0.01

MAX_DROPS_PER_EVENT = 10000

RESULTS_PATH = "touch_compare_results.txt"

EXPECT_DIFFS = 1281392
EXPECT_CROSS_CONTROL = 1132933
EXPECT_CROSS_REPAIRED = 0
EXPECT_DROPS = 21
EXPECT_MISSING_BIDS = 20
EXPECT_MISSING_ASKS = 448

out_lines = []


def emit(s=""):
    print(s)
    out_lines.append(s)


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


class Book:
    """Duplicate of repair_validate.py's Book: see the header for why it is"""

    def __init__(self, repair):
        self.st = State()
        self.wb = {}
        self.wa = {}
        self.repair = repair
        self.n_drops = 0
        self.n_repair_events = 0

    def snapshot(self, snap):
        uid = snap["lastUpdateId"]
        self.st.bids.load_snapshot(snap["bids"])
        self.st.asks.load_snapshot(snap["asks"])
        self.wb = {p: uid for p, _ in snap["bids"]}
        self.wa = {p: uid for p, _ in snap["asks"]}
        self.st.last_update_id = uid

    def diff(self, data):
        uid = data["u"]
        self.st.bids.apply(data["b"])
        self.st.asks.apply(data["a"])
        for p, q in data["b"]:
            if float(q) == 0.0:
                self.wb.pop(p, None)
            else:
                self.wb[p] = uid
        for p, q in data["a"]:
            if float(q) == 0.0:
                self.wa.pop(p, None)
            else:
                self.wa[p] = uid
        self.st.last_update_id = uid
        if self.repair:
            self.uncross(uid)

    def uncross(self, uid):
        dropped_here = 0
        while dropped_here < MAX_DROPS_PER_EVENT:
            bb = self.st.bids.best()
            ba = self.st.asks.best()
            if bb is None or ba is None:
                return
            if float(bb) < float(ba):
                break
            ub = self.wb.get(bb, -1)
            ua = self.wa.get(ba, -1)
            if ub <= ua:
                self.st.bids.book.pop(bb, None)
                self.wb.pop(bb, None)
            else:
                self.st.asks.book.pop(ba, None)
                self.wa.pop(ba, None)
            dropped_here += 1
            self.n_drops += 1
        if dropped_here:
            self.n_repair_events += 1


def run_until_target(repair):
    bk = Book(repair)
    target_snap = None
    n_cross = 0
    n_diffs = 0
    for day in CHAIN_DAYS:
        p = path_for(day)
        if not os.path.exists(p):
            emit("  missing file: " + p)
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if '"aggTrade"' in line:
                    continue
                rec = json.loads(line)
                rt = rec.get("type")
                if rt == "snapshot":
                    snap = rec["data"]
                    if snap["lastUpdateId"] == TARGET_SNAPSHOT_UID:
                        target_snap = {"ts": rec["local_timestamp"], "data": snap,
                                       "book_uid_before": bk.st.last_update_id}
                        return bk, target_snap, n_diffs, n_cross
                    bk.snapshot(snap)
                    continue
                if rt != "diff":
                    continue
                data = rec.get("data")
                if not data or data.get("e") != "depthUpdate":
                    continue
                if bk.st.last_update_id is None or data["u"] <= bk.st.last_update_id:
                    continue
                bk.diff(data)
                n_diffs += 1
                bb = bk.st.bids.best()
                ba = bk.st.asks.best()
                if bb is not None and ba is not None and float(bb) >= float(ba):
                    n_cross += 1
        sys.stdout.flush()
    return bk, target_snap, n_diffs, n_cross


def sorted_levels(book, is_bid):
    return sorted(book.items(), key=lambda kv: (-float(kv[0]) if is_bid else float(kv[0])))


def missing_count(book, snap_side):
    return sum(1 for p in snap_side if p not in book)


emit("=" * 78)
emit("Touch comparison at the 08-21 SNAPSHOT")
emit("=" * 78)
emit("The whole-book count in repair_validate_results.txt was identical in both")
emit("arms and therefore could not separate them. Volatility reads the mid and")
emit("spread reads the TOUCH, so this measures the front of the book.")
emit("")
emit("Chain: " + " -> ".join(CHAIN_DAYS) + "   (market_data_droplet only)")
emit("Ground truth: independently-fetched REST snapshot, lastUpdateId=%d,"
     % TARGET_SNAPSHOT_UID)
emit("              compared before it is applied.")
emit("Tick size: $%.2f (measured from exchangeInfo, NOTES.md verified fact)." % TICK)

emit("")
emit("Run 1 of 2: CONTROL (no repair)")
bk_c, snap_c, nd_c, nx_c = run_until_target(repair=False)
emit("  done: diffs=%d crossed=%d drops=%d" % (nd_c, nx_c, bk_c.n_drops))

emit("")
emit("run 2 of 2: REPAIRED (crossing-triggered rule, no time constant)")
bk_r, snap_r, nd_r, nx_r = run_until_target(repair=True)
emit("  done: diffs=%d crossed=%d drops=%d across %d repair events"
     % (nd_r, nx_r, bk_r.n_drops, bk_r.n_repair_events))

if snap_c is None or snap_r is None:
    emit("")
    emit("target snapshot not found. Cannot compare. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)

snap = snap_c["data"]
snap_bids = {p: q for p, q in snap["bids"]}
snap_asks = {p: q for p, q in snap["asks"]}

emit("")
emit("=" * 78)
emit("Self-check: does this duplicated machinery reproduce the committed run?")
emit("=" * 78)
checks = [
    ("diffs applied (control)", EXPECT_DIFFS, nd_c),
    ("diffs applied (repaired)", EXPECT_DIFFS, nd_r),
    ("crossed-book events (control)", EXPECT_CROSS_CONTROL, nx_c),
    ("crossed-book events (repaired)", EXPECT_CROSS_REPAIRED, nx_r),
    ("levels dropped (repaired)", EXPECT_DROPS, bk_r.n_drops),
    ("levels dropped (control)", 0, bk_c.n_drops),
    ("snapshot bids", 779, len(snap_bids)),
    ("snapshot asks", 1000, len(snap_asks)),
    ("missing bids (control)", EXPECT_MISSING_BIDS, missing_count(bk_c.st.bids.book, snap_bids)),
    ("missing bids (repaired)", EXPECT_MISSING_BIDS, missing_count(bk_r.st.bids.book, snap_bids)),
    ("missing asks (control)", EXPECT_MISSING_ASKS, missing_count(bk_c.st.asks.book, snap_asks)),
    ("missing asks (repaired)", EXPECT_MISSING_ASKS, missing_count(bk_r.st.asks.book, snap_asks)),
]
ok = True
for name, want, got in checks:
    good = (want == got)
    emit("  %-34s expected=%-10s got=%-10s %s"
         % (name, want, got, "OK" if good else "*** mismatch ***"))
    if not good:
        ok = False
if not ok:
    emit("")
    emit("self-check FAILED; the duplicated machinery has drifted from the")
    emit("committed run. Reporting no touch numbers. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)
emit("  all OK; the duplicate reproduces 3c87b1d exactly.")

true_bb = float(snap["bids"][0][0])
true_ba = float(snap["asks"][0][0])
true_mid = (true_bb + true_ba) / 2.0

emit("")
emit("=" * 78)
emit("The touch")
emit("=" * 78)
emit("  position of both books: last_update_id=%s" % snap_c["book_uid_before"])
emit("  snapshot lastUpdateId : %d   (invisible span of %d updates in between)"
     % (snap["lastUpdateId"], snap["lastUpdateId"] - snap_c["book_uid_before"]))
emit("  The snapshot's own touch was CREATED inside that span, so neither arm")
emit("  can be expected to hold it exactly. What matters is the distance.")
emit("")
emit("  Ground truth : best_bid=%.2f  best_ask=%.2f  mid=%.3f  spread=%.2f (%.1f ticks, %.4f bp)"
     % (true_bb, true_ba, true_mid, true_ba - true_bb,
        (true_ba - true_bb) / TICK, ((true_ba - true_bb) / true_mid) * 10000.0))

rows = []
for label, bk in (("CONTROL ", bk_c), ("REPAIRED", bk_r)):
    bb = bk.st.bids.best()
    ba = bk.st.asks.best()
    bbf = float(bb) if bb is not None else None
    baf = float(ba) if ba is not None else None
    mid = (bbf + baf) / 2.0 if (bbf is not None and baf is not None) else None
    crossed = (bbf is not None and baf is not None and bbf >= baf)
    emit("")
    emit("  %s : best_bid=%s  best_ask=%s  crossed=%s"
         % (label, ("%.2f" % bbf) if bbf is not None else "None",
            ("%.2f" % baf) if baf is not None else "None", crossed))
    if bbf is not None:
        d = bbf - true_bb
        emit("      bid error : %+.2f USD = %+.1f ticks = %+.4f bp"
             % (d, d / TICK, (d / true_mid) * 10000.0))
    if baf is not None:
        d = baf - true_ba
        emit("      ask error : %+.2f USD = %+.1f ticks = %+.4f bp"
             % (d, d / TICK, (d / true_mid) * 10000.0))
    if mid is not None:
        d = mid - true_mid
        emit("      mid error : %+.3f USD = %+.1f ticks = %+.4f bp   <== what volatility reads"
             % (d, d / TICK, (d / true_mid) * 10000.0))
        sp = baf - bbf
        emit("      spread    : %.2f USD = %.1f ticks = %.4f bp   <== what the spread series reads"
             % (sp, sp / TICK, (sp / mid) * 10000.0 if mid > 0 else float("nan")))
    rows.append((label, bbf, baf, mid, crossed))

emit("")
emit("  Difference between arms:")
_, bb_c, ba_c, mid_c, _ = rows[0]
_, bb_r, ba_r, mid_r, _ = rows[1]
if bb_c is not None and bb_r is not None:
    emit("    best_bid : control %.2f vs repaired %.2f  (difference %.2f USD)"
         % (bb_c, bb_r, bb_r - bb_c))
if ba_c is not None and ba_r is not None:
    emit("    best_ask : control %.2f vs repaired %.2f  (difference %.2f USD)"
         % (ba_c, ba_r, ba_r - ba_c))
if mid_c is not None and mid_r is not None:
    emit("    mid      : control %.3f vs repaired %.3f  (difference %.3f USD = %.1f bp)"
         % (mid_c, mid_r, mid_r - mid_c, ((mid_r - mid_c) / true_mid) * 10000.0))

emit("")
emit("=" * 78)
emit("Match rate by depth; is the error at the front or in the tail?")
emit("=" * 78)
emit("  Two measures per depth N:")
emit("    snap-side : of the SNAPSHOT's top N levels, how many does the book hold")
emit("                with the identical quantity string (the recall of truth)")
emit("    book-side : of the BOOK's top N levels, how many are in the snapshot's")
emit("                top N (catches spurious levels sitting at the front)")

for side_label, snap_side_list, is_bid in (("BIDS", snap["bids"], True),
                                           ("ASKS", snap["asks"], False)):
    emit("")
    emit("  %s" % side_label)
    emit("    %-9s %-28s %-28s" % ("depth", "CONTROL", "REPAIRED"))
    for N in (5, 10, 25):
        cells = []
        for bk in (bk_c, bk_r):
            book = bk.st.bids.book if is_bid else bk.st.asks.book
            topN_snap = snap_side_list[:N]
            hit = 0
            for p, q in topN_snap:
                if book.get(p) == q:
                    hit += 1
            mine = sorted_levels(book, is_bid)[:N]
            snap_topN_prices = set(p for p, _ in topN_snap)
            hit2 = sum(1 for p, _ in mine if p in snap_topN_prices)
            cells.append("snap %2d/%-2d   book %2d/%-2d" % (hit, N, hit2, N))
        emit("    top %-5d %-28s %-28s" % (N, cells[0], cells[1]))

emit("")
emit("=" * 78)
emit("The front of the book, side by side (top 5)")
emit("=" * 78)
for side_label, snap_side_list, is_bid in (("BIDS (desc)", snap["bids"], True),
                                           ("ASKS (asc)", snap["asks"], False)):
    emit("")
    emit("  %s" % side_label)
    emit("    %-5s %-26s %-26s %-26s" % ("rank", "SNAPSHOT (truth)", "CONTROL", "REPAIRED"))
    c_lv = sorted_levels(bk_c.st.bids.book if is_bid else bk_c.st.asks.book, is_bid)[:5]
    r_lv = sorted_levels(bk_r.st.bids.book if is_bid else bk_r.st.asks.book, is_bid)[:5]
    for i in range(5):
        s = snap_side_list[i] if i < len(snap_side_list) else ("-", "-")
        c = c_lv[i] if i < len(c_lv) else ("-", "-")
        r = r_lv[i] if i < len(r_lv) else ("-", "-")
        emit("    %-5d %-26s %-26s %-26s"
             % (i + 1, "%s @ %s" % (s[0], s[1]), "%s @ %s" % (c[0], c[1]),
                "%s @ %s" % (r[0], r[1])))

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print("")
print("Results written to " + RESULTS_PATH)
