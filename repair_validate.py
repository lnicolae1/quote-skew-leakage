# repair_validate.py: ground-truth validation of the crossing-triggered orphan repair

import json
import os
import sys

from compute_volatility import State, parse_ts

DATA_DIR = "market_data_droplet"

CHAIN_DAYS = ["08-15", "08-16", "08-17", "08-18", "08-19", "08-20", "08-21"]
TARGET_SNAPSHOT_UID = 4547092681

MAX_DROPS_PER_EVENT = 10000

RESULTS_PATH = "repair_validate_results.txt"

out_lines = []


def emit(s=""):
    print(s)
    out_lines.append(s)


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


class Book:
    """State plus the update-ID at which each resting price was last written"""

    def __init__(self, repair):
        self.st = State()
        self.wb = {}
        self.wa = {}
        self.repair = repair
        self.n_drops = 0
        self.n_repair_events = 0
        self.dropped_detail = []

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
                if len(self.dropped_detail) < 40:
                    self.dropped_detail.append(("bid", bb, ub, uid))
            else:
                self.st.asks.book.pop(ba, None)
                self.wa.pop(ba, None)
                if len(self.dropped_detail) < 40:
                    self.dropped_detail.append(("ask", ba, ua, uid))
            dropped_here += 1
            self.n_drops += 1
        if dropped_here:
            self.n_repair_events += 1


def run_until_target(repair):
    """Streams the chain and stops at the instant the target snapshot record is"""
    bk = Book(repair)
    target_snap = None
    n_cross_samples = 0
    n_diffs = 0

    for day in CHAIN_DAYS:
        p = path_for(day)
        if not os.path.exists(p):
            emit("  Missing file: " + p)
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
                        return bk, target_snap, n_diffs, n_cross_samples
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
                    n_cross_samples += 1
        emit("    chained %s  (last_update_id=%s, drops so far=%d)"
             % (day, bk.st.last_update_id, bk.n_drops))
        sys.stdout.flush()
    return bk, target_snap, n_diffs, n_cross_samples


def compare(bk, snap, label):
    """Level-for-level comparison at full string precision, respecting the"""
    emit("")
    emit("  --- %s ---" % label)

    snap_bids = {p: q for p, q in snap["bids"]}
    snap_asks = {p: q for p, q in snap["asks"]}

    worst_bid = min((float(p) for p in snap_bids), default=None)
    worst_ask = max((float(p) for p in snap_asks), default=None)

    for side_label, snap_side, my_book, worst, is_bid in (
            ("BIDS", snap_bids, bk.st.bids.book, worst_bid, True),
            ("ASKS", snap_asks, bk.st.asks.book, worst_ask, False)):
        match = 0
        qty_mismatch = []
        missing = []
        for p, q in snap_side.items():
            mine = my_book.get(p)
            if mine is None:
                missing.append((p, q))
            elif mine == q:
                match += 1
            else:
                qty_mismatch.append((p, q, mine))

        extra = []
        beyond_cap = 0
        for p in my_book:
            if p in snap_side:
                continue
            pf = float(p)
            inside = (pf >= worst) if is_bid else (pf <= worst)
            if inside:
                extra.append((p, my_book[p]))
            else:
                beyond_cap += 1

        total = len(snap_side)
        emit("    %s: %d/%d exact matches (price and quantity string)"
             % (side_label, match, total))
        emit("        quantity mismatches: %d   missing from my book: %d"
             % (len(qty_mismatch), len(missing)))
        emit("        extra levels inside snapshot range: %d" % len(extra))
        emit("        my levels beyond the snapshot's range (beyond cap, NOT a "
             "mismatch): %d" % beyond_cap)
        for p, q, mine in qty_mismatch[:8]:
            emit("          qty mismatch price=%s snapshot=%s mine=%s" % (p, q, mine))
        for p, q in missing[:8]:
            emit("          missing price=%s snapshot_qty=%s" % (p, q))
        for p, q in extra[:8]:
            emit("          extra   price=%s my_qty=%s" % (p, q))

    return None


emit("=" * 78)
emit("Ground-truth validation of the crossing-triggered orphan repair")
emit("=" * 78)
emit("Chain: " + " -> ".join(CHAIN_DAYS) + "   (droplet only)")
emit("Target: the independently-fetched REST snapshot at 08-21T00:02:11,")
emit("        lastUpdateId=%d. The book is compared against it before it is" % TARGET_SNAPSHOT_UID)
emit("        applied, so the snapshot is genuine external ground truth.")
emit("Rule under test: while best_bid >= best_ask, drop whichever touch level")
emit("        was written at the lower update-ID. No time constant.")

emit("")
emit("Run 1 of 2: CONTROL (no repair; what the committed pipeline does today)")
bk_c, snap_c, nd_c, nx_c = run_until_target(repair=False)

emit("")
emit("run 2 of 2: REPAIRED (crossing-triggered rule)")
bk_r, snap_r, nd_r, nx_r = run_until_target(repair=True)

if snap_c is None or snap_r is None:
    emit("")
    emit("target snapshot uid=%d was not found in the chain. Cannot validate."
         % TARGET_SNAPSHOT_UID)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)

emit("")
emit("=" * 78)
emit("Position at the comparison point")
emit("=" * 78)
snap = snap_c["data"]
emit("  snapshot timestamp   : %s" % snap_c["ts"])
emit("  snapshot lastUpdateId: %d" % snap["lastUpdateId"])
emit("  snapshot levels      : %d bids, %d asks" % (len(snap["bids"]), len(snap["asks"])))
emit("  CONTROL  book last_update_id before the snapshot: %s" % snap_c["book_uid_before"])
emit("  REPAIRED book last_update_id before the snapshot: %s" % snap_r["book_uid_before"])
gap_c = snap["lastUpdateId"] - (snap_c["book_uid_before"] or 0)
emit("  invisible span between my position and the snapshot: %d updates" % gap_c)
if gap_c > 0:
    emit("  ^ these updates were never delivered. An exact match is therefore not")
    emit("    expected: any level those %d updates changed will legitimately differ." % gap_c)
    emit("    The comparison is still decisive because the CONTROL and the REPAIRED")
    emit("    run face the identical handicap; only the rule differs between them.")

emit("")
emit("  CONTROL : diffs applied=%d, crossed-book events=%d (%.2f%%)"
     % (nd_c, nx_c, 100.0 * nx_c / nd_c if nd_c else 0))
emit("  REPAIRED: diffs applied=%d, crossed-book events=%d (%.2f%%), levels dropped=%d "
     "across %d repair events"
     % (nd_r, nx_r, 100.0 * nx_r / nd_r if nd_r else 0, bk_r.n_drops, bk_r.n_repair_events))
if bk_r.dropped_detail:
    emit("  first levels the rule dropped (side, price, uid_written, uid_now):")
    for d in bk_r.dropped_detail[:12]:
        emit("      %s %s written_at=%s dropped_at=%s" % d)

emit("")
emit("=" * 78)
emit("Level-for-level comparison against the 08-21 SNAPSHOT")
emit("=" * 78)
compare(bk_c, snap, "CONTROL (no repair)")
compare(bk_r, snap, "REPAIRED (crossing-triggered rule)")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print("")
print("Results written to " + RESULTS_PATH)
