# diagnose_ghost_0817.py: read-only diagnostic: is the 08-17..08-22 crossed-book stretch repairable?

import json
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.getcwd())

from compute_volatility import Side, State, parse_ts

DATA_DIR = "market_data_droplet"
SCRATCH = os.path.dirname(os.path.abspath(__file__))

STALE_SECONDS = 300
SPREAD_SAMPLE_STEP = 10


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


class Tracked:
    """State plus a per-price last-touch record"""

    def __init__(self):
        self.st = State()
        self.touch_bid = {}
        self.touch_ask = {}

    def apply_snapshot(self, snap, ts):
        uid = snap["lastUpdateId"]
        self.st.bids.load_snapshot(snap["bids"])
        self.st.asks.load_snapshot(snap["asks"])
        self.touch_bid = {p: (uid, ts) for p, _ in snap["bids"]}
        self.touch_ask = {p: (uid, ts) for p, _ in snap["asks"]}
        self.st.last_update_id = uid
        self.st.last_ts = ts

    def apply_diff(self, data, ts):
        uid = data["u"]
        self.st.bids.apply(data["b"])
        self.st.asks.apply(data["a"])
        for p, q in data["b"]:
            if float(q) == 0.0:
                self.touch_bid.pop(p, None)
            else:
                self.touch_bid[p] = (uid, ts)
        for p, q in data["a"]:
            if float(q) == 0.0:
                self.touch_ask.pop(p, None)
            else:
                self.touch_ask[p] = (uid, ts)
        self.st.last_update_id = uid
        self.st.last_ts = ts

    def best(self):
        bb = self.st.bids.best()
        ba = self.st.asks.best()
        return bb, ba

    def true_touch(self, now_ts):
        """Best bid / best ask after excluding levels not touched for"""
        now = parse_ts(now_ts)
        cutoff = now - timedelta(seconds=STALE_SECONDS)
        bb = None
        for p in self.st.bids.book:
            t = self.touch_bid.get(p)
            if t is None or parse_ts(t[1]) < cutoff:
                continue
            pf = float(p)
            if bb is None or pf > bb:
                bb = pf
        ba = None
        for p in self.st.asks.book:
            t = self.touch_ask.get(p)
            if t is None or parse_ts(t[1]) < cutoff:
                continue
            pf = float(p)
            if ba is None or pf < ba:
                ba = pf
        return bb, ba

    def stale_at_touch(self, now_ts):
        """Which side's touch is stale, and by how long"""
        now = parse_ts(now_ts)
        bb, ba = self.best()
        out = {}
        for label, price, tmap in (("bid", bb, self.touch_bid),
                                   ("ask", ba, self.touch_ask)):
            if price is None:
                out[label] = None
                continue
            t = tmap.get(price)
            if t is None:
                out[label] = (price, None, None)
            else:
                out[label] = (price, t[0], (now - parse_ts(t[1])).total_seconds())
        return out


def stream(tr, path, on_event=None, stop_after_crossings=None):
    """Copy of the committed per-line control flow (snapshot -> load, diff ->"""
    gaps = []
    snaps = []
    n_cross = 0
    n_diff = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' in line:
                continue
            rec = json.loads(line)
            rtype = rec.get("type")
            ts = rec.get("local_timestamp")

            if rtype == "snapshot":
                snap = rec["data"]
                uid = snap["lastUpdateId"]
                prev = tr.st.last_update_id
                span = None
                if prev is not None and uid > prev:
                    span = (prev, uid, uid - prev)
                    gaps.append({"kind": "invisible_span", "start_ts": tr.st.last_ts,
                                 "end_ts": ts, "lost": uid - prev})
                bb0, ba0 = tr.best()
                tr.apply_snapshot(snap, ts)
                bb1, ba1 = tr.best()
                snaps.append({"ts": ts, "uid": uid, "prev_uid": prev, "span": span,
                              "before": (bb0, ba0), "after": (bb1, ba1)})
                continue

            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if tr.st.last_update_id is None or data["u"] <= tr.st.last_update_id:
                continue
            if data["U"] != tr.st.last_update_id + 1:
                gaps.append({"kind": "resumption", "start_ts": tr.st.last_ts,
                             "end_ts": ts, "lost": data["U"] - tr.st.last_update_id - 1})
            prev_uid = tr.st.last_update_id
            tr.apply_diff(data, ts)
            n_diff += 1
            bb, ba = tr.best()
            if bb is not None and ba is not None and float(bb) >= float(ba):
                n_cross += 1
                if on_event is not None:
                    on_event(n_cross, rec, data, prev_uid, bb, ba, tr)
                if stop_after_crossings is not None and n_cross >= stop_after_crossings:
                    return gaps, snaps, n_diff, n_cross
    return gaps, snaps, n_diff, n_cross


def hdr(s):
    print("")
    print("=" * 78)
    print(s)
    print("=" * 78)


hdr("Part 1/2: find the ghost; chain 08-15 (own snapshot anchor) -> 08-16 -> 08-17")

tr = Tracked()

for day in ("08-15", "08-16"):
    g, s, nd, nc = stream(tr, path_for(day))
    bb, ba = tr.best()
    print("  chained %s: diffs=%d snapshots=%d gaps=%d crossed=%d  end bb=%s ba=%s crossed_now=%s"
          % (day, nd, len(s), len(g), nc, bb, ba,
             (bb is not None and ba is not None and float(bb) >= float(ba))))
    for sn in s:
        print("      snapshot ts=%s uid=%s prev_uid=%s invisible_span=%s"
              % (sn["ts"], sn["uid"], sn["prev_uid"], sn["span"]))

bb, ba = tr.best()
print("")
print("  Book entering 08-17: best_bid=%s best_ask=%s crossed=%s"
      % (bb, ba, (bb is not None and ba is not None and float(bb) >= float(ba))))

first = {}


def on_cross(n, rec, data, prev_uid, bb, ba, tr):
    if n != 1:
        return
    first["ts"] = rec["local_timestamp"]
    first["U"] = data["U"]
    first["u"] = data["u"]
    first["prev_uid"] = prev_uid
    first["bb"] = bb
    first["ba"] = ba
    first["b_updates"] = data["b"]
    first["a_updates"] = data["a"]
    first["top_bids"] = sorted(tr.st.bids.book.items(), key=lambda kv: -float(kv[0]))[:6]
    first["top_asks"] = sorted(tr.st.asks.book.items(), key=lambda kv: float(kv[0]))[:6]
    first["stale"] = tr.stale_at_touch(rec["local_timestamp"])
    first["true_touch"] = tr.true_touch(rec["local_timestamp"])


g17, s17, nd17, nc17 = stream(tr, path_for("08-17"), on_event=on_cross)

print("")
print("  08-17: diffs=%d snapshots=%d gaps=%d crossed_instances=%d" % (nd17, len(s17), len(g17), nc17))
for sn in s17:
    print("    SNAPSHOT ts=%s uid=%s prev_uid=%s invisible_span=%s"
          % (sn["ts"], sn["uid"], sn["prev_uid"], sn["span"]))
    print("        book before snapshot: bb=%s ba=%s" % sn["before"])
    print("        book after  snapshot: bb=%s ba=%s" % sn["after"])
for gg in g17:
    print("    gap %s: %s -> %s (%d updates lost)"
          % (gg["kind"], gg["start_ts"], gg["end_ts"], gg["lost"]))

if first:
    print("")
    print("  *** first crossing in 08-17 ***")
    print("    local_timestamp=%s  u=%s  U=%s  prev_last_update_id=%s  contiguous(U==prev+1)? %s"
          % (first["ts"], first["u"], first["U"], first["prev_uid"],
             first["U"] == first["prev_uid"] + 1))
    print("    best_bid=%s  best_ask=%s  bb-ba=%.8f"
          % (first["bb"], first["ba"], float(first["bb"]) - float(first["ba"])))
    print("    this diff's raw bid updates: %s" % (first["b_updates"],))
    print("    this diff's raw ask updates: %s" % (first["a_updates"],))
    print("    Top 6 bids (desc): %s" % (first["top_bids"],))
    print("    Top 6 asks (asc):  %s" % (first["top_asks"],))
    print("    staleness at the touch (price, last_touch_uid, seconds_since_touch):")
    print("      bid: %s" % (first["stale"]["bid"],))
    print("      ask: %s" % (first["stale"]["ask"],))
    print("    TRUE TOUCH excluding levels stale > %ds: bb=%s ba=%s"
          % (STALE_SECONDS, first["true_touch"][0], first["true_touch"][1]))
else:
    print("  no crossing found in 08-17 from this anchor.")

print("")
print("  End of 08-17: top levels with age since last touch")
now_ts = tr.st.last_ts
now = parse_ts(now_ts)
for label, book, tmap, rev in (("BIDS", tr.st.bids.book, tr.touch_bid, True),
                               ("ASKS", tr.st.asks.book, tr.touch_ask, False)):
    lv = sorted(book.items(), key=lambda kv: (-float(kv[0]) if rev else float(kv[0])))[:8]
    print("    %s (best first):" % label)
    for p, q in lv:
        t = tmap.get(p)
        age = (now - parse_ts(t[1])).total_seconds() if t else None
        print("      price=%s qty=%s last_touch_uid=%s age=%s s"
              % (p, q, t[0] if t else None, ("%.1f" % age) if age is not None else "?"))
bbt, bat = tr.true_touch(now_ts)
bb, ba = tr.best()
print("    Raw touch : bb=%s ba=%s" % (bb, ba))
print("    TRUE touch (stale>%ds dropped): bb=%s ba=%s" % (STALE_SECONDS, bbt, bat))

ghosts = []
if bbt is not None and bat is not None:
    for p in tr.st.asks.book:
        t = tr.touch_ask.get(p)
        if t and (now - parse_ts(t[1])).total_seconds() > STALE_SECONDS and float(p) < bat:
            ghosts.append(("ask", p, tr.st.asks.book[p], t))
    for p in tr.st.bids.book:
        t = tr.touch_bid.get(p)
        if t and (now - parse_ts(t[1])).total_seconds() > STALE_SECONDS and float(p) > bbt:
            ghosts.append(("bid", p, tr.st.bids.book[p], t))

print("")
print("  Ghost candidates (stale > %ds and sitting inside the true touch): %d"
      % (STALE_SECONDS, len(ghosts)))
for side, p, q, t in sorted(ghosts, key=lambda x: float(x[1]))[:40]:
    print("    side=%s price=%s qty=%s entered_or_last_touched_uid=%s ts=%s"
          % (side, p, q, t[0], t[1]))

sys.stdout.flush()

with open(os.path.join(SCRATCH, "ghost_target.txt"), "w", encoding="utf-8") as f:
    for side, p, q, t in ghosts:
        f.write(side + "\t" + p + "\t" + q + "\t" + str(t[0]) + "\t" + t[1] + "\n")
print("")
print("  (ghost candidate list written to ghost_target.txt for the next stage)")
