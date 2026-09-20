# split_0821.py: splits 2026-08-21 at its re-anchor

import json
import math
import os
import sys
from datetime import timedelta

from compute_volatility import State, parse_ts, SECONDS_PER_YEAR
from compute_volatility_clean import CrossAwareGridBuilder

DATA_DIR = "market_data_droplet"
RESULTS_PATH = "split_0821_results.txt"

CHAIN = ["08-15", "08-16", "08-17", "08-18", "08-19", "08-20"]
TARGET_DAY = "08-21"
ANCHOR_UID = 4547092681

VOL_STEP = 300
MAX_DROPS_PER_EVENT = 10000

EXPECT_VOL = 0.8657
EXPECT_VOL_N = 286
EXPECT_GRID_S = 86400
EXPECT_CROSSED_S = 0
EXPECT_DROPS_DAY = 4
CLEAN_MIN, CLEAN_MAX = 0.0770, 0.5418

out_lines = []


def emit(s=""):
    print(s)
    out_lines.append(s)


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


class Chain:
    def __init__(self, repair):
        self.st = State()
        self.wb = {}
        self.wa = {}
        self.repair = repair
        self.n_drops = 0

    def load_snapshot(self, snap):
        uid = snap["lastUpdateId"]
        self.st.bids.load_snapshot(snap["bids"])
        self.st.asks.load_snapshot(snap["asks"])
        self.wb = {p: uid for p, _ in snap["bids"]}
        self.wa = {p: uid for p, _ in snap["asks"]}
        self.st.last_update_id = uid

    def apply(self, data):
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
            self.uncross()

    def uncross(self):
        n = 0
        while n < MAX_DROPS_PER_EVENT:
            bb = self.st.bids.best()
            ba = self.st.asks.best()
            if bb is None or ba is None:
                return
            if float(bb) < float(ba):
                return
            if self.wb.get(bb, -1) <= self.wa.get(ba, -1):
                self.st.bids.book.pop(bb, None)
                self.wb.pop(bb, None)
            else:
                self.st.asks.book.pop(ba, None)
                self.wa.pop(ba, None)
            n += 1
            self.n_drops += 1


def stream(ch, path, gaps, on_event, seed_only_from_uid=None):
    """One pass over a file"""
    trade_ts = []
    started = seed_only_from_uid is None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' in line:
                if not started:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                d = rec.get("data")
                if d is not None and d.get("e") == "aggTrade" and d.get("a") is not None:
                    trade_ts.append((rec["local_timestamp"], d["a"]))
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            rt = rec.get("type")
            ts = rec.get("local_timestamp")

            if rt == "snapshot":
                snap = rec["data"]
                uid = snap["lastUpdateId"]
                if not started:
                    if uid != seed_only_from_uid:
                        continue
                    started = True
                    ch.load_snapshot(snap)
                    ch.st.last_ts = ts
                    bb, ba = ch.st.bids.best(), ch.st.asks.best()
                    if bb is not None and ba is not None:
                        on_event(ts, float(bb), float(ba), uid)
                    continue
                if ch.st.last_update_id is not None and uid > ch.st.last_update_id:
                    gaps.append({"start_ts": ch.st.last_ts, "end_ts": ts,
                                 "lost": uid - ch.st.last_update_id})
                ch.load_snapshot(snap)
                ch.st.last_ts = ts
                bb, ba = ch.st.bids.best(), ch.st.asks.best()
                if bb is not None and ba is not None:
                    on_event(ts, float(bb), float(ba), uid)
                continue

            if rt != "diff" or not started:
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if ch.st.last_update_id is None or data["u"] <= ch.st.last_update_id:
                continue
            if data["U"] != ch.st.last_update_id + 1:
                gaps.append({"start_ts": ch.st.last_ts, "end_ts": ts,
                             "lost": data["U"] - ch.st.last_update_id - 1})
            ch.apply(data)
            ch.st.last_ts = ts
            bb, ba = ch.st.bids.best(), ch.st.asks.best()
            if bb is not None and ba is not None:
                on_event(ts, float(bb), float(ba), data["u"])
    return trade_ts


def vol_from_grid(grid, gaps, label, split_at=None, side=None):
    """The committed clean estimator"""
    grid_start, mids, crossed = grid.finalize()
    gap_dt = [(parse_ts(g["start_ts"]), parse_ts(g["end_ts"]))
              for g in gaps if g["start_ts"] and g["end_ts"]]
    sub_m = mids[::VOL_STEP]
    sub_c = crossed[::VOL_STEP]
    rets = []
    n_excl = 0
    n_side = 0
    for i in range(len(sub_m) - 1):
        t_i = grid_start + timedelta(seconds=i * VOL_STEP)
        t_j = grid_start + timedelta(seconds=(i + 1) * VOL_STEP)
        if split_at is not None:
            if side == "pre" and t_j > split_at:
                continue
            if side == "post" and t_i < split_at:
                continue
        n_side += 1
        if any(gs < t_j and ge > t_i for gs, ge in gap_dt):
            n_excl += 1
            continue
        if sub_c[i] or sub_c[i + 1]:
            n_excl += 1
            continue
        p0, p1 = sub_m[i], sub_m[i + 1]
        if p0 <= 0 or p1 <= 0:
            n_excl += 1
            continue
        rets.append(math.log(p1 / p0))
    if len(rets) < 2:
        return {"label": label, "n": len(rets), "n_excl": n_excl,
                "n_candidate": n_side, "ann": None,
                "grid_start": grid_start, "grid_len": len(mids),
                "crossed_s": sum(1 for c in crossed if c)}
    n = len(rets)
    mu = sum(rets) / n
    var = sum((r - mu) ** 2 for r in rets) / n
    sd = math.sqrt(var)
    return {"label": label, "n": n, "n_excl": n_excl, "n_candidate": n_side,
            "ann": sd * math.sqrt(SECONDS_PER_YEAR / VOL_STEP),
            "grid_start": grid_start, "grid_len": len(mids),
            "crossed_s": sum(1 for c in crossed if c)}


emit("=" * 78)
emit("split_0821.py; splitting 2026-08-21 at its re-anchor")
emit("=" * 78)
emit("Anchor: in-file REST snapshot, lastUpdateId=%d, 779 bids / 1000 asks." % ANCHOR_UID)
emit("Estimator: 5-min sampling of a 1s previous-tick forward-filled mid grid,")
emit("           gap-straddle and crossed-book excluded. The committed one.")
emit("40-day clean volatility range for reference: %.2f%% .. %.2f%%"
     % (CLEAN_MIN * 100, CLEAN_MAX * 100))

emit("")
emit("=" * 78)
emit("ARM A: chained  (08-15 snapshot -> 08-16 .. 08-20 -> 08-21, repair on)")
emit("=" * 78)
chA = Chain(repair=True)
for d in CHAIN:
    g = []
    stream(chA, path_for(d), g, lambda *a: None)
    emit("  chained %s  last_update_id=%s  cumulative drops=%d"
         % (d, chA.st.last_update_id, chA.n_drops))
    sys.stdout.flush()

drops_before = chA.n_drops
gapsA = []
gridA = CrossAwareGridBuilder()
anchor_ts_holder = {}


def on_event_A(ts, bb, ba, uid):
    if uid == ANCHOR_UID and "ts" not in anchor_ts_holder:
        anchor_ts_holder["ts"] = ts
    gridA.on_event(ts, uid, (bb + ba) / 2.0, bb >= ba)


tradesA = stream(chA, path_for(TARGET_DAY), gapsA, on_event_A)
drops_day = chA.n_drops - drops_before
fullA = vol_from_grid(gridA, gapsA, "full day")

emit("")
emit("  full day: grid_s=%d crossed_s=%d returns=%d excluded=%d vol=%s"
     % (fullA["grid_len"], fullA["crossed_s"], fullA["n"], fullA["n_excl"],
        ("%.2f%%" % (fullA["ann"] * 100)) if fullA["ann"] is not None else "n/a"))
emit("  drops on 08-21: %d" % drops_day)

emit("")
emit("  self-check against regime_recover_results.txt (41ffc03):")
checks = [("annualised vol", EXPECT_VOL, round(fullA["ann"], 4) if fullA["ann"] else None),
          ("returns kept", EXPECT_VOL_N, fullA["n"]),
          ("grid seconds", EXPECT_GRID_S, fullA["grid_len"]),
          ("crossed seconds", EXPECT_CROSSED_S, fullA["crossed_s"]),
          ("drops on the day", EXPECT_DROPS_DAY, drops_day)]
ok = True
for name, want, got in checks:
    good = (want == got)
    emit("    %-20s committed=%-10s got=%-10s %s"
         % (name, want, got, "OK" if good else "*** mismatch ***"))
    if not good:
        ok = False
if not ok:
    emit("")
    emit("  self-check FAILED; this script does not reproduce the committed day.")
    emit("  Reporting nothing further. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)
emit("    all OK; ARM A reproduces the committed 08-21 exactly.")

anchor_dt = parse_ts(anchor_ts_holder["ts"])
day_start = fullA["grid_start"]
pre_s = (anchor_dt - day_start).total_seconds()
post_s = fullA["grid_len"] - pre_s

emit("")
emit("=" * 78)
emit("The naive split, and why it has no power")
emit("=" * 78)
emit("  grid starts   : %s" % day_start.isoformat())
emit("  anchor lands  : %s" % anchor_dt.isoformat())
emit("  PRE-ANCHOR  segment: %8.1f s  = %5.2f%% of the day" % (pre_s, 100.0 * pre_s / fullA["grid_len"]))
emit("  post-anchor segment: %8.1f s  = %5.2f%% of the day" % (post_s, 100.0 * post_s / fullA["grid_len"]))
emit("  A 5-minute return needs >= 301 s of span.")

pre = vol_from_grid(gridA, gapsA, "pre-anchor", split_at=anchor_dt, side="pre")
post = vol_from_grid(gridA, gapsA, "post-anchor", split_at=anchor_dt, side="post")

emit("")
emit("  %-14s %-12s %-10s %-10s %-10s" % ("segment", "candidates", "kept", "excluded", "vol"))
for seg in (pre, post, fullA):
    emit("  %-14s %-12d %-10d %-10d %-10s"
         % (seg["label"], seg["n_candidate"], seg["n"], seg["n_excl"],
            ("%.2f%%" % (seg["ann"] * 100)) if seg["ann"] is not None else "n/a"))

emit("")
if pre["n"] < 2:
    emit("  >>> PRE-ANCHOR has %d USABLE 5-minute returns. The split cannot separate" % pre["n"])
    emit("      the two halves, and no volatility is quoted for it. The anchor lands")
    emit("      %.0f s into the day, far short of the 301 s one return requires." % pre_s)
    emit("")
    emit("  >>> But THE SAME arithmetic answers the question. %.2f%% of the day is"
         % (100.0 * post_s / fullA["grid_len"]))
    emit("      already post-anchor, so the committed 86.57% is, to within rounding,")
    emit("      a post-anchor measurement. 'Chaining artifact' cannot be what makes")
    emit("      it extreme; the chain governs only the first %.0f seconds." % pre_s)
    emit("      ARM B below tests that directly instead of by splitting.")
else:
    emit("  Pre-anchor carries %d usable returns; both segments are quotable." % pre["n"])

emit("")
emit("=" * 78)
emit("ARM B: fresh  (08-21 alone, book seeded only by the 00:02:11 snapshot)")
emit("=" * 78)
emit("  No six-day chain, no inherited state, no carried orphans. Every record")
emit("  before the snapshot is ignored. This is the day as the exchange's own")
emit("  book describes it.")
chB = Chain(repair=True)
gapsB = []
gridB = CrossAwareGridBuilder()
tradesB = stream(chB, path_for(TARGET_DAY), gapsB,
                 lambda ts, bb, ba, uid: gridB.on_event(ts, uid, (bb + ba) / 2.0, bb >= ba),
                 seed_only_from_uid=ANCHOR_UID)
fullB = vol_from_grid(gridB, gapsB, "ARM B full")
emit("")
emit("  grid_s=%d  crossed_s=%d  returns=%d  excluded=%d  drops=%d"
     % (fullB["grid_len"], fullB["crossed_s"], fullB["n"], fullB["n_excl"], chB.n_drops))
emit("  annualised vol: %s"
     % (("%.2f%%" % (fullB["ann"] * 100)) if fullB["ann"] is not None else "n/a"))

emit("")
emit("=" * 78)
emit("Lambda control; the defect cannot touch the trade stream")
emit("=" * 78)
pre_ids = set()
post_ids = set()
first_t = last_t = None
for ts, aid in tradesA:
    t = parse_ts(ts)
    if first_t is None:
        first_t = t
    last_t = t
    (pre_ids if t < anchor_dt else post_ids).add(aid)
pre_span = (anchor_dt - first_t).total_seconds() if first_t else 0.0
post_span = (last_t - anchor_dt).total_seconds() if last_t else 0.0
emit("  %-14s %-10s %-12s %-12s" % ("segment", "trades", "span (s)", "lambda"))
emit("  %-14s %-10d %-12.1f %-12s"
     % ("pre-anchor", len(pre_ids), pre_span,
        ("%.5f" % (len(pre_ids) / pre_span)) if pre_span > 0 else "n/a"))
emit("  %-14s %-10d %-12.1f %-12s"
     % ("post-anchor", len(post_ids), post_span,
        ("%.5f" % (len(post_ids) / post_span)) if post_span > 0 else "n/a"))
emit("  %-14s %-10d %-12.1f %-12.5f"
     % ("full day", len(pre_ids | post_ids),
        (last_t - first_t).total_seconds(),
        len(pre_ids | post_ids) / (last_t - first_t).total_seconds()))
emit("")
emit("  Trade counts are an independent readout of how busy each segment was.")
emit("  They neither confirm nor refute the volatility; they say whether the two")
emit("  parts of the day differ in activity at all.")

emit("")
emit("=" * 78)
emit("The comparison that decides it")
emit("=" * 78)
emit("  ARM A  chained through six days, repair on : %s"
      % (("%.2f%%" % (fullA["ann"] * 100)) if fullA["ann"] is not None else "n/a"))
emit("  ARM B  fresh from the exchange's snapshot  : %s"
      % (("%.2f%%" % (fullB["ann"] * 100)) if fullB["ann"] is not None else "n/a"))
if fullA["ann"] and fullB["ann"]:
    d = fullB["ann"] - fullA["ann"]
    emit("  difference: %+.2f pp  (%+.2f%% relative)"
         % (d * 100, 100.0 * d / fullA["ann"]))
    emit("  40-day clean maximum: %.2f%%" % (CLEAN_MAX * 100))
    emit("")
    if fullB["ann"] > CLEAN_MAX:
        emit("  ARM B is still far ABOVE THE clean maximum. The extremity does not come")
        emit("  from the six-day chain: it survives rebuilding the day from the")
        emit("  exchange's own book. On the fixed reading, 08-21 is genuinely violent")
        emit("  and the recovery is sound on it.")
    else:
        emit("  ARM B falls back into the clean range. The extremity was the chain, and")
        emit("  08-21 should be reported post-anchor only.")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print("")
print("Results written to " + RESULTS_PATH)
