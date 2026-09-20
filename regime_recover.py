# regime_recover.py: recovers 2026-08-17..08-22 and adds them back to the regime analysis

import json
import math
import os
import sys
from datetime import date, timedelta

from compute_volatility import State, parse_ts, SECONDS_PER_YEAR
from compute_volatility_clean import CrossAwareGridBuilder
from spread_depth_profile import percentile

DATA_DIR = "market_data_droplet"
CACHE_PATH = "regime_scan_rows.json"
RESULTS_PATH = "regime_recover_results.txt"

WARMUP = ["08-15", "08-16"]
RECOVER = ["08-17", "08-18", "08-19", "08-20", "08-21", "08-22"]

VOL_STEP_SECONDS = 300
STALE_SECONDS = 300
GAP_ADJACENCY_N = [0, 30, 120]

MAX_DROPS_PER_EVENT = 10000

CLEAN_STALE_LO, CLEAN_STALE_HI = 0.75, 2.70

COMMITTED = {
    "08-17": (195603, 2328, 1033667, 86399, 85921, 1, 286, 468, 2, 143, 1),
    "08-18": (168832, 1552, 824593, 86400, 86400, 0, 287, 0, 0, 0, 0),
    "08-19": (337017, 7393, 2324599, 86399, 86399, 0, 287, 0, 0, 0, 0),
    "08-20": (446613, 8983, 2833630, 86400, 86400, 0, 287, 0, 0, 0, 0),
    "08-21": (547143, 11930, 4005932, 86400, 86385, 0, 287, 14, 2, 453, 1),
    "08-22": (377878, 6506, 2259848, 86400, 86400, 0, 287, 0, 0, 0, 0),
}

out_lines = []


def emit(s=""):
    print(s)
    out_lines.append(s)


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


class Chain:
    """State plus the update-ID and timestamp at which each resting price was"""

    def __init__(self, repair):
        self.st = State()
        self.wb_uid = {}
        self.wa_uid = {}
        self.wb_t = {}
        self.wa_t = {}
        self.repair = repair
        self.n_drops = 0

    def load_snapshot(self, snap, tf):
        uid = snap["lastUpdateId"]
        self.st.bids.load_snapshot(snap["bids"])
        self.st.asks.load_snapshot(snap["asks"])
        self.wb_uid = {p: uid for p, _ in snap["bids"]}
        self.wa_uid = {p: uid for p, _ in snap["asks"]}
        self.wb_t = {p: tf for p, _ in snap["bids"]}
        self.wa_t = {p: tf for p, _ in snap["asks"]}
        self.st.last_update_id = uid

    def apply(self, data, tf):
        uid = data["u"]
        self.st.bids.apply(data["b"])
        self.st.asks.apply(data["a"])
        for p, q in data["b"]:
            if float(q) == 0.0:
                self.wb_uid.pop(p, None)
                self.wb_t.pop(p, None)
            else:
                self.wb_uid[p] = uid
                self.wb_t[p] = tf
        for p, q in data["a"]:
            if float(q) == 0.0:
                self.wa_uid.pop(p, None)
                self.wa_t.pop(p, None)
            else:
                self.wa_uid[p] = uid
                self.wa_t[p] = tf
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
            ub = self.wb_uid.get(bb, -1)
            ua = self.wa_uid.get(ba, -1)
            if ub <= ua:
                self.st.bids.book.pop(bb, None)
                self.wb_uid.pop(bb, None)
                self.wb_t.pop(bb, None)
            else:
                self.st.asks.book.pop(ba, None)
                self.wa_uid.pop(ba, None)
                self.wa_t.pop(ba, None)
            n += 1
            self.n_drops += 1


def stream_day(ch, path, gaps, on_event):
    """Duplicate of regime_scan.py's _stream_day, with the book mutation routed"""
    n_lines = n_snapshot = n_diff = n_trade_lines = 0
    trade_ids = set()
    first_ts = last_ts = None
    first_uid = last_uid = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1

            if '"aggTrade"' in line:
                n_trade_lines += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                ts_str = rec.get("local_timestamp")
                if ts_str is not None:
                    if first_ts is None:
                        first_ts = ts_str
                    last_ts = ts_str
                data = rec.get("data")
                if data is not None and data.get("e") == "aggTrade":
                    aid = data.get("a")
                    if aid is not None:
                        trade_ids.add(aid)
                continue

            try:
                rec = json.loads(line)
            except ValueError:
                continue
            rtype = rec.get("type")
            rec_ts = rec.get("local_timestamp")
            if rec_ts is not None:
                if first_ts is None:
                    first_ts = rec_ts
                last_ts = rec_ts

            if rtype == "snapshot":
                snap = rec["data"]
                snap_uid = snap["lastUpdateId"]
                if ch.st.last_update_id is not None and snap_uid > ch.st.last_update_id:
                    gaps.append({"kind": "invisible_span", "start_ts": ch.st.last_ts,
                                 "end_ts": rec_ts, "updates_lost": snap_uid - ch.st.last_update_id})
                elif ch.st.last_update_id is not None and snap_uid < ch.st.last_update_id:
                    raise RuntimeError("Backward anchor jump in " + path)
                tf = parse_ts(rec_ts).timestamp()
                ch.load_snapshot(snap, tf)
                ch.st.last_ts = rec_ts
                n_snapshot += 1
                if first_uid is None:
                    first_uid = snap_uid
                last_uid = snap_uid
                bb, ba = ch.st.bids.best(), ch.st.asks.best()
                if bb is not None and ba is not None:
                    on_event(rec_ts, tf, ch, float(bb), float(ba))
                continue

            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if ch.st.last_update_id is None or data["u"] <= ch.st.last_update_id:
                continue
            if data["U"] != ch.st.last_update_id + 1:
                gaps.append({"kind": "resumption", "start_ts": ch.st.last_ts,
                             "end_ts": rec_ts,
                             "updates_lost": data["U"] - ch.st.last_update_id - 1})
            tf = parse_ts(rec_ts).timestamp()
            ch.apply(data, tf)
            ch.st.last_ts = rec_ts
            n_diff += 1
            if first_uid is None:
                first_uid = data["u"]
            last_uid = data["u"]
            bb, ba = ch.st.bids.best(), ch.st.asks.best()
            if bb is not None and ba is not None:
                on_event(rec_ts, tf, ch, float(bb), float(ba))

    return {"lines": n_lines, "snapshots": n_snapshot, "diffs": n_diff,
            "trade_lines": n_trade_lines, "unique_trades": len(trade_ids),
            "first_ts": first_ts, "last_ts": last_ts,
            "first_uid": first_uid, "last_uid": last_uid}


def measure_day(ch, day):
    """Duplicate of regime_scan.py's measure_day volatility/spread block, plus"""
    gaps = []
    vol_grid = CrossAwareGridBuilder()
    spr_grid = CrossAwareGridBuilder()
    stale_grid = CrossAwareGridBuilder()

    def on_event(ts_str, tf, ch, bb, ba):
        crossed = bb >= ba
        mid = (bb + ba) / 2.0
        vol_grid.on_event(ts_str, 0, mid, crossed)
        rel_bps = ((ba - bb) / mid) * 10000.0 if mid > 0 else 0.0
        spr_grid.on_event(ts_str, 0, rel_bps, crossed)
        bbs = ch.st.bids.best()
        bas = ch.st.asks.best()
        ab = tf - ch.wb_t.get(bbs, tf)
        aa = tf - ch.wa_t.get(bas, tf)
        stale = 1.0 if (ab > STALE_SECONDS or aa > STALE_SECONDS) else 0.0
        stale_grid.on_event(ts_str, 0, stale, crossed)

    counts = stream_day(ch, path_for(day), gaps, on_event)
    if vol_grid.grid_start is None:
        return None

    grid_start, mids, crossed_flags = vol_grid.finalize()
    _, spreads, _ = spr_grid.finalize()
    _, stales, _ = stale_grid.finalize()

    gap_dt = []
    for g in gaps:
        if g["start_ts"] is None or g["end_ts"] is None:
            continue
        gap_dt.append((parse_ts(g["start_ts"]), parse_ts(g["end_ts"])))

    step = VOL_STEP_SECONDS
    sub_mids = mids[::step]
    sub_crossed = crossed_flags[::step]
    rets = []
    n_excl = 0
    for i in range(len(sub_mids) - 1):
        t_i = grid_start + timedelta(seconds=i * step)
        t_ip1 = grid_start + timedelta(seconds=(i + 1) * step)
        straddles = any(gs < t_ip1 and ge > t_i for gs, ge in gap_dt)
        p0, p1 = sub_mids[i], sub_mids[i + 1]
        touches_crossed = sub_crossed[i] or sub_crossed[i + 1]
        if not straddles and not touches_crossed and p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
        else:
            n_excl += 1

    if len(rets) >= 2:
        n = len(rets)
        mean_r = sum(rets) / n
        var_r = sum((r - mean_r) ** 2 for r in rets) / n
        std_r = math.sqrt(var_r)
        ann = std_r * math.sqrt(SECONDS_PER_YEAR / step)
    else:
        n, ann = len(rets), None

    clean = []
    n_cross_sec = 0
    for i in range(len(spreads)):
        t_i = grid_start + timedelta(seconds=i)
        if crossed_flags[i]:
            n_cross_sec += 1
            continue
        if any(gs <= t_i <= ge for gs, ge in gap_dt):
            continue
        clean.append(spreads[i])
    clean.sort()

    n_s = len(mids)
    adjacency = {}
    for N in GAP_ADJACENCY_N:
        cnt = 0
        for i in range(n_s):
            t_i = grid_start + timedelta(seconds=i)
            if any((gs - timedelta(seconds=N)) <= t_i <= (ge + timedelta(seconds=N))
                   for gs, ge in gap_dt):
                cnt += 1
        adjacency[N] = 100.0 * cnt / n_s if n_s else 0.0
    stale_pct = 100.0 * sum(1 for s in stales if s >= 0.5) / n_s if n_s else 0.0

    span_s = None
    lam = None
    if counts["first_ts"] and counts["last_ts"]:
        span_s = (parse_ts(counts["last_ts"]) - parse_ts(counts["first_ts"])).total_seconds()
        if span_s > 0:
            lam = counts["unique_trades"] / span_s

    return {
        "day": day,
        "grid_seconds": n_s, "crossed_seconds": n_cross_sec,
        "vol_n_returns": n, "vol_excluded": n_excl, "vol_ann": ann,
        "median_spread_bp": percentile(clean, 50) if clean else None,
        "clean_spread_samples": len(clean),
        "n_gaps": len(gaps),
        "updates_lost": sum(g["updates_lost"] for g in gaps),
        "adjacency": adjacency, "stale_pct": stale_pct,
        "span_seconds": span_s, "lambda": lam,
        "size": os.path.getsize(path_for(day)),
        "drops": ch.n_drops,
        "uid_span": (counts["last_uid"] - counts["first_uid"])
        if (counts["first_uid"] is not None and counts["last_uid"] is not None) else None,
    }, counts


def run_arm(repair, label):
    emit("")
    emit("  %s" % label)
    ch = Chain(repair)
    for d in WARMUP:
        g = []
        stream_day(ch, path_for(d), g, lambda *a: None)
        emit("    warm-up %s chained (last_update_id=%s, drops=%d)"
             % (d, ch.st.last_update_id, ch.n_drops))
        sys.stdout.flush()
    rows = []
    for d in RECOVER:
        before = ch.n_drops
        r, counts = measure_day(ch, d)
        r["drops_this_day"] = ch.n_drops - before
        r["counts"] = counts
        rows.append(r)
        emit("    %s: grid_s=%d crossed_s=%d vol_n=%d vol=%s spread_n=%d drops=%d"
             % (d, r["grid_seconds"], r["crossed_seconds"], r["vol_n_returns"],
                ("%.2f%%" % (r["vol_ann"] * 100)) if r["vol_ann"] is not None else "n/a",
                r["clean_spread_samples"], r["drops_this_day"]))
        sys.stdout.flush()
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys)) if len(xs) >= 3 else None


emit("=" * 78)
emit("regime_recover.py; recovering 2026-08-17 .. 08-22")
emit("=" * 78)
emit("Repair: while best_bid >= best_ask, drop whichever touch level was written")
emit("        at the lower update-ID. No time constant. (Validated at 6c9c249:")
emit("        mid error 704.5371 bp -> 0.6378 bp, zero false positives.)")
emit("Anchor: 08-15's own snapshot, chained through 08-16 as warm-up.")
emit("Scope : market_data_droplet only. No committed module touched.")

control_rows = run_arm(False, "ARM 1 of 2: CONTROL (no repair)")

emit("")
emit("=" * 78)
emit("Self-check: does the duplicated loop reproduce regime_scan_results.txt?")
emit("=" * 78)
fields = ["lines", "unique_trades", "uid_span", "grid_seconds", "crossed_seconds",
          "vol_n_returns", "vol_excluded", "clean_spread_samples", "n_gaps",
          "updates_lost", "snapshots"]
ok = True
for r in control_rows:
    want = COMMITTED[r["day"]]
    got = (r["counts"]["lines"], r["counts"]["unique_trades"], r["uid_span"],
           r["grid_seconds"], r["crossed_seconds"], r["vol_n_returns"],
           r["vol_excluded"], r["clean_spread_samples"], r["n_gaps"],
           r["updates_lost"], r["counts"]["snapshots"])
    bad = [fields[i] for i in range(len(fields)) if want[i] != got[i]]
    emit("  %s  %s%s" % (r["day"], "OK" if not bad else "*** mismatch ***",
                         ("  fields: " + ", ".join(bad)) if bad else ""))
    if bad:
        ok = False
        for i in range(len(fields)):
            if want[i] != got[i]:
                emit("      %-22s committed=%-12s got=%-12s" % (fields[i], want[i], got[i]))
    if r["vol_ann"] is not None:
        emit("      *** control produced a volatility where the committed run had none")
        ok = False
if not ok:
    emit("")
    emit("self-check FAILED. The duplicate has drifted from 32803ca; it cannot be")
    emit("trusted to measure the repaired arm either. Reporting nothing. HALTING.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    sys.exit(1)
emit("  all six days OK on all 11 fields, and no volatility in any control day.")

rep_rows = run_arm(True, "ARM 2 of 2: REPAIRED")

emit("")
emit("=" * 78)
emit("What the repair recovered")
emit("=" * 78)
emit("  %-10s %-9s %-9s %-8s %-9s %-11s %-9s"
     % ("day", "crossed%", "->crossed%", "drops", "vol", "spread bp", "spread n"))
for c, r in zip(control_rows, rep_rows):
    emit("  %-10s %8.2f%% %9.2f%% %8d %9s %11s %9d"
         % (r["day"], 100.0 * c["crossed_seconds"] / c["grid_seconds"],
            100.0 * r["crossed_seconds"] / r["grid_seconds"], r["drops_this_day"],
            ("%.2f%%" % (r["vol_ann"] * 100)) if r["vol_ann"] is not None else "n/a",
            ("%.4f" % r["median_spread_bp"]) if r["median_spread_bp"] is not None else "n/a",
            r["clean_spread_samples"]))

emit("")
emit("=" * 78)
emit("Spread decision; the rule was fixed before the run")
emit("=" * 78)
emit("  (A) gap-adjacent exposure < 1% of 1s samples within 120s of a gap, every day")
emit("  (B) stale touch within the clean-day range %.2f%%-%.2f%% (spread_stale_test), every day"
     % (CLEAN_STALE_LO, CLEAN_STALE_HI))
emit("")
emit("  %-10s %-10s %-10s %-10s %-12s" % ("day", "N=0", "N=30", "N=120", "stale touch"))
for r in rep_rows:
    emit("  %-10s %9.4f%% %9.4f%% %9.4f%% %11.4f%%"
         % (r["day"], r["adjacency"][0], r["adjacency"][30], r["adjacency"][120],
            r["stale_pct"]))

fail_a = [r["day"] for r in rep_rows if r["adjacency"][120] >= 1.0]
fail_b = [r["day"] for r in rep_rows if r["stale_pct"] > CLEAN_STALE_HI]
emit("")
emit("  (A) days failing the 1%% gap-adjacency bound: %s"
     % (", ".join(fail_a) if fail_a else "None"))
emit("  (B) days exceeding the clean-day stale ceiling %.2f%%: %s"
     % (CLEAN_STALE_HI, ", ".join(fail_b) if fail_b else "None"))
SPREAD_ADMITTED = (not fail_a) and (not fail_b)
emit("")
if SPREAD_ADMITTED:
    emit("  => spread admitted for the six recovered days. Both conditions hold.")
else:
    emit("  => Spread not admitted. The volatility recovery stands on its own;")
    emit("     the spread series for these six days stays out.")

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    cached = json.load(f)
for c in cached:
    y, m, dd = (int(x) for x in c["date"].split("-"))
    c["d"] = date(y, m, dd)

rep_by_day = {r["day"]: r for r in rep_rows}
merged = []
n_replaced = 0
for c in cached:
    if not c.get("usable"):
        continue
    key = c["d"].strftime("%m-%d")
    if key in rep_by_day:
        r = rep_by_day[key]
        merged.append({"d": c["d"], "vol": r["vol_ann"], "spread": r["median_spread_bp"],
                       "lam": r["lambda"], "size": float(r["size"]), "recovered": True})
        n_replaced += 1
    elif c["vol_ann"] is not None:
        merged.append({"d": c["d"], "vol": c["vol_ann"], "spread": c["median_spread_bp"],
                       "lam": c["lambda"], "size": float(c["size"]), "recovered": False})
merged.sort(key=lambda x: x["d"])
old = [m for m in merged if not m["recovered"]]
new = [m for m in merged if m["recovered"]]

emit("")
emit("=" * 78)
emit("Prediction test; does each recovered day land where size predicts?")
emit("=" * 78)
emit("  OLS of volatility on file size, fitted on the %d previously clean days only," % len(old))
emit("  so the recovered days are a genuine out-of-sample test of the recovery.")
ox = [m["size"] for m in old]
oy = [m["vol"] for m in old]
n0 = len(ox)
mx, my = sum(ox) / n0, sum(oy) / n0
sxx = sum((x - mx) ** 2 for x in ox)
sxy = sum((ox[i] - mx) * (oy[i] - my) for i in range(n0))
slope = sxy / sxx
intercept = my - slope * mx
resid = [oy[i] - (intercept + slope * ox[i]) for i in range(n0)]
rse = math.sqrt(sum(r * r for r in resid) / (n0 - 2))
emit("  fit: vol = %.6e + %.6e * size    residual SD = %.4f pp   r = %+.4f"
     % (intercept, slope, rse * 100, pearson(ox, oy)))
emit("  in-sample size range: %.0f .. %.0f bytes" % (min(ox), max(ox)))
emit("")
emit("  %-10s %-12s %-11s %-11s %-11s %-8s" %
     ("day", "size", "predicted", "measured", "residual", "in SDs"))
for m in new:
    pred = intercept + slope * m["size"]
    res = m["vol"] - pred
    flag = ""
    if m["size"] > max(ox):
        flag = "  <-- size EXTRAPOLATES beyond the fitted range"
    emit("  %-10s %12.0f %10.2f%% %10.2f%% %+10.2fpp %+7.2f%s"
         % (m["d"].isoformat(), m["size"], pred * 100, m["vol"] * 100,
            res * 100, res / rse, flag))
big = [m for m in new if abs((m["vol"] - (intercept + slope * m["size"])) / rse) > 3.0]
emit("")
if big:
    emit("  *** %d recovered day(s) miss the prediction by more than 3 residual SDs: %s"
         % (len(big), ", ".join(m["d"].isoformat() for m in big)))
    emit("      Per the standing instruction this is evidence the recovery is wrong,")
    emit("      not a finding about the market.")
else:
    emit("  No recovered day misses by more than 3 residual SDs. The recovery is")
    emit("  consistent with the relation fitted on days it had no part in.")

emit("")
emit("=" * 78)
emit("Updated regime analysis; %d days (was %d)" % (len(merged), len(old)))
emit("=" * 78)
vols = [m["vol"] for m in merged]
sizes = [m["size"] for m in merged]
lams = [m["lam"] for m in merged]
sv = sorted(vols)
emit("  realised vol: min=%.2f%% median=%.2f%% max=%.2f%% max/min=%.3fx"
     % (sv[0] * 100, percentile(sv, 50) * 100, sv[-1] * 100, sv[-1] / sv[0]))
lo = min(merged, key=lambda m: m["vol"])
hi = max(merged, key=lambda m: m["vol"])
emit("    lowest  %s %.2f%%   highest %s %.2f%%"
     % (lo["d"].isoformat(), lo["vol"] * 100, hi["d"].isoformat(), hi["vol"] * 100))
sl = sorted(lams)
emit("  lambda      : min=%.5f median=%.5f max=%.5f max/min=%.3fx"
     % (sl[0], percentile(sl, 50), sl[-1], sl[-1] / sl[0]))
ss = sorted(sizes)
emit("  file size   : min=%.0f median=%.0f max=%.0f max/min=%.3fx"
     % (ss[0], percentile(ss, 50), ss[-1], ss[-1] / ss[0]))

emit("")
emit("  SIZE vs volatility")
emit("    %d days (committed, 32803ca): Pearson +0.9413  Spearman +0.9396" % len(old))
rp, rs = pearson(sizes, vols), spearman(sizes, vols)
t = rp * math.sqrt((len(merged) - 2) / (1 - rp * rp))
emit("    %d days (with recovery)     : Pearson %+.4f  Spearman %+.4f  (r2=%.4f, t=%+.3f on %d df)"
     % (len(merged), rp, rs, rp * rp, t, len(merged) - 2))
emit("    vol vs lambda: Pearson %+.4f   size vs lambda: Pearson %+.4f"
     % (pearson(vols, lams), pearson(sizes, lams)))

emit("")
emit("=" * 78)
emit("Does the weekday/WEEKEND split survive?")
emit("=" * 78)
emit("  The six recovered days are five weekdays (08-17 Mon .. 08-21 Fri) and one")
emit("  saturday (08-22). Their absence was censoring the WEEKDAY side, so this is")
emit("  the adversarial test of the split, not a confirmation of it.")
wend = [m for m in merged if m["d"].weekday() >= 5]
wday = [m for m in merged if m["d"].weekday() < 5]
for name, grp in (("WEEKDAY", wday), ("WEEKEND", wend)):
    v = sorted(g["vol"] for g in grp)
    l = sorted(g["lam"] for g in grp)
    emit("  %-8s n=%2d  vol min=%.2f%% median=%.2f%% mean=%.2f%% max=%.2f%%   lambda median=%.5f"
         % (name, len(v), v[0] * 100, percentile(v, 50) * 100,
            (sum(v) / len(v)) * 100, v[-1] * 100, percentile(l, 50)))
wd_v = sorted(g["vol"] for g in wday)
we_v = sorted(g["vol"] for g in wend)
emit("  median ratio weekday/weekend: %.3fx  (was 2.150x on %d days)"
     % (percentile(wd_v, 50) / percentile(we_v, 50), len(old)))

by_vol = sorted(merged, key=lambda m: m["vol"])
emit("")
emit("  the test: how far down the volatility ranking is the split clean?")
for k in (5, 8, 10, 12, 14, 16):
    if k > len(by_vol):
        continue
    bottom = by_vol[:k]
    nw = sum(1 for m in bottom if m["d"].weekday() >= 5)
    emit("    bottom %2d volatility days: %2d of %2d are weekend (%5.1f%%) %s"
         % (k, nw, k, 100.0 * nw / k, "all weekend" if nw == k else ""))
emit("    weekend base rate: %d of %d days (%.1f%%)"
     % (len(wend), len(merged), 100.0 * len(wend) / len(merged)))

emit("")
emit("  Where the recovered days rank (1 = lowest volatility of %d):" % len(merged))
for m in new:
    emit("    rank %2d/%d  %s (%s)  vol=%6.2f%%  lambda=%.5f  size=%10.0f"
         % (by_vol.index(m) + 1, len(merged), m["d"].isoformat(),
            m["d"].strftime("%a"), m["vol"] * 100, m["lam"], m["size"]))

emit("")
emit("  Recovered days beside their clean neighbours (chronological):")
for m in merged:
    if date(2026, 8, 14) <= m["d"] <= date(2026, 8, 29):
        emit("    %s (%s) %s vol=%6.2f%%  spread=%s  lambda=%.5f  size=%10.0f"
             % (m["d"].isoformat(), m["d"].strftime("%a"),
                "RECOVERED" if m["recovered"] else "         ",
                m["vol"] * 100,
                ("%7.4fbp" % m["spread"]) if m["spread"] is not None else "    n/a",
                m["lam"], m["size"]))

emit("")
emit("  depth profile remains excluded for these six days, two independent reasons")
emit("  already on record: a stale-drop rule removes real deep resting size, and the")
emit("  deep ask side here is structurally incomplete (448 missing asks at 08-21).")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print("")
print("Results written to " + RESULTS_PATH)
