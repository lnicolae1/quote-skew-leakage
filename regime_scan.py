# regime_scan.py: per-day calibration quantities across the droplet-collected data

import json
import math
import os
import sys
from datetime import date, timedelta

from compute_volatility import Side, State, parse_ts, SECONDS_PER_YEAR
from compute_volatility_clean import (
    CrossAwareGridBuilder, stream_reconstruct_with_cross,
)
from spread_depth_profile import percentile

DATA_DIR = "market_data_droplet"

WARMUP_START = date(2026, 7, 11)
USABLE_START = date(2026, 7, 15)
USABLE_END = date(2026, 8, 29)

VOL_STEP_SECONDS = 300

RESULTS_PATH = "regime_scan_results.txt"
CACHE_PATH = "regime_scan_rows.json"


def _stream_day(path, state, gaps, on_event):
    """A copy of compute_volatility_clean.stream_reconstruct_with_cross's body"""
    n_lines = 0
    n_snapshot = 0
    n_diff = 0
    trade_ids = set()
    n_trade_lines = 0
    first_ts = None
    last_ts = None
    first_uid = None
    last_uid = None

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
                if state.last_update_id is not None and snap_uid > state.last_update_id:
                    gaps.append({"kind": "invisible_span", "start_ts": state.last_ts,
                                 "end_ts": rec_ts, "start_uid": state.last_update_id,
                                 "end_uid": snap_uid,
                                 "updates_lost": snap_uid - state.last_update_id})
                elif state.last_update_id is not None and snap_uid < state.last_update_id:
                    raise RuntimeError("Backward anchor jump in " + path)
                state.bids.load_snapshot(snap["bids"])
                state.asks.load_snapshot(snap["asks"])
                state.last_update_id = snap_uid
                state.last_ts = rec_ts
                n_snapshot += 1
                if first_uid is None:
                    first_uid = snap_uid
                last_uid = snap_uid
                bb, ba = state.bids.best(), state.asks.best()
                if bb is not None and ba is not None:
                    bbf, baf = float(bb), float(ba)
                    crossed = bbf >= baf
                    mval = (bbf + baf) / 2.0
                    on_event(rec_ts, state.last_update_id, mval, crossed, bbf, baf)
                continue

            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if state.last_update_id is None or data["u"] <= state.last_update_id:
                continue
            if data["U"] != state.last_update_id + 1:
                gaps.append({"kind": "resumption", "start_ts": state.last_ts,
                             "end_ts": rec_ts, "start_uid": state.last_update_id,
                             "end_uid": data["U"],
                             "updates_lost": data["U"] - state.last_update_id - 1})
            state.bids.apply(data["b"])
            state.asks.apply(data["a"])
            state.last_update_id = data["u"]
            state.last_ts = rec_ts
            n_diff += 1
            if first_uid is None:
                first_uid = data["u"]
            last_uid = data["u"]
            bb, ba = state.bids.best(), state.asks.best()
            if bb is not None and ba is not None:
                bbf, baf = float(bb), float(ba)
                crossed = bbf >= baf
                mval = (bbf + baf) / 2.0
                on_event(rec_ts, state.last_update_id, mval, crossed, bbf, baf)

    return {
        "lines": n_lines, "snapshots": n_snapshot, "diffs": n_diff,
        "trade_lines": n_trade_lines, "unique_trades": len(trade_ids),
        "first_ts": first_ts, "last_ts": last_ts,
        "first_uid": first_uid, "last_uid": last_uid,
    }


def _write_synthetic(path):
    """A synthetic day exercising: an anchor snapshot, sub-second diff"""
    lines = []

    def snap(ts, uid, bids, asks):
        lines.append(json.dumps({
            "local_timestamp": ts, "type": "snapshot",
            "data": {"lastUpdateId": uid, "bids": bids, "asks": asks}}))

    def diff(ts, U, u, b, a):
        lines.append(json.dumps({
            "local_timestamp": ts, "type": "diff", "stream": "btcusd@depth@100ms",
            "data": {"e": "depthUpdate", "E": 1, "s": "BTCUSD",
                     "U": U, "u": u, "b": b, "a": a}}))

    def trade(ts, aid):
        lines.append(json.dumps({
            "local_timestamp": ts, "type": "diff", "stream": "btcusd@aggTrade",
            "data": {"e": "aggTrade", "a": aid, "p": "100.01", "q": "0.5",
                     "m": True}}))

    T = "2026-07-15T00:00:"
    snap(T + "00.000000+00:00", 1000,
         [["100.00", "1.0"], ["99.99", "2.0"]],
         [["100.02", "1.0"], ["100.03", "2.0"]])
    trade(T + "00.100000+00:00", 1)
    diff(T + "00.200000+00:00", 1001, 1001, [["100.00", "1.5"]], [])
    diff(T + "00.700000+00:00", 1002, 1002, [], [["100.02", "0.5"]])
    trade(T + "00.900000+00:00", 2)
    diff(T + "01.300000+00:00", 1003, 1003, [["100.01", "1.0"]], [])
    diff(T + "02.100000+00:00", 1004, 1004, [["100.05", "1.0"]], [])
    trade(T + "02.500000+00:00", 3)
    trade(T + "02.500000+00:00", 3)
    diff(T + "11.400000+00:00", 1005, 1005, [["100.05", "0"]], [])
    diff(T + "13.000000+00:00", 1010, 1010, [], [["100.04", "3.0"]])
    diff(T + "13.050000+00:00", 1011, 1011, [["99.98", "1.0"]], [])
    trade(T + "14.000000+00:00", 4)
    diff(T + "20.750000+00:00", 1012, 1012, [], [["100.03", "0"]])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def self_test():
    print("=== self-test: duplicated loop vs committed stream_reconstruct_with_cross ===")
    tmp_dir = os.environ.get("TEMP") or "."
    path = os.path.join(tmp_dir, "regime_scan_selftest.jsonl")
    _write_synthetic(path)

    grid_ref = CrossAwareGridBuilder()
    state_ref = State()
    gaps_ref = []
    stream_reconstruct_with_cross(path, state_ref, gaps_ref, grid_ref.on_event)
    start_ref, mids_ref, cross_ref = grid_ref.finalize()

    grid_new = CrossAwareGridBuilder()
    state_new = State()
    gaps_new = []

    def adapter(ts, uid, mid, crossed, bb, ba):
        grid_new.on_event(ts, uid, mid, crossed)

    counts = _stream_day(path, state_new, gaps_new, adapter)
    start_new, mids_new, cross_new = grid_new.finalize()

    ok = True

    def check(name, a, b):
        nonlocal ok
        good = (a == b)
        print("  " + ("PASS" if good else "FAIL") + "  " + name +
              ("" if good else ("  ref=" + repr(a) + "  new=" + repr(b))))
        if not good:
            ok = False

    check("grid_start", start_ref, start_new)
    check("len(mids)", len(mids_ref), len(mids_new))
    check("len(crossed_flags)", len(cross_ref), len(cross_new))
    if len(mids_ref) == len(mids_new):
        mismatches = [i for i in range(len(mids_ref)) if mids_ref[i] != mids_new[i]]
        check("mids elementwise (" + str(len(mids_ref)) + " samples)", [], mismatches)
        xmis = [i for i in range(len(cross_ref)) if cross_ref[i] != cross_new[i]]
        check("crossed flags elementwise", [], xmis)
    check("final last_update_id", state_ref.last_update_id, state_new.last_update_id)
    check("gap count", len(gaps_ref), len(gaps_new))

    check("unique aggTrade ids (4 sent, one duplicated)", 4, counts["unique_trades"])
    check("aggTrade lines seen (5, incl. the duplicate)", 5, counts["trade_lines"])
    check("first applied uid", 1000, counts["first_uid"])
    check("last applied uid", 1012, counts["last_uid"])
    check("at least one crossed second present", True, any(cross_new))
    check("grid spans the multi-second silences", True, len(mids_new) >= 21)

    try:
        os.remove(path)
    except OSError:
        pass

    if not ok:
        print("\nSELF-test FAILED; halting before any day is read.")
        sys.exit(1)
    print("  All PASS. The duplicated loop reproduces the committed one elementwise.\n")


def day_paths():
    out = []
    d = WARMUP_START
    while d <= USABLE_END:
        p = os.path.join(DATA_DIR, "btcusd_" + d.isoformat() + ".jsonl")
        out.append((d, p, os.path.exists(p)))
        d += timedelta(days=1)
    return out


def measure_day(path, state):
    """Reads one day, chaining state"""
    gaps = []
    vol_grid = CrossAwareGridBuilder()
    spr_grid = CrossAwareGridBuilder()

    def on_event(ts_str, uid, mid, crossed, bb, ba):
        vol_grid.on_event(ts_str, uid, mid, crossed)
        rel_bps = ((ba - bb) / mid) * 10000.0 if mid > 0 else 0.0
        spr_grid.on_event(ts_str, uid, rel_bps, crossed)

    counts = _stream_day(path, state, gaps, on_event)

    if vol_grid.grid_start is None:
        return None, counts, gaps

    grid_start, mids, crossed_flags = vol_grid.finalize()
    _, spreads, _ = spr_grid.finalize()

    gap_dt_intervals = []
    for g in gaps:
        if g["start_ts"] is None or g["end_ts"] is None:
            continue
        gap_dt_intervals.append((parse_ts(g["start_ts"]), parse_ts(g["end_ts"])))

    step = VOL_STEP_SECONDS
    sub_mids = mids[::step]
    sub_crossed = crossed_flags[::step]
    n_pairs = len(sub_mids) - 1
    rets_clean = []
    n_excl = 0
    for i in range(n_pairs):
        t_i = grid_start + timedelta(seconds=i * step)
        t_ip1 = grid_start + timedelta(seconds=(i + 1) * step)
        straddles_gap = any(gs < t_ip1 and ge > t_i for gs, ge in gap_dt_intervals)
        p0, p1 = sub_mids[i], sub_mids[i + 1]
        touches_crossed = sub_crossed[i] or sub_crossed[i + 1]
        if not straddles_gap and not touches_crossed and p0 > 0 and p1 > 0:
            rets_clean.append(math.log(p1 / p0))
        else:
            n_excl += 1

    if len(rets_clean) >= 2:
        n = len(rets_clean)
        mean_r = sum(rets_clean) / n
        var_r = sum((r - mean_r) ** 2 for r in rets_clean) / n
        std_r = math.sqrt(var_r)
        ann = std_r * math.sqrt(SECONDS_PER_YEAR / step)
    else:
        n, std_r, ann = len(rets_clean), None, None

    clean_spreads = []
    n_cross_sec = 0
    for i in range(len(spreads)):
        t_i = grid_start + timedelta(seconds=i)
        if crossed_flags[i]:
            n_cross_sec += 1
            continue
        if any(gs <= t_i <= ge for gs, ge in gap_dt_intervals):
            continue
        clean_spreads.append(spreads[i])
    clean_spreads.sort()
    med_spread = percentile(clean_spreads, 50) if clean_spreads else None
    p25_spread = percentile(clean_spreads, 25) if clean_spreads else None
    p75_spread = percentile(clean_spreads, 75) if clean_spreads else None

    span_s = None
    lam = None
    if counts["first_ts"] and counts["last_ts"]:
        span_s = (parse_ts(counts["last_ts"]) - parse_ts(counts["first_ts"])).total_seconds()
        if span_s > 0:
            lam = counts["unique_trades"] / span_s

    row = {
        "grid_seconds": len(mids),
        "crossed_seconds": n_cross_sec,
        "vol_n_returns": n,
        "vol_excluded": n_excl,
        "vol_std": std_r,
        "vol_ann": ann,
        "median_spread_bp": med_spread,
        "p25_spread_bp": p25_spread,
        "p75_spread_bp": p75_spread,
        "clean_spread_samples": len(clean_spreads),
        "span_seconds": span_s,
        "lambda": lam,
        "n_gaps": len(gaps),
        "updates_lost": sum(g["updates_lost"] for g in gaps),
    }
    row.update(counts)
    return row, counts, gaps


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def ranks(vals):
    """Average ranks, ties shared"""
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
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def t_stat_for_r(r, n):
    if r is None or n < 3 or abs(r) >= 1.0:
        return None
    return r * math.sqrt((n - 2) / (1 - r * r))


CACHE_FIELDS = ("size", "lines", "unique_trades", "lambda", "span_seconds",
                "vol_ann", "vol_n_returns", "vol_excluded", "median_spread_bp",
                "p25_spread_bp", "p75_spread_bp", "clean_spread_samples",
                "grid_seconds", "crossed_seconds", "n_gaps", "updates_lost",
                "snapshots", "diffs", "first_uid", "last_uid", "usable")


def save_cache(rows):
    """The per-day numbers, so the analysis below can be re-derived without a"""
    out = []
    for r in rows:
        d = {k: r.get(k) for k in CACHE_FIELDS}
        d["date"] = r["date"].isoformat()
        out.append(d)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def load_cache():
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        y, m, dd = (int(x) for x in r["date"].split("-"))
        r["date"] = date(y, m, dd)
    return rows


def main():
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    from_cache = "--from-cache" in sys.argv

    if not from_cache:
        self_test()

    emit("=== regime_scan.py ===")
    emit("Data: " + DATA_DIR + " (read-only)")
    emit("Warm-up (chained for the book anchor, stats printed but excluded): "
         + WARMUP_START.isoformat() + " .. " + (USABLE_START - timedelta(days=1)).isoformat())
    emit("Usable window (selected by date): " + USABLE_START.isoformat()
         + " .. " + USABLE_END.isoformat())
    emit("Volatility: 5-minute sampling of a 1s previous-tick forward-filled mid grid,")
    emit("  gap-excluded and crossed-book-excluded (compute_volatility_clean.py's estimator).")
    emit("Spread: ((best_ask-best_bid)/mid)*10000, clean 1s samples (spread_depth_profile.py).")
    emit("lambda: unique aggTrades (deduped by id `a`) over the file's raw timestamp span,")
    emit("  NOT gap-corrected.")
    emit("")

    state = State()
    rows = []

    if from_cache:
        rows = load_cache()
        emit("=== Per-day scan (loaded from " + CACHE_PATH + ", no re-parse) ===")
        for r in rows:
            emit("  " + r["date"].isoformat() + " (" + r["date"].strftime("%a") + ")"
                 + "  size=" + ("%10d" % r["size"])
                 + "  vol5m=" + (("%7.2f%%" % (r["vol_ann"] * 100))
                                 if r["vol_ann"] is not None else "    n/a")
                 + "  med_spread_bp=" + (("%8.4f" % r["median_spread_bp"])
                                         if r["median_spread_bp"] is not None else "     n/a")
                 + "  lambda=" + ("%9.5f" % r["lambda"])
                 + "  crossed_s=" + str(r["crossed_seconds"]))
        analyze(rows, emit)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print("\nResults written to " + RESULTS_PATH)
        return

    emit("=== Per-day scan ===")
    for d, path, exists in day_paths():
        usable = (d >= USABLE_START)
        tag = "USABLE " if usable else "warmup "
        if not exists:
            emit("  " + d.isoformat() + "  [" + tag + "] File missing; skipped: " + path)
            continue
        size = os.path.getsize(path)
        row, counts, gaps = measure_day(path, state)
        if row is None:
            emit("  " + d.isoformat() + "  [" + tag + "] no grid produced (" +
                 str(counts["lines"]) + " lines, " + str(size) + " bytes); "
                 "state.last_update_id=" + str(state.last_update_id))
            continue
        row["date"] = d
        row["usable"] = usable
        row["size"] = size
        rows.append(row)

        uid_span = (row["last_uid"] - row["first_uid"]) if (
            row["first_uid"] is not None and row["last_uid"] is not None) else None

        vol_s = ("%.2f%%" % (row["vol_ann"] * 100)) if row["vol_ann"] is not None else "n/a"
        spr_s = ("%.4f" % row["median_spread_bp"]) if row["median_spread_bp"] is not None else "n/a"
        lam_s = ("%.5f" % row["lambda"]) if row["lambda"] is not None else "n/a"
        emit("  " + d.isoformat() + "  [" + tag + "]"
             + "  size=" + ("%10d" % size)
             + "  msgs=" + ("%9d" % row["lines"])
             + "  uid_span=" + ("%9s" % str(uid_span))
             + "  vol5m=" + ("%8s" % vol_s)
             + "  med_spread_bp=" + ("%8s" % spr_s)
             + "  trades=" + ("%7d" % row["unique_trades"])
             + "  lambda=" + ("%9s" % lam_s))
        emit("             grid_s=" + str(row["grid_seconds"])
             + "  crossed_s=" + str(row["crossed_seconds"])
             + "  vol_n=" + str(row["vol_n_returns"])
             + "  vol_excl=" + str(row["vol_excluded"])
             + "  spread_n=" + str(row["clean_spread_samples"])
             + "  span_s=" + (("%.1f" % row["span_seconds"]) if row["span_seconds"] else "n/a")
             + "  gaps=" + str(row["n_gaps"])
             + "  updates_lost=" + str(row["updates_lost"])
             + "  snaps=" + str(row["snapshots"]))

    save_cache(rows)
    print("Per-day numbers cached to " + CACHE_PATH +
          " (re-run the analysis with --from-cache, no re-parse needed)")

    analyze(rows, emit)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\nResults written to " + RESULTS_PATH)


def analyze(rows, emit):
    use = [r for r in rows if r["usable"] and r["vol_ann"] is not None]

    emit("")
    emit("=== crossed-book contamination (stated first: it decides which days exist) ===")
    emit("  A stuck 'ghost' level whose removal is never seen in the diff stream leaves the")
    emit("  reconstructed book crossed (best_bid >= best_ask) until the next snapshot")
    emit("  re-anchors it. Same defect class as the documented local 07-09 -> 07-10 event.")
    usable_rows = [r for r in rows if r["usable"]]
    dirty = [r for r in usable_rows if r["crossed_seconds"] > 0]
    dead = [r for r in usable_rows if r["vol_ann"] is None]
    emit("  Usable-window days on disk: " + str(len(usable_rows)))
    emit("  Days with any crossed second: " + str(len(dirty)))
    for r in dirty:
        pct = 100.0 * r["crossed_seconds"] / r["grid_seconds"] if r["grid_seconds"] else 0.0
        emit("    %s (%s)  crossed %6d/%5d s = %6.2f%%   vol_returns_left=%3d %s"
             % (r["date"].isoformat(), r["date"].strftime("%a"), r["crossed_seconds"],
                r["grid_seconds"], pct, r["vol_n_returns"],
                "<-- No volatility possible" if r["vol_ann"] is None else ""))
    emit("  Days lost entirely (no computable volatility): " + str(len(dead))
         + " -> " + ", ".join(r["date"].isoformat() for r in dead))
    lost_bytes = sum(r["size"] for r in dead)
    all_bytes = sum(r["size"] for r in usable_rows)
    emit("  Bytes lost to contamination: %d of %d (%.1f%% of the usable window)"
         % (lost_bytes, all_bytes, 100.0 * lost_bytes / all_bytes if all_bytes else 0.0))
    if dead:
        biggest = max(usable_rows, key=lambda r: r["size"])
        emit("  note: the largest file in the usable window is " + biggest["date"].isoformat()
             + " (" + str(biggest["size"]) + " bytes, " + str(biggest["unique_trades"])
             + " trades)" + (" and it is one of the lost days."
                             if biggest["vol_ann"] is None else "."))

    emit("")
    emit("Usable days with a computable volatility: " + str(len(use)))

    if len(use) < 3:
        emit("Too few usable days to answer the three questions. Stopping.")
        return

    vols = [r["vol_ann"] for r in use]
    sprs = [r["median_spread_bp"] for r in use]
    lams = [r["lambda"] for r in use]
    sizes = [float(r["size"]) for r in use]
    msgs = [float(r["lines"]) for r in use]

    def rng_block(name, vals, fmt, scale=1.0):
        sv = sorted(vals)
        lo, hi = sv[0], sv[-1]
        med = percentile(sv, 50)
        p25 = percentile(sv, 25)
        p75 = percentile(sv, 75)
        ratio = (hi / lo) if lo > 0 else float("inf")
        emit("  " + name + ":")
        emit(("    min=" + fmt + "  p25=" + fmt + "  median=" + fmt + "  p75=" + fmt +
              "  max=" + fmt + "  max/min=%.3fx")
             % (lo * scale, p25 * scale, med * scale, p75 * scale, hi * scale, ratio))
        return lo, hi, med, ratio

    emit("")
    emit("=== Question 1: range across the " + str(len(use)) + " usable days ===")
    v_lo, v_hi, v_med, v_ratio = rng_block("realised vol, 5min, annualized", vols, "%.2f%%", 100.0)
    lo_day = min(use, key=lambda r: r["vol_ann"])
    hi_day = max(use, key=lambda r: r["vol_ann"])
    emit("    lowest  vol day: " + lo_day["date"].isoformat() + "  (" +
         ("%.2f%%" % (lo_day["vol_ann"] * 100)) + ", size=" + str(lo_day["size"]) + ")")
    emit("    highest vol day: " + hi_day["date"].isoformat() + "  (" +
         ("%.2f%%" % (hi_day["vol_ann"] * 100)) + ", size=" + str(hi_day["size"]) + ")")

    rng_block("median relative spread, bp", sprs, "%.4f")
    slo = min(use, key=lambda r: r["median_spread_bp"])
    shi = max(use, key=lambda r: r["median_spread_bp"])
    emit("    narrowest day: " + slo["date"].isoformat() + "  (" +
         ("%.4f bp" % slo["median_spread_bp"]) + ")")
    emit("    widest    day: " + shi["date"].isoformat() + "  (" +
         ("%.4f bp" % shi["median_spread_bp"]) + ")")

    rng_block("lambda, unique aggTrades/sec", lams, "%.5f")
    llo = min(use, key=lambda r: r["lambda"])
    lhi = max(use, key=lambda r: r["lambda"])
    emit("    quietest day: " + llo["date"].isoformat() + "  (" +
         ("%.5f/s" % llo["lambda"]) + ")")
    emit("    busiest  day: " + lhi["date"].isoformat() + "  (" +
         ("%.5f/s" % lhi["lambda"]) + ")")

    rng_block("file size, bytes", sizes, "%.0f")
    rng_block("message count", msgs, "%.0f")

    emit("")
    emit("=== Question 2: does size predict volatility? ===")
    n = len(use)

    def corr_line(label, xs, ys):
        rp = pearson(xs, ys)
        rs = spearman(xs, ys)
        tp = t_stat_for_r(rp, n)
        emit("  " + label)
        emit("    Pearson  r = " + (("%+.4f" % rp) if rp is not None else "n/a") +
             "   (r^2 = " + (("%.4f" % (rp * rp)) if rp is not None else "n/a") +
             ", t = " + (("%+.3f" % tp) if tp is not None else "n/a") +
             " on " + str(n - 2) + " df)")
        emit("    Spearman r = " + (("%+.4f" % rs) if rs is not None else "n/a"))

    corr_line("file size  vs  realised vol:", sizes, vols)
    corr_line("message count  vs  realised vol:", msgs, vols)
    corr_line("file size  vs  lambda:", sizes, lams)
    corr_line("file size  vs  median spread:", sizes, sprs)
    corr_line("realised vol  vs  median spread:", vols, sprs)
    corr_line("realised vol  vs  lambda:", vols, lams)
    corr_line("lambda  vs  median spread:", lams, sprs)
    emit("  (two-sided |t| > 2.02 is p<0.05 at " + str(n - 2) + " df; "
         "|t| > 2.69 is p<0.01)")

    emit("")
    emit("=== Question 3: is there a defensible split? ===")
    emit("  Sorted volatility series, with the jump to the next day shown, so a genuine")
    emit("  boundary is visible as a jump rather than asserted:")
    by_vol = sorted(use, key=lambda r: r["vol_ann"])
    for i, r in enumerate(by_vol):
        if i + 1 < len(by_vol):
            nxt = by_vol[i + 1]["vol_ann"]
            jump = nxt - r["vol_ann"]
            rel = jump / r["vol_ann"] if r["vol_ann"] > 0 else float("inf")
            jt = "   jump_to_next=%+.3fpp (%+.2f%%)" % (jump * 100, rel * 100)
        else:
            jt = ""
        emit("    %2d  %s  vol=%7.2f%%  spread=%7.4fbp  lambda=%.5f  size=%10d%s"
             % (i + 1, r["date"].isoformat(), r["vol_ann"] * 100,
                r["median_spread_bp"], r["lambda"], r["size"], jt))

    emit("")
    emit("  Largest consecutive jumps in the sorted volatility series:")
    jumps = []
    for i in range(len(by_vol) - 1):
        a, b = by_vol[i], by_vol[i + 1]
        jumps.append((b["vol_ann"] - a["vol_ann"], i, a, b))
    jumps.sort(reverse=True, key=lambda t: t[0])
    for jump, i, a, b in jumps[:8]:
        emit("    +%.3f pp between rank %d (%s, %.2f%%) and rank %d (%s, %.2f%%)"
             "; a cut here gives %d low / %d high"
             % (jump * 100, i + 1, a["date"].isoformat(), a["vol_ann"] * 100,
                i + 2, b["date"].isoformat(), b["vol_ann"] * 100,
                i + 1, len(by_vol) - i - 1))

    steps = [j[0] for j in jumps]
    steps_sorted = sorted(steps)
    med_step = percentile(steps_sorted, 50)
    emit("")
    emit("  Median consecutive step in the sorted series: %.4f pp" % (med_step * 100))
    emit("  Largest step: %.4f pp  =  %.2fx the median step"
         % (jumps[0][0] * 100, (jumps[0][0] / med_step) if med_step > 0 else float("inf")))
    emit("  For reference, a sample of %d draws from a single distribution still produces"
         % len(by_vol))
    emit("  uneven order statistics; a real regime boundary should be several times the")
    emit("  median step and should reproduce on an independent variable (spread or lambda).")

    emit("")
    emit("  Same series sorted by median spread, for the independent-variable check:")
    by_spr = sorted(use, key=lambda r: r["median_spread_bp"])
    for i, r in enumerate(by_spr):
        emit("    %2d  %s  spread=%7.4fbp  vol=%7.2f%%  lambda=%.5f"
             % (i + 1, r["date"].isoformat(), r["median_spread_bp"],
                r["vol_ann"] * 100, r["lambda"]))

    emit("")
    emit("  Day-by-date volatility series (chronological), to show whether high-vol days")
    emit("  cluster in time (a regime) or are scattered (day-to-day noise):")
    for r in use:
        emit("    %s  vol=%7.2f%%  spread=%7.4fbp  lambda=%.5f  size=%10d"
             % (r["date"].isoformat(), r["vol_ann"] * 100, r["median_spread_bp"],
                r["lambda"], r["size"]))

    v_seq = [r["vol_ann"] for r in use]
    if len(v_seq) >= 4:
        lag1 = pearson(v_seq[:-1], v_seq[1:])
        emit("")
        emit("  Lag-1 autocorrelation of daily volatility (chronological): "
             + (("%+.4f" % lag1) if lag1 is not None else "n/a"))
        emit("  (positive and sizeable => high-vol days cluster into contiguous stretches,")
        emit("   which is what a regime means; near zero => independent day-to-day draws,")
        emit("   in which case a 'regime' split is just a quantile relabelling.)")

    emit("")
    emit("=== Candidate split: WEEKDAY vs WEEKEND (a variable not in the data) ===")
    wend = [r for r in use if r["date"].weekday() >= 5]
    wday = [r for r in use if r["date"].weekday() < 5]
    emit("  weekday days n=" + str(len(wday)) + "   weekend days n=" + str(len(wend)))

    def grp(name, group, key, fmt, scale=1.0):
        if not group:
            emit("  " + name + ": (empty)")
            return
        v = sorted(r[key] for r in group if r[key] is not None)
        if not v:
            emit("  " + name + ": (no values)")
            return
        emit(("  %-9s %-16s n=%2d  min=" + fmt + "  median=" + fmt +
              "  mean=" + fmt + "  max=" + fmt)
             % (name, key, len(v), v[0] * scale, percentile(v, 50) * scale,
                (sum(v) / len(v)) * scale, v[-1] * scale))

    grp("WEEKDAY", wday, "vol_ann", "%7.2f%%", 100.0)
    grp("WEEKEND", wend, "vol_ann", "%7.2f%%", 100.0)
    grp("WEEKDAY", wday, "lambda", "%9.5f")
    grp("WEEKEND", wend, "lambda", "%9.5f")
    grp("WEEKDAY", wday, "size", "%12.0f")
    grp("WEEKEND", wend, "size", "%12.0f")
    grp("WEEKDAY", wday, "median_spread_bp", "%8.4f")
    grp("WEEKEND", wend, "median_spread_bp", "%8.4f")

    by_vol_asc = sorted(use, key=lambda r: r["vol_ann"])
    emit("")
    emit("  the test: how far down the volatility ranking is the split clean?")
    for k in (5, 8, 10, 12, 14, 16):
        if k > len(by_vol_asc):
            continue
        bottom = by_vol_asc[:k]
        n_wend = sum(1 for r in bottom if r["date"].weekday() >= 5)
        emit("    bottom %2d volatility days: %2d of %2d are weekend  (%5.1f%%)  %s"
             % (k, n_wend, k, 100.0 * n_wend / k,
                "all weekend" if n_wend == k else ""))
    n_wend_tot = len(wend)
    emit("    for reference, weekend days are %d of %d usable days (%.1f%%) overall"
         % (n_wend_tot, len(use), 100.0 * n_wend_tot / len(use)))

    emit("")
    emit("  Every usable weekend day, by volatility rank (rank 1 = lowest of all "
         + str(len(use)) + "):")
    for r in sorted(wend, key=lambda r: r["vol_ann"]):
        rank = by_vol_asc.index(r) + 1
        emit("    rank %2d/%d  %s (%s)  vol=%7.2f%%  lambda=%.5f  size=%10d"
             % (rank, len(use), r["date"].isoformat(), r["date"].strftime("%a"),
                r["vol_ann"] * 100, r["lambda"], r["size"]))

    wd_v = sorted(r["vol_ann"] for r in wday)
    we_v = sorted(r["vol_ann"] for r in wend)
    if wd_v and we_v:
        emit("")
        emit("  Separation: lowest weekday = %.2f%% ; highest weekend = %.2f%%"
             % (wd_v[0] * 100, we_v[-1] * 100))
        overlap = [r for r in use
                   if wd_v[0] <= r["vol_ann"] <= we_v[-1]]
        emit("  Days inside the overlap band: " + str(len(overlap))
             + " of " + str(len(use)))
        emit("  Median ratio weekday/weekend: %.3fx"
             % (percentile(wd_v, 50) / percentile(we_v, 50)))


if __name__ == "__main__":
    main()
