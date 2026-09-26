# july_citability.py: three real-data figures cited in the paper, from the raw July spot files (one read-only pass)
# - (1) lambda: unique aggTrades per second of coverage, overall and by UTC hour
#   recorded: 0.0451 trades/s = 39,184 / 869,497 s; hours 0.0247 (11:00) to 0.0676 (13:00)
# - (4) longest gap between consecutive records, any type; recorded 22.9 s
# - (5) 2026-07-11 splice: local through u = 4495452846, cloud from U = 4495452847, no id skipped or duplicated
# - window: local market_data/btcusd_2026-07-05..07-11 up to the splice diff; cloud market_data_cloud/btcusd_2026-07-11..07-15
# - btcusdt_* (futures) not read; coverage = sum over files of (last - first record time)
# - anchor: per-file aggTrade counts must equal trade_size_results.txt

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))

SPLICE_UID = 4495452846
LOCAL_FILES = ["market_data/btcusd_2026-07-%02d.jsonl" % d for d in range(5, 12)]
CLOUD_FILES = ["market_data_cloud/btcusd_2026-07-%02d.jsonl" % d
               for d in range(11, 16)]
TRADE_RESULTS = os.path.join(_HERE, "trade_size_results.txt")

RECORDED = {"lambda": 0.0451, "n_trades": 39184, "coverage": 869497,
            "hour_min": (11, 0.0247), "hour_max": (13, 0.0676),
            "count_range": (890, 2433), "gap": 22.9}
FULL_DAY_HOURS = 23.0


def _finite(x, what):
    if x is None or not math.isfinite(x):
        raise ValueError("NON-FINITE %s: %r -- refusing to continue" % (what, x))
    return x


def _rate(num, den, what):
    _finite(num, "numerator of " + what)
    _finite(den, "denominator of " + what)
    if den <= 0:
        raise ValueError("%s: denominator %r is not positive" % (what, den))
    return num / den


def parse_ts(s):
    return datetime.fromisoformat(s).timestamp()


# the pass
class Chain:
    """Accumulates one chain of files: coverage, gaps, trades, depth ids"""

    def __init__(self, name):
        self.name = name
        self.file_spans = []
        self.gaps = []
        self.last_ts = None
        self.depth_last_u = None
        self.resumption = []
        self.snapshots = []


def stream_file(path, chain, trades, trade_counts, stop_at_uid=None,
                splice_state=None):
    """Stream one file into chain; returns records read."""
    first = last = None
    n = 0
    stopped = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data") or {}
            if data.get("e") == "aggTrade":
                trade_counts[path] = trade_counts.get(path, 0) + 1
            if stopped:
                if data.get("e") == "aggTrade" and data["a"] not in trades:
                    trades[data["a"]] = parse_ts(rec["local_timestamp"])
                if splice_state is not None and data.get("e") == "depthUpdate":
                    splice_state["tail_diffs"] += 1
                continue

            n += 1
            ts = parse_ts(rec["local_timestamp"])
            if first is None:
                first = ts
            if chain.last_ts is not None:
                chain.gaps.append((ts - chain.last_ts, chain.last_ts, ts, path))
            chain.last_ts = ts
            last = ts

            if rec.get("type") == "snapshot":
                uid = rec["data"]["lastUpdateId"]
                chain.snapshots.append((path, ts, uid))
                chain.depth_last_u = uid
            elif data.get("e") == "depthUpdate":
                if chain.depth_last_u is not None and data["u"] > chain.depth_last_u:
                    if data["U"] != chain.depth_last_u + 1:
                        chain.resumption.append((path, chain.depth_last_u + 1,
                                                 data["U"]))
                    if splice_state is not None and splice_state.get("cloud") \
                            and splice_state["first_after"] is None \
                            and data["u"] > SPLICE_UID:
                        splice_state["first_after"] = (data["U"], data["u"], ts)
                    chain.depth_last_u = data["u"]
                elif splice_state is not None and splice_state.get("cloud") \
                        and data["u"] <= SPLICE_UID:
                    splice_state["cloud_stale"] += 1
                if stop_at_uid is not None and data["u"] == stop_at_uid:
                    splice_state["local_end"] = (data["U"], data["u"], ts)
                    stopped = True
            elif data.get("e") == "aggTrade":
                if data["a"] not in trades:
                    trades[data["a"]] = ts
    if first is None:
        raise ValueError("%s has no records" % path)
    chain.file_spans.append((path, first, last))
    return n


def measure(root):
    """The whole measurement"""
    trades, trade_counts = {}, {}
    local, cloud = Chain("local"), Chain("cloud")
    splice = {"local_end": None, "first_after": None, "cloud_stale": 0,
              "tail_diffs": 0}
    for rel in LOCAL_FILES:
        stop = SPLICE_UID if rel.endswith("07-11.jsonl") else None
        stream_file(os.path.join(root, rel), local, trades, trade_counts,
                    stop_at_uid=stop, splice_state=splice)
    splice["local_last_u"] = local.depth_last_u
    splice["local_resumption"] = list(local.resumption)
    splice["cloud"] = True
    for rel in CLOUD_FILES:
        stream_file(os.path.join(root, rel), cloud, trades, trade_counts,
                    splice_state=splice)
    splice["cloud_snapshots"] = list(cloud.snapshots)
    splice["cloud_resumption"] = list(cloud.resumption)
    return {"trades": trades, "trade_counts": trade_counts,
            "chains": (local, cloud), "splice": splice}


# analysis
def coverage_by_hour(spans):
    """Split each (first, last) span into UTC hour-of-day bins, seconds"""
    by_hour = [0.0] * 24
    by_day = {}
    for _f, a, b in spans:
        t = a
        while t < b:
            hour_start = math.floor(t / 3600.0) * 3600.0
            end = min(b, hour_start + 3600.0)
            h = int((t // 3600) % 24)
            by_hour[h] += end - t
            day = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0.0) + (end - t)
            t = end
    return by_hour, by_day


def lambda_stats(m):
    local, cloud = m["chains"]
    spans = local.file_spans + cloud.file_spans
    coverage = sum(b - a for _f, a, b in spans)
    continuous = sum(ch.file_spans[-1][2] - ch.file_spans[0][1]
                     for ch in (local, cloud))
    trades = m["trades"]
    n = len(trades)
    by_hour, day_cov = coverage_by_hour(spans)
    hour_n = [0] * 24
    day_n = {}
    for ts in trades.values():
        hour_n[int((ts // 3600) % 24)] += 1
        day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
        day_n[day] = day_n.get(day, 0) + 1
    # no coverage -> None
    hourly = [(h, hour_n[h], by_hour[h],
               _rate(hour_n[h], by_hour[h], "rate at hour %d" % h)
               if by_hour[h] > 0 else None)
              for h in range(24)]
    daily = [(d, day_n.get(d, 0), day_cov[d],
              _rate(day_n.get(d, 0), day_cov[d], "rate on " + d))
             for d in sorted(day_cov)]
    full = [r for _d, _n, c, r in daily if c >= FULL_DAY_HOURS * 3600.0]
    return {"n": n, "coverage": coverage, "continuous": continuous,
            "lam": _rate(n, coverage, "lambda"),
            "lam_cont": _rate(n, continuous, "lambda (continuous span)"),
            "poisson_se": _rate(math.sqrt(n), coverage, "Poisson SE"),
            "hourly": hourly, "daily": daily, "full_days": full}


def gap_stats(m, top=10):
    local, cloud = m["chains"]
    gaps = sorted(local.gaps + cloud.gaps, reverse=True)
    for g in gaps[:top]:
        _finite(g[0], "gap")
    return gaps[:top]


def splice_verdict(s):
    a = (s["local_end"] is not None and s["local_end"][1] == SPLICE_UID
         and s["local_last_u"] == SPLICE_UID)
    snaps = s["cloud_snapshots"]
    b = bool(snaps) and snaps[0][2] == SPLICE_UID
    c = s["first_after"] is not None and s["first_after"][0] == SPLICE_UID + 1
    return a, b, c


def parse_trade_results(lines):
    """Per-file aggTrade record counts and the unique total, as committed"""
    per_file, total = {}, None
    for ln in lines:
        s = ln.strip()
        if s.startswith("market_data") and "aggTrade records" in s:
            name, rest = s.split(":", 1)
            per_file[name.replace("\\", "/")] = int(rest.split()[0])
        if s.startswith("Unique trades"):
            total = int(s.rsplit(":", 1)[1])
    if len(per_file) != 12 or total is None:
        raise ValueError("trade_size_results.txt did not parse: %d files, "
                         "total %r" % (len(per_file), total))
    return per_file, total


def anchor_check(m, committed):
    per_file, total = committed
    fails = []
    for rel in LOCAL_FILES + CLOUD_FILES:
        got = next((v for k, v in m["trade_counts"].items()
                    if k.replace("\\", "/").endswith(rel)), None)
        want = per_file[rel]
        ok = got == want
        print("  %-44s aggTrade records got %6s committed %6d  %s"
              % (rel, got, want, "OK" if ok else "*** miss ***"))
        if not ok:
            fails.append(rel)
    ok = len(m["trades"]) == total
    print("  %-44s got %6d committed %6d  %s" % ("unique aggTrades (all files)",
                                                  len(m["trades"]), total,
                                                  "OK" if ok else "*** miss ***"))
    if not ok:
        fails.append("unique total")
    return fails


# report
def fmt_ts(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S.%f")


def report(m):
    out = {}
    L = lambda_stats(m)
    out["lambda"] = L
    print("=" * 110)
    print("(1) Trade arrival rate; unique aggTrades per second of coverage")
    print("=" * 110)
    print("  unique aggTrades              %d        (recorded %d)"
          % (L["n"], RECORDED["n_trades"]))
    print("  coverage, sum of file spans   %.3f s  (recorded %d s; difference "
          "%+.3f s)" % (L["coverage"], RECORDED["coverage"],
                        L["coverage"] - RECORDED["coverage"]))
    print("  coverage, continuous spans    %.3f s  (also counts the day-boundary "
          "gaps)" % L["continuous"])
    print("  LAMBDA                        %.6f trades/s  (%.4f at 4 dp; recorded "
          "%.4f)" % (L["lam"], L["lam"], RECORDED["lambda"]))
    print("    on the continuous span      %.6f trades/s" % L["lam_cont"])
    print("    Poisson SE (a floor)        %.6f   (trades cluster; "
          "understates the uncertainty)" % L["poisson_se"])
    fd = L["full_days"]
    if len(fd) >= 2:
        mfd = statistics.mean(fd)
        sfd = statistics.stdev(fd)
        print("    daily rates, %d full days    mean %.6f  sd %.6f  SE %.6f  "
              "range %.4f-%.4f" % (len(fd), mfd, sfd, sfd / math.sqrt(len(fd)),
                                   min(fd), max(fd)))
        out["daily_se"] = sfd / math.sqrt(len(fd))
    print("")
    print("  by UTC day:")
    for d, n, c, r in L["daily"]:
        print("    %s  %5d trades  %8.1f s  (%5.2f h)  %.4f/s%s"
              % (d, n, c, c / 3600.0, r,
                 "" if c >= FULL_DAY_HOURS * 3600.0 else "   partial day"))
    print("")
    print("  by UTC hour of day (all days pooled):")
    for h, n, c, r in L["hourly"]:
        if r is None:
            print("    %02d:00  %5d trades  no coverage" % (h, n))
        else:
            print("    %02d:00  %5d trades  %9.1f s  %.4f/s" % (h, n, c, r))
    covered = [z for z in L["hourly"] if z[3] is not None]
    hmin = min(covered, key=lambda z: z[3])
    hmax = max(covered, key=lambda z: z[3])
    counts = [n for _h, n, _c, _r in covered]
    print("  hours with coverage: %d of 24" % len(covered))
    print("  min hour %02d:00 at %.4f/s (recorded %02d:00 at %.4f); max hour "
          "%02d:00 at %.4f/s (recorded %02d:00 at %.4f)"
          % (hmin[0], hmin[3], RECORDED["hour_min"][0], RECORDED["hour_min"][1],
             hmax[0], hmax[3], RECORDED["hour_max"][0], RECORDED["hour_max"][1]))
    ratio = ("%.2f" % _rate(hmax[3], hmin[3], "hour ratio")
             if hmin[3] > 0 else "undefined (the quietest hour has no trades)")
    print("  per-hour counts %d-%d (recorded %d-%d); max/min rate ratio %s"
          % (min(counts), max(counts), RECORDED["count_range"][0],
             RECORDED["count_range"][1], ratio))
    lam_ok = (L["n"] == RECORDED["n_trades"]
              and round(L["lam"], 4) == RECORDED["lambda"])
    hr_ok = (hmin[0] == RECORDED["hour_min"][0]
             and round(hmin[3], 4) == RECORDED["hour_min"][1]
             and hmax[0] == RECORDED["hour_max"][0]
             and round(hmax[3], 4) == RECORDED["hour_max"][1])
    out.update(lam_ok=lam_ok, hours_ok=hr_ok, hmin=hmin, hmax=hmax,
               counts=(min(counts), max(counts)))
    print("  LAMBDA %s.  Hourly extremes %s." % (
        "Reproduces" if lam_ok else "Does not reproduce",
        "reproduce" if hr_ok else "Do not reproduce"))

    print("")
    print("=" * 110)
    print("(4) Longest gap between consecutive records, any type, within each "
          "chain")
    print("=" * 110)
    gaps = gap_stats(m)
    for g, a, b, f in gaps:
        print("  %8.3f s   %s -> %s   %s" % (g, fmt_ts(a), fmt_ts(b),
                                            f.replace("\\", "/")))
    gap_ok = round(gaps[0][0], 1) == RECORDED["gap"]
    out.update(gap=gaps[0][0], gap_ok=gap_ok, gaps=gaps)
    print("  Longest %.3f s (%.1f at 1 dp; recorded %.1f s): %s"
          % (gaps[0][0], gaps[0][0], RECORDED["gap"],
             "reproduces" if gap_ok else "Does not reproduce"))

    print("")
    print("=" * 110)
    print("(5) THE 2026-07-11 splice at update id %d" % SPLICE_UID)
    print("=" * 110)
    s = m["splice"]
    a, b, c = splice_verdict(s)
    le = s["local_end"]
    print("  (a) local chain: diff ending at u=%s (U=%s) at %s; last applied u "
          "= %s  -> %s" % (le[1] if le else None, le[0] if le else None,
                           fmt_ts(le[2]) if le else "-", s["local_last_u"],
                           "OK" if a else "FAIL"))
    print("      local chain resumption gaps (cross-check, expected 4, 11, 1): %s"
          % [(f.replace("\\", "/")[-16:], g - e) for f, e, g in
             s["local_resumption"]])
    print("      overlap tail after it, excluded: %d depth diffs"
          % s["tail_diffs"])
    snaps = s["cloud_snapshots"]
    print("  (b) cloud 07-11 first snapshot lastUpdateId = %s at %s  -> %s"
          % (snaps[0][2] if snaps else None,
             fmt_ts(snaps[0][1]) if snaps else "-", "OK" if b else "FAIL"))
    fa = s["first_after"]
    print("  (c) first cloud diff with u > %d: U=%s u=%s at %s  -> %s"
          % (SPLICE_UID, fa[0] if fa else None, fa[1] if fa else None,
             fmt_ts(fa[2]) if fa else "-", "OK" if c else "FAIL"))
    print("      cloud diffs at or below the splice id, discarded: %d"
          % s["cloud_stale"])
    print("  cloud chain resumption gaps after the splice: %d %s"
          % (len(s["cloud_resumption"]),
             [(f.replace("\\", "/"), e, g) for f, e, g in s["cloud_resumption"]]))
    out.update(splice=(a, b, c), splice_ok=a and b and c)
    print("  splice %s: local through %d, cloud from %s."
          % ("Holds" if (a and b and c) else "Does not hold", SPLICE_UID,
             fa[0] if fa else None))
    return out


# self-tests on planted files
def _planted_tree(tmp):
    """Two tiny chains with a known splice, gaps and trades"""
    import itertools

    def rec(ts, typ, data):
        return json.dumps({"local_timestamp": ts, "type": typ, "data": data})

    def diff(ts, U, u):
        return rec(ts, "diff", {"e": "depthUpdate", "U": U, "u": u,
                                "b": [], "a": []})

    def trade(ts, a):
        return rec(ts, "diff", {"e": "aggTrade", "a": a, "q": "0.001"})

    base = "2026-07-%02dT%s+00:00"
    os.makedirs(os.path.join(tmp, "market_data"))
    os.makedirs(os.path.join(tmp, "market_data_cloud"))
    ids = itertools.count(1)
    for d in range(5, 12):
        lines = []
        if d == 5:
            lines.append(rec(base % (d, "10:00:00.000000"), "snapshot",
                             {"lastUpdateId": 100, "bids": [], "asks": []}))
            lines.append(diff(base % (d, "10:00:01.000000"), 101, 102))
            lines.append(trade(base % (d, "10:00:02.000000"), next(ids)))
            lines.append(diff(base % (d, "10:00:32.500000"), 103, 104))
        else:
            lines.append(trade(base % (d, "00:00:00.000000"), next(ids)))
        if d == 11:
            lines.append(diff(base % (d, "11:00:00.000000"), 105,
                              SPLICE_UID))
            lines.append(trade(base % (d, "11:00:05.000000"), 9999))
            lines.append(diff(base % (d, "11:00:06.000000"), SPLICE_UID + 1,
                              SPLICE_UID + 5))
        with open(os.path.join(tmp, "market_data",
                               "btcusd_2026-07-%02d.jsonl" % d), "w") as f:
            f.write("\n".join(lines) + "\n")
    for d in range(11, 16):
        lines = []
        if d == 11:
            lines.append(rec(base % (d, "10:59:59.000000"), "snapshot",
                             {"lastUpdateId": SPLICE_UID, "bids": [],
                              "asks": []}))
            lines.append(diff(base % (d, "11:00:00.100000"), SPLICE_UID - 3,
                              SPLICE_UID - 1))
            lines.append(diff(base % (d, "11:00:00.200000"), SPLICE_UID + 1,
                              SPLICE_UID + 5))
            lines.append(trade(base % (d, "11:00:05.000000"), 9999))
        lines.append(trade(base % (d, "12:00:00.000000"), next(ids)))
        with open(os.path.join(tmp, "market_data_cloud",
                               "btcusd_2026-07-%02d.jsonl" % d), "w") as f:
            f.write("\n".join(lines) + "\n")


def selftest():
    import io
    import contextlib
    import tempfile
    print("=" * 110)
    print("Self-tests; july_citability.py, on planted files. No real data read.")
    print("=" * 110)
    with tempfile.TemporaryDirectory() as tmp:
        _planted_tree(tmp)
        m = measure(tmp)
        # 7 local + 1 tail (9999) + 5 cloud; the cloud copy of 9999 is not counted again
        assert len(m["trades"]) == 13, len(m["trades"])
        g = gap_stats(m)
        loc_gaps = sorted(x[0] for x in m["chains"][0].gaps)
        assert any(abs(x - 30.5) < 1e-9 for x in loc_gaps)
        a, b, c = splice_verdict(m["splice"])
        assert (a, b, c) == (True, True, True), (a, b, c)
        assert m["splice"]["tail_diffs"] == 1 and m["splice"]["cloud_stale"] == 1
        # local 07-11 span ends at the splice record
        spans = {f.replace("\\", "/")[-16:]: (x, y)
                 for f, x, y in m["chains"][0].file_spans}
        x, y = spans["2026-07-11.jsonl"]
        assert abs((y - x) - 11 * 3600.0) < 1e-6, y - x
        L = lambda_stats(m)
        assert L["n"] == 13 and L["coverage"] > 0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = report(m)
        assert out["splice_ok"] and not out["lam_ok"]
        # broken splice: cloud skips one id after the splice
        p = os.path.join(tmp, "market_data_cloud", "btcusd_2026-07-11.jsonl")
        with open(p) as f:
            txt = f.read()
        with open(p, "w") as f:
            f.write(txt.replace('"U": %d, "u": %d' % (SPLICE_UID + 1,
                                                      SPLICE_UID + 5),
                                '"U": %d, "u": %d' % (SPLICE_UID + 2,
                                                      SPLICE_UID + 5)))
        m2 = measure(tmp)
        assert splice_verdict(m2["splice"]) == (True, True, False)
        assert m2["splice"]["cloud_resumption"], "the skipped id was not logged"
    print("  T-PASS      PASS  planted chains: 13 unique trades with the duplicate "
          "id counted once; the 30.5 s")
    print("                    gap found; the local 07-11 span ends at the splice "
          "record, the tail feeds only")
    print("                    trade counts; splice (a)(b)(c) hold, and a planted "
          "skipped id fails (c) and is logged.")
    for bad in (0.0, -1.0, float("nan")):
        try:
            _rate(1.0, bad, "planted")
            raise AssertionError("_rate accepted %r" % bad)
        except ValueError:
            pass
    print("  T-guard     PASS  zero, negative and NaN denominators raise.")
    with open(TRADE_RESULTS, encoding="utf-8-sig") as f:
        per_file, total = parse_trade_results(f.read().splitlines())
    assert total == 39184 and per_file["market_data/btcusd_2026-07-05.jsonl"] == 876
    assert sum(per_file.values()) == 39184
    print("  T-parse     PASS  trade_size_results.txt parses to 12 files summing "
          "to %d unique trades." % total)
    print("")
    print("  All self-tests PASSED.")
    return 0


def main():
    root = os.environ.get("MARKET_DATA_ROOT", _HERE)
    print("=" * 110)
    print("July citability; lambda, the longest gap and the 07-11 splice, from "
          "the raw spot files")
    print("=" * 110)
    print("Local chain: %s .. %s through u=%d.  Cloud chain: %s .. %s."
          % (LOCAL_FILES[0], LOCAL_FILES[-1], SPLICE_UID, CLOUD_FILES[0],
             CLOUD_FILES[-1]))
    print("Read-only. Timestamps are local_timestamp. Readings fixed "
          "in the header.")
    print("")
    with open(TRADE_RESULTS, encoding="utf-8-sig") as f:
        committed = parse_trade_results(f.read().splitlines())
    m = measure(root)
    print("Anchor; per-file aggTrade counts and the unique total must equal "
          "trade_size_results.txt")
    fails = anchor_check(m, committed)
    if fails:
        raise AssertionError("ANCHOR FAILED on %s -- halting" % ", ".join(fails))
    print("  anchor HELD.")
    print("")
    report(m)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    selftest()
    print("")
    main()
