# recovery_test.py: read-only; recovery of the six lost days under each candidate repair
# - raw touch as regime_scan; true touch: best levels touched within STALE_SECONDS
# - repair (b) works iff true touch stays ~1 tick; repair (a): book before/after each snapshot

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from compute_volatility import State, parse_ts

DATA_DIR = "market_data_droplet"
STALE_SECONDS = 300
SAMPLE_STEP = 10


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


class Tracked:
    def __init__(self):
        self.st = State()
        self.tb = {}
        self.ta = {}

    def snapshot(self, snap, tf):
        self.st.bids.load_snapshot(snap["bids"])
        self.st.asks.load_snapshot(snap["asks"])
        self.tb = {p: tf for p, _ in snap["bids"]}
        self.ta = {p: tf for p, _ in snap["asks"]}
        self.st.last_update_id = snap["lastUpdateId"]

    def diff(self, data, tf):
        self.st.bids.apply(data["b"])
        self.st.asks.apply(data["a"])
        for p, q in data["b"]:
            if float(q) == 0.0:
                self.tb.pop(p, None)
            else:
                self.tb[p] = tf
        for p, q in data["a"]:
            if float(q) == 0.0:
                self.ta.pop(p, None)
            else:
                self.ta[p] = tf
        self.st.last_update_id = data["u"]

    def raw(self):
        bb = self.st.bids.best()
        ba = self.st.asks.best()
        if bb is None or ba is None:
            return None
        return float(bb), float(ba)

    def true(self, now_f):
        cutoff = now_f - STALE_SECONDS
        tbb = None
        for p, t in self.tb.items():
            if t < cutoff:
                continue
            pf = float(p)
            if tbb is None or pf > tbb:
                tbb = pf
        tba = None
        for p, t in self.ta.items():
            if t < cutoff:
                continue
            pf = float(p)
            if tba is None or pf < tba:
                tba = pf
        return tbb, tba


def run(tr, day, report):
    n = raw_x = true_x = true_none = 0
    true_bps = []
    raw_bps = []
    n_stale_levels_sum = 0
    next_s = None
    last_cross_clear_ts = None
    snap_events = []
    with open(path_for(day), "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' in line:
                continue
            rec = json.loads(line)
            rt = rec.get("type")
            tstr = rec["local_timestamp"]
            tf = parse_ts(tstr).timestamp()
            if rt == "snapshot":
                before = tr.raw()
                tr.snapshot(rec["data"], tf)
                after = tr.raw()
                snap_events.append((tstr, rec["data"]["lastUpdateId"], before, after, tf))
                if next_s is None:
                    next_s = tf
                continue
            if rt != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if tr.st.last_update_id is None or data["u"] <= tr.st.last_update_id:
                continue
            tr.diff(data, tf)
            if next_s is None:
                next_s = tf
            while next_s is not None and tf >= next_s:
                now = next_s
                next_s += SAMPLE_STEP
                r = tr.raw()
                if r is None:
                    continue
                n += 1
                bb, ba = r
                if bb >= ba:
                    raw_x += 1
                else:
                    mid = (bb + ba) / 2.0
                    raw_bps.append(((ba - bb) / mid) * 10000.0)
                tbb, tba = tr.true(now)
                if tbb is None or tba is None:
                    true_none += 1
                elif tbb >= tba:
                    true_x += 1
                else:
                    tmid = (tbb + tba) / 2.0
                    true_bps.append(((tba - tbb) / tmid) * 10000.0)
                cutoff = now - STALE_SECONDS
                if bb >= ba:
                    n_stale_levels_sum += sum(1 for p, t in tr.ta.items()
                                              if t < cutoff and float(p) <= bb)
    if not report:
        return
    raw_bps.sort()
    true_bps.sort()
    print("")
    print("  --- %s ---  samples=%d" % (day, n))
    print("    RAW  crossed: %6d (%6.2f%%)   usable raw samples: %d"
          % (raw_x, 100.0 * raw_x / n if n else 0, len(raw_bps)))
    print("    TRUE crossed: %6d (%6.2f%%)   usable true samples: %d   true-touch empty: %d"
          % (true_x, 100.0 * true_x / n if n else 0, len(true_bps), true_none))
    if raw_bps:
        print("    RAW  spread bp: p25=%.4f median=%.4f p75=%.4f"
              % (pct(raw_bps, 25), pct(raw_bps, 50), pct(raw_bps, 75)))
    if true_bps:
        print("    TRUE spread bp: p25=%.4f median=%.4f p75=%.4f"
              % (pct(true_bps, 25), pct(true_bps, 50), pct(true_bps, 75)))
    if raw_x:
        print("    mean # stale ask levels at or below best_bid while crossed: %.2f"
              % (n_stale_levels_sum / raw_x))
    for tstr, uid, before, after, tf in snap_events:
        print("    SNAPSHOT %s uid=%s" % (tstr, uid))
        print("        before: %s  crossed=%s" % (before, before and before[0] >= before[1]))
        print("        after : %s  crossed=%s" % (after, after and after[0] >= after[1]))


print("=" * 78)
print("RECOVERY test: per-day recovery by repair")
print("STALE_SECONDS=%d  SAMPLE_STEP=%ds" % (STALE_SECONDS, SAMPLE_STEP))
print("=" * 78)

tr = Tracked()
for day in ("08-15", "08-16"):
    run(tr, day, False)
    print("  (chained %s)" % day)
    sys.stdout.flush()

for day in ("08-17", "08-18", "08-19", "08-20", "08-21", "08-22", "08-23"):
    run(tr, day, True)
    sys.stdout.flush()
