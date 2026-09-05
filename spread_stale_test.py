# spread_stale_test.py: read-only: is the spread series contaminated by stuck inside-touch levels?

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())

from compute_volatility import Side, State, parse_ts

DATA_DIR = "market_data_droplet"

STALE_SECONDS = 300
SAMPLE_STEP = 10


def path_for(day):
    return os.path.join(DATA_DIR, "btcusd_2026-" + day + ".jsonl")


def ts_float(s):
    return parse_ts(s).timestamp()


def percentile(sorted_vals, p):
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

    def sample(self, now_f):
        bb = self.st.bids.best()
        ba = self.st.asks.best()
        if bb is None or ba is None:
            return None
        bbf, baf = float(bb), float(ba)
        cutoff = now_f - STALE_SECONDS
        raw_stale_bid = self.tb.get(bb, 0.0) < cutoff
        raw_stale_ask = self.ta.get(ba, 0.0) < cutoff

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

        raw_mid = (bbf + baf) / 2.0
        raw_bps = ((baf - bbf) / raw_mid) * 10000.0 if raw_mid > 0 else None
        crossed = bbf >= baf

        true_bps = None
        if tbb is not None and tba is not None:
            tmid = (tbb + tba) / 2.0
            if tmid > 0:
                true_bps = ((tba - tbb) / tmid) * 10000.0
        return {"raw_bps": raw_bps, "true_bps": true_bps, "crossed": crossed,
                "stale_touch": raw_stale_bid or raw_stale_ask,
                "stale_bid": raw_stale_bid, "stale_ask": raw_stale_ask}


def run_day(tr, day, collect):
    """Streams one day, sampling every SAMPLE_STEP seconds of wall clock"""
    raws, trues = [], []
    n = n_stale = n_crossed = n_stale_bid = n_stale_ask = 0
    next_sample = None
    with open(path_for(day), "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' in line:
                continue
            rec = json.loads(line)
            rtype = rec.get("type")
            tf = ts_float(rec["local_timestamp"])

            if rtype == "snapshot":
                tr.snapshot(rec["data"], tf)
                if next_sample is None:
                    next_sample = tf
                continue
            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if tr.st.last_update_id is None or data["u"] <= tr.st.last_update_id:
                continue
            tr.diff(data, tf)
            if next_sample is None:
                next_sample = tf
            while collect and next_sample is not None and tf >= next_sample:
                s = tr.sample(next_sample)
                next_sample += SAMPLE_STEP
                if s is None:
                    continue
                n += 1
                if s["crossed"]:
                    n_crossed += 1
                if s["stale_touch"]:
                    n_stale += 1
                if s["stale_bid"]:
                    n_stale_bid += 1
                if s["stale_ask"]:
                    n_stale_ask += 1
                if s["raw_bps"] is not None and not s["crossed"]:
                    raws.append(s["raw_bps"])
                if s["true_bps"] is not None:
                    trues.append(s["true_bps"])
    if not collect:
        return
    raws.sort()
    trues.sort()
    print("")
    print("  --- %s ---" % day)
    print("    samples=%d  crossed=%d (%.2f%%)  raw touch stale>%ds=%d (%.2f%%)"
          " [stale bid=%d, stale ask=%d]"
          % (n, n_crossed, 100.0 * n_crossed / n if n else 0, STALE_SECONDS,
             n_stale, 100.0 * n_stale / n if n else 0, n_stale_bid, n_stale_ask))
    print("    median spread raw  (as regime_scan measured it): %s bp"
          % (("%.4f" % percentile(raws, 50)) if raws else "n/a"))
    print("    median spread TRUE (stale levels dropped)      : %s bp"
          % (("%.4f" % percentile(trues, 50)) if trues else "n/a"))
    if raws and trues:
        r, t = percentile(raws, 50), percentile(trues, 50)
        if r and r > 0:
            print("    ratio TRUE/RAW = %.1fx   <== >5x means the raw figure is an artifact"
                  % (t / r))


print("=" * 78)
print("Question 4: is the spread series contaminated by THE SAME mechanism?")
print("=" * 78)
print("STALE_SECONDS=%d  SAMPLE_STEP=%ds" % (STALE_SECONDS, SAMPLE_STEP))

print("")
print("chain A: anchor 07-19 (own snapshot) -> 07-20 -> 07-21")
trA = Tracked()
run_day(trA, "07-19", True)
run_day(trA, "07-20", True)
run_day(trA, "07-21", True)

print("")
print("chain B: anchor 08-26 (own snapshot) -> 08-27 -> 08-28")
trB = Tracked()
run_day(trB, "08-26", True)
run_day(trB, "08-27", True)
run_day(trB, "08-28", True)
