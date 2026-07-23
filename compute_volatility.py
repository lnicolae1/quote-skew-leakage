# compute_volatility.py: annualized sigma of BTCUSD mid log returns, full spot dataset
# - local 07-05..07-11 chained to cloud 07-11..07-15 at SPLICE_UID
# - mid, not trades (bid-ask bounce biases sigma high; Roll 1984)
# - returns straddling a coverage gap excluded

import heapq
import json
import math
from datetime import datetime, timezone, timedelta

LOCAL_DIR = "market_data"
CLOUD_DIR = "market_data_cloud"

LOCAL_DAYS = ["07-05", "07-06", "07-07", "07-08", "07-09", "07-10", "07-11"]
CLOUD_DAYS = ["07-11", "07-12", "07-13", "07-14", "07-15"]

# local ends u=4495452846, cloud starts U=4495452847
SPLICE_UID = 4495452846

SECONDS_PER_YEAR = 365 * 24 * 3600


def parse_ts(s):
    return datetime.fromisoformat(s)


class Side:
    """One book side; best price via lazy-deletion heap."""

    def __init__(self, is_bid):
        self.is_bid = is_bid
        self.book = {}
        self.heap = []    # (key, price_str); bid key = -price

    def _key(self, price_str):
        pf = float(price_str)
        return -pf if self.is_bid else pf

    def apply(self, updates):
        for price_str, qty_str in updates:
            if float(qty_str) == 0.0:
                self.book.pop(price_str, None)
            else:
                is_new = price_str not in self.book
                self.book[price_str] = qty_str
                if is_new:
                    heapq.heappush(self.heap, (self._key(price_str), price_str))

    def load_snapshot(self, levels):
        self.book = {p: q for p, q in levels}
        self.heap = [(self._key(p), p) for p, _ in levels]
        heapq.heapify(self.heap)

    def best(self):
        while self.heap:
            _, price_str = self.heap[0]
            if price_str in self.book:
                return price_str
            heapq.heappop(self.heap)
        return None


def mid_price(bids, asks):
    bb = bids.best()
    ba = asks.best()
    if bb is None or ba is None:
        return None
    bb_f, ba_f = float(bb), float(ba)
    return (bb_f + ba_f) / 2.0, bb_f, ba_f


class State:
    def __init__(self):
        self.bids = Side(is_bid=True)
        self.asks = Side(is_bid=False)
        self.last_update_id = None
        self.last_ts = None


def stream_reconstruct(path, state, gaps, on_event, stop_at_uid=None, min_start_uid=None):
    """Apply snapshot/depthUpdate records to state; on_event if uid > min_start_uid; stop once last_update_id >= stop_at_uid; returns counts."""
    n_snapshot = 0
    n_diff = 0
    n_crossed = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if stop_at_uid is not None and state.last_update_id is not None \
                    and state.last_update_id >= stop_at_uid:
                break

            if '"aggTrade"' in line:
                continue

            rec = json.loads(line)
            rtype = rec.get("type")
            rec_ts = rec.get("local_timestamp")

            if rtype == "snapshot":
                snap = rec["data"]
                snap_uid = snap["lastUpdateId"]

                if state.last_update_id is not None:
                    if snap_uid > state.last_update_id:
                        gaps.append({
                            "kind": "invisible_span",
                            "start_ts": state.last_ts, "end_ts": rec_ts,
                            "start_uid": state.last_update_id, "end_uid": snap_uid,
                            "updates_lost": snap_uid - state.last_update_id,
                        })
                    elif snap_uid < state.last_update_id:
                        raise RuntimeError(
                            "Backward anchor jump in " + path + ": snapshot lastUpdateId=" +
                            str(snap_uid) + " behind current position=" + str(state.last_update_id)
                        )

                state.bids.load_snapshot(snap["bids"])
                state.asks.load_snapshot(snap["asks"])
                state.last_update_id = snap_uid
                state.last_ts = rec_ts
                n_snapshot += 1

                m = mid_price(state.bids, state.asks)
                if m is not None:
                    mval, bb, ba = m
                    if bb >= ba:
                        n_crossed += 1
                    if min_start_uid is None or state.last_update_id > min_start_uid:
                        on_event(rec_ts, state.last_update_id, mval)
                continue

            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if state.last_update_id is None:
                continue
            if data["u"] <= state.last_update_id:
                continue

            if data["U"] != state.last_update_id + 1:
                gaps.append({
                    "kind": "resumption",
                    "start_ts": state.last_ts, "end_ts": rec_ts,
                    "start_uid": state.last_update_id, "end_uid": data["U"],
                    "updates_lost": data["U"] - state.last_update_id - 1,
                })

            state.bids.apply(data["b"])
            state.asks.apply(data["a"])
            state.last_update_id = data["u"]
            state.last_ts = rec_ts
            n_diff += 1

            m = mid_price(state.bids, state.asks)
            if m is not None:
                mval, bb, ba = m
                if bb >= ba:
                    n_crossed += 1
                if min_start_uid is None or state.last_update_id > min_start_uid:
                    on_event(rec_ts, state.last_update_id, mval)

    return {"snapshots": n_snapshot, "diffs": n_diff, "crossed": n_crossed}


class GridBuilder:
    """Forward-fill mid onto a 1 s grid (streaming)."""

    def __init__(self):
        self.grid_start = None
        self.next_grid_time = None
        self.last_mid = None
        self.samples = []
        self.last_event_ts = None

    def on_event(self, ts_str, uid, mid):
        ts = parse_ts(ts_str)
        if self.grid_start is None:
            self.grid_start = ts.replace(microsecond=0)
            self.next_grid_time = self.grid_start
            self.samples.append(mid)
            self.next_grid_time += timedelta(seconds=1)
            self.last_mid = mid
            self.last_event_ts = ts
            return

        while self.next_grid_time <= ts:
            self.samples.append(self.last_mid)
            self.next_grid_time += timedelta(seconds=1)

        self.last_mid = mid
        self.last_event_ts = ts

    def finalize(self):
        # pad through the last event's second
        while self.next_grid_time <= self.last_event_ts:
            self.samples.append(self.last_mid)
            self.next_grid_time += timedelta(seconds=1)
        return self.grid_start, self.samples


def load_known_gaps_cross_check(paths):
    found = []
    for p in paths:
        gp = p + ".gaps.json"
        try:
            with open(gp, "r") as f:
                data = json.load(f)
                for g in data:
                    found.append((p, g))
        except FileNotFoundError:
            found.append((p, None))
    return found


def build_local_path(day):
    return LOCAL_DIR + "\\btcusd_2026-" + day + ".jsonl"


def build_cloud_path(day):
    return CLOUD_DIR + "\\btcusd_2026-" + day + ".jsonl"


def main():
    gaps = []
    grid = GridBuilder()

    print("=== Local chain (07-05 .. 07-11, truncated at splice uid) ===")
    local_state = State()
    for day in LOCAL_DAYS:
        path = build_local_path(day)
        stop_uid = SPLICE_UID if day == "07-11" else None
        counts = stream_reconstruct(path, local_state, gaps, grid.on_event, stop_at_uid=stop_uid)
        print("  " + path + ": " + str(counts["snapshots"]) + " snapshots, " +
              str(counts["diffs"]) + " diffs applied, " + str(counts["crossed"]) +
              " crossed-book instances, last_update_id=" + str(local_state.last_update_id))

    print("\nLocal chain stopped at last_update_id=" + str(local_state.last_update_id) +
          " (expect " + str(SPLICE_UID) + ")")
    assert local_state.last_update_id == SPLICE_UID, "local truncation did not land exactly on splice uid"

    print("\n=== Cloud chain (07-11 .. 07-15, own snapshot anchor) ===")
    cloud_state = State()
    for day in CLOUD_DAYS:
        path = build_cloud_path(day)
        min_uid = SPLICE_UID if day == "07-11" else None
        counts = stream_reconstruct(path, cloud_state, gaps, grid.on_event, min_start_uid=min_uid)
        print("  " + path + ": " + str(counts["snapshots"]) + " snapshots, " +
              str(counts["diffs"]) + " diffs applied, " + str(counts["crossed"]) +
              " crossed-book instances, last_update_id=" + str(cloud_state.last_update_id))

    grid_start, samples = grid.finalize()
    print("\nGrid built: start=" + grid_start.isoformat() + ", " + str(len(samples)) +
          " one-second samples (span = " + str(round(len(samples) / 3600, 2)) + " hours)")

    print("\n=== Gaps ===")
    total_lost = 0
    gap_dt_intervals = []
    for g in gaps:
        print("  " + g["kind"] + ": " + g["start_ts"] + " -> " + g["end_ts"] +
              " (" + str(g["updates_lost"]) + " updates lost)")
        total_lost += g["updates_lost"]
        gap_dt_intervals.append((parse_ts(g["start_ts"]), parse_ts(g["end_ts"])))
    print("Total gaps found: " + str(len(gaps)) + ", total updates lost: " + str(total_lost))

    # ---- volatility per interval ----
    intervals = [("1s", 1), ("10s", 10), ("1min", 60), ("5min", 300)]

    print("\n=== Volatility from mid-price log returns ===")
    for label, step in intervals:
        sub = samples[::step]
        n_pairs = len(sub) - 1
        rets = []
        n_excluded = 0
        for i in range(n_pairs):
            t_i = grid_start + timedelta(seconds=i * step)
            t_ip1 = grid_start + timedelta(seconds=(i + 1) * step)
            straddles = any(gs < t_ip1 and ge > t_i for gs, ge in gap_dt_intervals)
            if straddles:
                n_excluded += 1
                continue
            p0, p1 = sub[i], sub[i + 1]
            if p0 <= 0 or p1 <= 0:
                continue
            rets.append(math.log(p1 / p0))

        n = len(rets)
        mean_r = sum(rets) / n
        var_r = sum((r - mean_r) ** 2 for r in rets) / n
        std_r = math.sqrt(var_r)
        periods_per_year = SECONDS_PER_YEAR / step
        annualized = std_r * math.sqrt(periods_per_year)

        print(f"  interval={label:>5}  n_returns={n:>8}  excluded(gap)={n_excluded:>3}  "
              f"mean={mean_r: .3e}  std={std_r: .3e}  annualized_sigma={annualized*100:6.2f}%")


if __name__ == "__main__":
    main()
