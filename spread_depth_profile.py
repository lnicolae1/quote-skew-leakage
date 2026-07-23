# spread_depth_profile.py: spread and depth profile over the validated spot dataset
# - local 07-05..07-11 chained to cloud 07-11..07-15 at splice uid 4495452846
# - reuses compute_volatility.py's book state unchanged; own loop to get the full book each second
# - 1 s grid, previous-tick forward fill (sample taken before applying each event)
# - excludes uninitialized, crossed and gap-straddling samples

import bisect
import json
import math
from collections import deque
from datetime import timedelta

from compute_volatility import (
    Side, State, parse_ts, build_local_path, build_cloud_path,
    LOCAL_DAYS, CLOUD_DAYS, SPLICE_UID,
)

# bucket edges, bp: [0,1),[1,2),[2,5),[5,10),[10,20),[20,50),[50,100),[100,inf)
BUCKET_EDGES = [1, 2, 5, 10, 20, 50, 100]
BUCKET_LABELS = ["0-1bp", "1-2bp", "2-5bp", "5-10bp", "10-20bp", "20-50bp", "50-100bp", "100+bp"]
N_BUCKETS = len(BUCKET_LABELS)

GAP_BUFFER_SECONDS = 120


def compute_book_stats(bids, asks, edges):
    """Best bid/ask, spread and per-bucket resting size by distance from mid (bp); None if a side is empty."""
    bb = bids.best()
    ba = asks.best()
    if bb is None or ba is None:
        return None
    bbf = float(bb)
    baf = float(ba)
    mid = (bbf + baf) / 2.0
    if mid <= 0:
        return None
    crossed = bbf >= baf
    abs_spread = baf - bbf
    rel_spread_bps = (abs_spread / mid) * 10000.0

    bid_qty = [0.0] * N_BUCKETS
    bid_cnt = [0] * N_BUCKETS
    ask_qty = [0.0] * N_BUCKETS
    ask_cnt = [0] * N_BUCKETS

    for price_str, qty_str in bids.book.items():
        p = float(price_str)
        dist_bps = (mid - p) / mid * 10000.0
        idx = bisect.bisect_right(edges, dist_bps)
        if idx >= N_BUCKETS:
            idx = N_BUCKETS - 1
        bid_qty[idx] += float(qty_str)
        bid_cnt[idx] += 1

    for price_str, qty_str in asks.book.items():
        p = float(price_str)
        dist_bps = (p - mid) / mid * 10000.0
        idx = bisect.bisect_right(edges, dist_bps)
        if idx >= N_BUCKETS:
            idx = N_BUCKETS - 1
        ask_qty[idx] += float(qty_str)
        ask_cnt[idx] += 1

    return {
        "crossed": crossed,
        "abs_spread": abs_spread,
        "rel_spread_bps": rel_spread_bps,
        "bid_qty": bid_qty, "bid_cnt": bid_cnt,
        "ask_qty": ask_qty, "ask_cnt": ask_cnt,
        "n_bid_levels": len(bids.book),
        "n_ask_levels": len(asks.book),
    }


class ProfileGrid:
    """Forward-fills book stats onto a 1 s grid; trailing buffer lets late-detected gaps exclude samples."""

    def __init__(self):
        self.grid_start = None
        self.next_grid_time = None
        self.pending = deque()

        self.n_samples_total = 0
        self.n_book_empty = 0
        self.n_crossed = 0
        self.n_gap_excluded = 0
        self.n_clean = 0

        self.abs_spreads = []
        self.rel_spreads = []

        self.bucket_bid_qty_sum = [0.0] * N_BUCKETS
        self.bucket_bid_cnt_sum = [0] * N_BUCKETS
        self.bucket_ask_qty_sum = [0.0] * N_BUCKETS
        self.bucket_ask_cnt_sum = [0] * N_BUCKETS

        self.max_bid_levels = 0
        self.max_ask_levels = 0

        self.sum_bid_levels_clean = 0
        self.sum_ask_levels_clean = 0
        self.n_clean_bid_over_1000 = 0
        self.n_clean_ask_over_1000 = 0

    def note_levels(self, n_bid, n_ask):
        if n_bid > self.max_bid_levels:
            self.max_bid_levels = n_bid
        if n_ask > self.max_ask_levels:
            self.max_ask_levels = n_ask

    def advance(self, ts, bids, asks):
        if self.grid_start is None:
            self.grid_start = ts.replace(microsecond=0)
            self.next_grid_time = self.grid_start
            return
        while self.next_grid_time <= ts:
            stats = compute_book_stats(bids, asks, BUCKET_EDGES)
            self.pending.append([self.next_grid_time, stats, False])
            self.next_grid_time += timedelta(seconds=1)

    def register_gap(self, gap_start_dt, gap_end_dt):
        for item in self.pending:
            if gap_start_dt <= item[0] <= gap_end_dt:
                item[2] = True

    def _finalize_one(self, item):
        _, stats, gap_excluded = item
        self.n_samples_total += 1
        if stats is None:
            self.n_book_empty += 1
            return
        if stats["crossed"]:
            self.n_crossed += 1
            return
        if gap_excluded:
            self.n_gap_excluded += 1
            return
        self.n_clean += 1
        self.abs_spreads.append(stats["abs_spread"])
        self.rel_spreads.append(stats["rel_spread_bps"])
        for i in range(N_BUCKETS):
            self.bucket_bid_qty_sum[i] += stats["bid_qty"][i]
            self.bucket_bid_cnt_sum[i] += stats["bid_cnt"][i]
            self.bucket_ask_qty_sum[i] += stats["ask_qty"][i]
            self.bucket_ask_cnt_sum[i] += stats["ask_cnt"][i]
        self.sum_bid_levels_clean += stats["n_bid_levels"]
        self.sum_ask_levels_clean += stats["n_ask_levels"]
        if stats["n_bid_levels"] > 1000:
            self.n_clean_bid_over_1000 += 1
        if stats["n_ask_levels"] > 1000:
            self.n_clean_ask_over_1000 += 1

    def flush_ready(self, current_ts):
        cutoff = current_ts - timedelta(seconds=GAP_BUFFER_SECONDS)
        while self.pending and self.pending[0][0] < cutoff:
            self._finalize_one(self.pending.popleft())

    def finalize_all(self):
        while self.pending:
            self._finalize_one(self.pending.popleft())


def stream_reconstruct_profile(path, state, gaps, grid, stop_at_uid=None, min_start_uid=None):
    n_snapshot = 0
    n_diff = 0

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
                ts = parse_ts(rec_ts)

                gate = (min_start_uid is None) or (
                    state.last_update_id is not None and state.last_update_id > min_start_uid)
                if gate:
                    grid.advance(ts, state.bids, state.asks)

                if state.last_update_id is not None:
                    if snap_uid > state.last_update_id:
                        gap_start_dt = parse_ts(state.last_ts)
                        gaps.append({
                            "kind": "invisible_span", "start_ts": state.last_ts, "end_ts": rec_ts,
                            "start_uid": state.last_update_id, "end_uid": snap_uid,
                            "updates_lost": snap_uid - state.last_update_id,
                        })
                        grid.register_gap(gap_start_dt, ts)
                    elif snap_uid < state.last_update_id:
                        raise RuntimeError(
                            "Backward anchor jump in " + path + ": snapshot lastUpdateId=" +
                            str(snap_uid) + " behind current position=" + str(state.last_update_id))

                state.bids.load_snapshot(snap["bids"])
                state.asks.load_snapshot(snap["asks"])
                state.last_update_id = snap_uid
                state.last_ts = rec_ts
                n_snapshot += 1
                grid.note_levels(len(state.bids.book), len(state.asks.book))
                if gate:
                    grid.flush_ready(ts)
                continue

            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if state.last_update_id is None or data["u"] <= state.last_update_id:
                continue

            ts = parse_ts(rec_ts)
            gate = (min_start_uid is None) or (state.last_update_id > min_start_uid)
            if gate:
                grid.advance(ts, state.bids, state.asks)

            if data["U"] != state.last_update_id + 1:
                gap_start_dt = parse_ts(state.last_ts)
                gaps.append({
                    "kind": "resumption", "start_ts": state.last_ts, "end_ts": rec_ts,
                    "start_uid": state.last_update_id, "end_uid": data["U"],
                    "updates_lost": data["U"] - state.last_update_id - 1,
                })
                grid.register_gap(gap_start_dt, ts)

            state.bids.apply(data["b"])
            state.asks.apply(data["a"])
            state.last_update_id = data["u"]
            state.last_ts = rec_ts
            n_diff += 1
            grid.note_levels(len(state.bids.book), len(state.asks.book))
            if gate:
                grid.flush_ready(ts)

    return {"snapshots": n_snapshot, "diffs": n_diff}


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def stats_block(sorted_vals):
    n = len(sorted_vals)
    mean_v = sum(sorted_vals) / n
    var_v = sum((x - mean_v) ** 2 for x in sorted_vals) / n
    std_v = math.sqrt(var_v)
    pcts = {p: percentile(sorted_vals, p) for p in (1, 5, 25, 50, 75, 95, 99)}
    return mean_v, std_v, pcts


def main():
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    gaps = []
    grid = ProfileGrid()

    emit("=== spread_depth_profile.py ===")
    emit("Reconstructing local chain (07-05 .. 07-11, truncated at splice uid)...")
    local_state = State()
    for day in LOCAL_DAYS:
        path = build_local_path(day)
        stop_uid = SPLICE_UID if day == "07-11" else None
        counts = stream_reconstruct_profile(path, local_state, gaps, grid, stop_at_uid=stop_uid)
        print("  " + path + ": " + str(counts["snapshots"]) + " snapshots, " +
              str(counts["diffs"]) + " diffs, last_update_id=" + str(local_state.last_update_id))
    assert local_state.last_update_id == SPLICE_UID, "local truncation did not land exactly on splice uid"

    emit("\nReconstructing cloud chain (07-11 .. 07-15, own snapshot anchor)...")
    cloud_state = State()
    for day in CLOUD_DAYS:
        path = build_cloud_path(day)
        min_uid = SPLICE_UID if day == "07-11" else None
        counts = stream_reconstruct_profile(path, cloud_state, gaps, grid, min_start_uid=min_uid)
        print("  " + path + ": " + str(counts["snapshots"]) + " snapshots, " +
              str(counts["diffs"]) + " diffs, last_update_id=" + str(cloud_state.last_update_id))

    grid.finalize_all()

    total_lost = sum(g["updates_lost"] for g in gaps)
    emit("\n=== Gap intervals detected during reconstruction ===")
    for g in gaps:
        emit("  " + g["kind"] + ": " + g["start_ts"] + " -> " + g["end_ts"] +
             " (" + str(g["updates_lost"]) + " updates lost)")
    emit("Total gaps found: " + str(len(gaps)) + ", total updates lost: " + str(total_lost))

    emit("\n=== Sample coverage ===")
    emit("Total 1s grid samples considered: " + str(grid.n_samples_total))
    emit("  excluded (book not yet initialized): " + str(grid.n_book_empty))
    emit("  excluded (crossed book, best_bid >= best_ask): " + str(grid.n_crossed))
    emit("  excluded (gap-straddling, within " + str(GAP_BUFFER_SECONDS) + "s buffer window of a detected gap): "
         + str(grid.n_gap_excluded))
    emit("  Clean samples used for stats below: " + str(grid.n_clean))
    if grid.n_samples_total > 0:
        emit("  clean fraction: " + str(round(100 * grid.n_clean / grid.n_samples_total, 3)) + "%")

    # task 1: spread
    emit("\n=== task 1: Spread (clean samples only) ===")
    if grid.n_clean == 0:
        emit("  No clean samples; cannot compute spread stats.")
    else:
        abs_sorted = sorted(grid.abs_spreads)
        rel_sorted = sorted(grid.rel_spreads)

        mean_a, std_a, pct_a = stats_block(abs_sorted)
        emit("\n  Absolute spread (best_ask - best_bid), in USD:")
        emit(f"    n={len(abs_sorted)}  mean={mean_a:.6f}  median={pct_a[50]:.6f}  std={std_a:.6f}")
        for p in (1, 5, 25, 50, 75, 95, 99):
            emit(f"    p{p:>2}: {pct_a[p]:.6f}")

        mean_r, std_r, pct_r = stats_block(rel_sorted)
        emit("\n  Relative spread (spread / mid), in basis points:")
        emit(f"    n={len(rel_sorted)}  mean={mean_r:.4f}  median={pct_r[50]:.4f}  std={std_r:.4f}")
        for p in (1, 5, 25, 50, 75, 95, 99):
            emit(f"    p{p:>2}: {pct_r[p]:.4f}")

    # task 2: depth profile
    emit("\n=== task 2: Depth profile (avg resting size per distance bucket, clean samples only) ===")
    if grid.n_clean == 0:
        emit("  No clean samples; cannot compute depth profile.")
    else:
        emit(f"\n  {'bucket':>10} | {'bid avg qty (BTC)':>18} | {'bid avg #levels':>16} | "
             f"{'ask avg qty (BTC)':>18} | {'ask avg #levels':>16}")
        emit("  " + "-" * 90)
        for i, label in enumerate(BUCKET_LABELS):
            bid_avg_qty = grid.bucket_bid_qty_sum[i] / grid.n_clean
            bid_avg_cnt = grid.bucket_bid_cnt_sum[i] / grid.n_clean
            ask_avg_qty = grid.bucket_ask_qty_sum[i] / grid.n_clean
            ask_avg_cnt = grid.bucket_ask_cnt_sum[i] / grid.n_clean
            emit(f"  {label:>10} | {bid_avg_qty:18.4f} | {bid_avg_cnt:16.2f} | "
                 f"{ask_avg_qty:18.4f} | {ask_avg_cnt:16.2f}")

        emit("\n  Average total levels per clean sample: bid=" +
             str(round(grid.sum_bid_levels_clean / grid.n_clean, 2)) + "  ask=" +
             str(round(grid.sum_ask_levels_clean / grid.n_clean, 2)))

    # task 3: 1000-level cap
    emit("\n=== task 3: 1000-level snapshot-cap caveat ===")
    emit("  REST snapshots are capped at 1000 levels/side (limit=1000); the")
    emit("  diff stream is uncapped, so the reconstructed book can exceed")
    emit("  1000 levels/side as diffs add levels.")
    emit(f"  Max bid levels seen at any point in the full stream: {grid.max_bid_levels}")
    emit(f"  Max ask levels seen at any point in the full stream: {grid.max_ask_levels}")
    if grid.n_clean > 0:
        emit(f"  Clean samples with >1000 bid levels: {grid.n_clean_bid_over_1000} "
             f"({round(100 * grid.n_clean_bid_over_1000 / grid.n_clean, 3)}%)")
        emit(f"  Clean samples with >1000 ask levels: {grid.n_clean_ask_over_1000} "
             f"({round(100 * grid.n_clean_ask_over_1000 / grid.n_clean, 3)}%)")
    beyond_cap = grid.max_bid_levels > 1000 or grid.max_ask_levels > 1000
    if beyond_cap:
        emit("  >>> Beyond-cap levels are present in this dataset (levels/side exceeded 1000 at")
        emit("      least once). Those excess levels can only sit deep in the book; a snapshot")
        emit("      anchor can never itself contain more than 1000 levels/side, so anything past")
        emit("      level 1000 was built entirely from diff-stream additions accumulating since")
        emit("      the last (re)connect. Practically, this means the far-tail bucket (100+bp)")
        emit("      is the one most likely to be contaminated by this beyond-cap accumulation --")
        emit("      it may not represent 'real' resting depth as cleanly as a full independent")
        emit("      snapshot would, since it's a mix of originally-snapshotted levels and levels")
        emit("      added later purely via diffs with no independent cross-check beyond rank 1000")
        emit("      (per the verified fact: never compare reconstructed depth to a snapshot past")
        emit("      rank 1000). The near-touch buckets (0-1bp .. 50-100bp) are NOT affected --")
        emit("      they sit well inside the 1000-level cap essentially always.")
    else:
        emit("  No sample in this run exceeded 1000 levels/side; no beyond-cap contamination")
        emit("      observed in this dataset window.")

    with open("spread_depth_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\nResults written to spread_depth_results.txt")


if __name__ == "__main__":
    main()
