# compute_volatility_clean.py: compute_volatility.py plus exclusion of crossed-book samples (best_bid >= best_ask)
# - reuses Side and State unchanged; duplicated per-line loop threads a crossed flag to the grid

import json
import math
from datetime import timedelta

from compute_volatility import (
    Side, State, parse_ts, build_local_path, build_cloud_path,
    LOCAL_DAYS, CLOUD_DAYS, SPLICE_UID, SECONDS_PER_YEAR,
)


def stream_reconstruct_with_cross(path, state, gaps, on_event, stop_at_uid=None, min_start_uid=None):
    """compute_volatility.stream_reconstruct plus a crossed flag passed to on_event."""
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
                if state.last_update_id is not None and snap_uid > state.last_update_id:
                    gaps.append({"kind": "invisible_span", "start_ts": state.last_ts,
                                 "end_ts": rec_ts, "start_uid": state.last_update_id,
                                 "end_uid": snap_uid, "updates_lost": snap_uid - state.last_update_id})
                elif state.last_update_id is not None and snap_uid < state.last_update_id:
                    raise RuntimeError("Backward anchor jump in " + path)
                state.bids.load_snapshot(snap["bids"])
                state.asks.load_snapshot(snap["asks"])
                state.last_update_id = snap_uid
                state.last_ts = rec_ts
                bb, ba = state.bids.best(), state.asks.best()
                if bb is not None and ba is not None:
                    bbf, baf = float(bb), float(ba)
                    crossed = bbf >= baf
                    mval = (bbf + baf) / 2.0
                    if min_start_uid is None or state.last_update_id > min_start_uid:
                        on_event(rec_ts, state.last_update_id, mval, crossed)
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
                             "end_uid": data["U"], "updates_lost": data["U"] - state.last_update_id - 1})
            state.bids.apply(data["b"])
            state.asks.apply(data["a"])
            state.last_update_id = data["u"]
            state.last_ts = rec_ts
            bb, ba = state.bids.best(), state.asks.best()
            if bb is not None and ba is not None:
                bbf, baf = float(bb), float(ba)
                crossed = bbf >= baf
                mval = (bbf + baf) / 2.0
                if min_start_uid is None or state.last_update_id > min_start_uid:
                    on_event(rec_ts, state.last_update_id, mval, crossed)


class CrossAwareGridBuilder:
    """GridBuilder forward fill with a crossed flag per 1 s sample."""

    def __init__(self):
        self.grid_start = None
        self.next_grid_time = None
        self.last_mid = None
        self.last_crossed = None
        self.mids = []
        self.crossed_flags = []
        self.last_event_ts = None

    def on_event(self, ts_str, uid, mid, crossed):
        ts = parse_ts(ts_str)
        if self.grid_start is None:
            self.grid_start = ts.replace(microsecond=0)
            self.next_grid_time = self.grid_start
            self.mids.append(mid)
            self.crossed_flags.append(crossed)
            self.next_grid_time += timedelta(seconds=1)
            self.last_mid = mid
            self.last_crossed = crossed
            self.last_event_ts = ts
            return
        while self.next_grid_time <= ts:
            self.mids.append(self.last_mid)
            self.crossed_flags.append(self.last_crossed)
            self.next_grid_time += timedelta(seconds=1)
        self.last_mid = mid
        self.last_crossed = crossed
        self.last_event_ts = ts

    def finalize(self):
        while self.next_grid_time <= self.last_event_ts:
            self.mids.append(self.last_mid)
            self.crossed_flags.append(self.last_crossed)
            self.next_grid_time += timedelta(seconds=1)
        return self.grid_start, self.mids, self.crossed_flags


def main():
    gaps = []
    grid = CrossAwareGridBuilder()

    print("=== Reconstructing local chain (07-05 .. 07-11, truncated at splice uid) ===")
    local_state = State()
    for day in LOCAL_DAYS:
        path = build_local_path(day)
        stop_uid = SPLICE_UID if day == "07-11" else None
        stream_reconstruct_with_cross(path, local_state, gaps, grid.on_event, stop_at_uid=stop_uid)
        print("  " + path + " done, last_update_id=" + str(local_state.last_update_id))
    assert local_state.last_update_id == SPLICE_UID

    print("\n=== Reconstructing cloud chain (07-11 .. 07-15, own snapshot anchor) ===")
    cloud_state = State()
    for day in CLOUD_DAYS:
        path = build_cloud_path(day)
        min_uid = SPLICE_UID if day == "07-11" else None
        stream_reconstruct_with_cross(path, cloud_state, gaps, grid.on_event, min_start_uid=min_uid)
        print("  " + path + " done, last_update_id=" + str(cloud_state.last_update_id))

    grid_start, mids, crossed_flags = grid.finalize()
    n_crossed_seconds = sum(1 for c in crossed_flags if c)
    print("\nGrid built: " + str(len(mids)) + " one-second samples, " +
          str(n_crossed_seconds) + " of them crossed (" +
          str(round(100 * n_crossed_seconds / len(mids), 3)) + "%)")

    gap_dt_intervals = [(parse_ts(g["start_ts"]), parse_ts(g["end_ts"])) for g in gaps]
    print("Gap intervals (same as compute_volatility.py): " + str(len(gap_dt_intervals)))

    intervals = [("1s", 1), ("10s", 10), ("1min", 60), ("5min", 300)]

    print("\n=== Volatility, gap-excluded only (original, for reference) vs "
          "gap+crossed-excluded (clean) ===")
    for label, step in intervals:
        sub_mids = mids[::step]
        sub_crossed = crossed_flags[::step]
        n_pairs = len(sub_mids) - 1

        rets_orig = []
        rets_clean = []
        n_excl_gap_only = 0
        n_excl_clean = 0

        for i in range(n_pairs):
            t_i = grid_start + timedelta(seconds=i * step)
            t_ip1 = grid_start + timedelta(seconds=(i + 1) * step)
            straddles_gap = any(gs < t_ip1 and ge > t_i for gs, ge in gap_dt_intervals)
            p0, p1 = sub_mids[i], sub_mids[i + 1]

            if not straddles_gap and p0 > 0 and p1 > 0:
                rets_orig.append(math.log(p1 / p0))
            else:
                n_excl_gap_only += 1

            touches_crossed = sub_crossed[i] or sub_crossed[i + 1]
            if not straddles_gap and not touches_crossed and p0 > 0 and p1 > 0:
                rets_clean.append(math.log(p1 / p0))
            else:
                n_excl_clean += 1

        def stats(rets):
            n = len(rets)
            mean_r = sum(rets) / n
            var_r = sum((r - mean_r) ** 2 for r in rets) / n
            std_r = math.sqrt(var_r)
            periods_per_year = SECONDS_PER_YEAR / step
            return n, mean_r, std_r, std_r * math.sqrt(periods_per_year)

        n_o, mean_o, std_o, ann_o = stats(rets_orig)
        n_c, mean_c, std_c, ann_c = stats(rets_clean)

        print(f"  interval={label:>5}")
        print(f"    gap-excluded only : n={n_o:>8}  excluded={n_excl_gap_only:>7}  "
              f"std={std_o: .3e}  annualized_sigma={ann_o*100:6.2f}%")
        print(f"    gap+crossed-excl. : n={n_c:>8}  excluded={n_excl_clean:>7}  "
              f"std={std_c: .3e}  annualized_sigma={ann_c*100:6.2f}%")


if __name__ == "__main__":
    main()
