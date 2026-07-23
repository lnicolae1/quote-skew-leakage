# diagnose_crossing.py: read-only diagnostic for crossed-book instances in compute_volatility's run

import json
from compute_volatility import (
    Side, State, stream_reconstruct, build_local_path, mid_price
)

LOCAL_DAYS = ["07-05", "07-06", "07-07", "07-08", "07-09", "07-10", "07-11"]


def noop(ts, uid, mid):
    pass


def chain_through_days(days):
    """Reconstruct state by chaining the unmodified stream_reconstruct"""
    state = State()
    gaps = []
    for day in days:
        path = build_local_path(day)
        print("  chaining " + path + " ...")
        stream_reconstruct(path, state, gaps, noop)
    return state, gaps


def top_n(side, n, reverse):
    prices = sorted(side.book.keys(), key=float, reverse=reverse)[:n]
    return [(p, side.book[p]) for p in prices]


def diagnose_day(path, state, stop_at_first_cross_detail=True):
    """Duplicated diff-apply loop (same logic as stream_reconstruct) with"""
    first_cross = None
    crossings = []
    n_events = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' in line:
                continue
            rec = json.loads(line)
            rtype = rec.get("type")
            rec_ts = rec.get("local_timestamp")

            if rtype == "snapshot":
                snap = rec["data"]
                snap_uid = snap["lastUpdateId"]
                gap_before = None
                if state.last_update_id is not None and snap_uid > state.last_update_id:
                    gap_before = (state.last_update_id, snap_uid, snap_uid - state.last_update_id)
                state.bids.load_snapshot(snap["bids"])
                state.asks.load_snapshot(snap["asks"])
                state.last_update_id = snap_uid
                state.last_ts = rec_ts
                print("  SNAPSHOT at " + rec_ts + " lastUpdateId=" + str(snap_uid) +
                      " invisible_span_before=" + str(gap_before))
                bb = state.bids.best()
                ba = state.asks.best()
                if bb is not None and ba is not None and float(bb) >= float(ba):
                    print("    *** book is crossed immediately after this snapshot: bb=" +
                          bb + " ba=" + ba)
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

            contiguous = (data["U"] == state.last_update_id + 1)
            prev_last = state.last_update_id

            state.bids.apply(data["b"])
            state.asks.apply(data["a"])
            state.last_update_id = data["u"]
            state.last_ts = rec_ts
            n_events += 1

            bb = state.bids.best()
            ba = state.asks.best()
            if bb is not None and ba is not None:
                bbf, baf = float(bb), float(ba)
                if bbf >= baf:
                    info = {
                        "ts": rec_ts, "u": data["u"], "U": data["U"],
                        "prev_last_update_id": prev_last, "contiguous": contiguous,
                        "bb": bb, "ba": ba, "bbf": bbf, "baf": baf, "diff": bbf - baf,
                        "raw_b": data["b"], "raw_a": data["a"],
                    }
                    crossings.append(info)
                    if first_cross is None:
                        first_cross = info
                        print("\n  *** first crossing in this file ***")
                        print("  local_timestamp=" + rec_ts + "  update_id(u)=" + str(data["u"]) +
                              "  U=" + str(data["U"]))
                        print("  previous last_update_id=" + str(prev_last) +
                              "  contiguous (U == prev+1)? " + str(contiguous))
                        print("  best_bid=" + bb + "  best_ask=" + ba +
                              "  best_bid - best_ask=" + str(bbf - baf))
                        print("  this diff's raw bid updates: " + str(data["b"]))
                        print("  this diff's raw ask updates: " + str(data["a"]))
                        print("  Top 5 bids (desc): " + str(top_n(state.bids, 5, True)))
                        print("  Top 5 asks (asc):  " + str(top_n(state.asks, 5, False)))
                        if stop_at_first_cross_detail:
                            pass

    return first_cross, crossings, n_events


def main():
    print("=== Chaining state through 07-05..07-08 (unmodified stream_reconstruct) ===")
    state, gaps = chain_through_days(["07-05", "07-06", "07-07", "07-08"])
    print("State after 07-08: last_update_id=" + str(state.last_update_id))
    bb0, ba0 = state.bids.best(), state.asks.best()
    print("Book state entering 07-09: best_bid=" + str(bb0) + " best_ask=" + str(ba0) +
          " crossed? " + str(bb0 is not None and ba0 is not None and float(bb0) >= float(ba0)))

    print("\n=== Diagnosing 07-09 (task 1 + task 2) ===")
    path_09 = build_local_path("07-09")
    first_cross_09, crossings_09, n_events_09 = diagnose_day(path_09, state)
    print("\n07-09 summary: total diffs=" + str(n_events_09) +
          " total crossed instances=" + str(len(crossings_09)))

    if first_cross_09 is not None:
        print("\n--- Ghost search: did a deletion (qty=0) for the offending price ever arrive? ---")
        bb_val = first_cross_09["bbf"]
        ba_val = first_cross_09["baf"]
        print("At first crossing: best_bid=" + str(bb_val) + " best_ask=" + str(ba_val))

        ghost_side = None
        ghost_price = None
        candidates = [first_cross_09["bb"], first_cross_09["ba"]]
        print("Searching " + path_09 + " and " + build_local_path("07-10") +
              " for any deletion (qty->0) of price " + str(candidates) + " ...")

        for price_str in candidates:
            found_deletions = []
            for p in [path_09, build_local_path("07-10")]:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if '"aggTrade"' not in line and '"' + price_str + '"' in line:
                            rec = json.loads(line)
                            if rec.get("type") != "diff":
                                continue
                            data = rec.get("data", {})
                            if data.get("e") != "depthUpdate":
                                continue
                            for side_name, arr in (("b", data.get("b", [])), ("a", data.get("a", []))):
                                for pp, qq in arr:
                                    if pp == price_str and float(qq) == 0.0:
                                        found_deletions.append((p, rec.get("local_timestamp"), data["U"], data["u"], side_name))
            print("  price=" + price_str + ": deletions(qty=0) found = " + str(len(found_deletions)))
            for d in found_deletions[:10]:
                print("    " + str(d))

    print("\n=== Diagnosing 07-10 for median crossing magnitude ===")
    path_10 = build_local_path("07-10")
    first_cross_10, crossings_10, n_events_10 = diagnose_day(path_10, state, stop_at_first_cross_detail=False)
    print("07-10 summary: total diffs=" + str(n_events_10) +
          " total crossed instances=" + str(len(crossings_10)))
    all_diffs = sorted(c["diff"] for c in (crossings_09 + crossings_10))
    if all_diffs:
        mid_idx = len(all_diffs) // 2
        median = all_diffs[mid_idx] if len(all_diffs) % 2 == 1 else (all_diffs[mid_idx - 1] + all_diffs[mid_idx]) / 2
        print("Median crossing magnitude (bb-ba) across 07-09+07-10: " + str(median))
        print("Min crossing magnitude: " + str(all_diffs[0]) + "  Max: " + str(all_diffs[-1]))

    print("\n=== task 4: confirm clearing at 07-11 00:04:10 snapshot ===")
    path_11 = build_local_path("07-11")
    n_crossed_after_snapshot = 0
    n_events_after_snapshot = 0
    seen_target_snapshot = False
    with open(path_11, "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' in line:
                continue
            rec = json.loads(line)
            rtype = rec.get("type")
            rec_ts = rec.get("local_timestamp")
            if rtype == "snapshot":
                snap = rec["data"]
                snap_uid = snap["lastUpdateId"]
                state.bids.load_snapshot(snap["bids"])
                state.asks.load_snapshot(snap["asks"])
                state.last_update_id = snap_uid
                state.last_ts = rec_ts
                bb = state.bids.best()
                ba = state.asks.best()
                crossed_now = bb is not None and ba is not None and float(bb) >= float(ba)
                print("  snapshot at " + rec_ts + " lastUpdateId=" + str(snap_uid) +
                      " bb=" + str(bb) + " ba=" + str(ba) + " crossed=" + str(crossed_now))
                if snap_uid == 4494915679:
                    seen_target_snapshot = True
                    print("  ^ this is the 00:04:10 re-anchor snapshot (lastUpdateId=4494915679)")
                continue
            if rtype != "diff":
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            if state.last_update_id is None or data["u"] <= state.last_update_id:
                continue
            state.bids.apply(data["b"])
            state.asks.apply(data["a"])
            state.last_update_id = data["u"]
            state.last_ts = rec_ts
            if seen_target_snapshot:
                n_events_after_snapshot += 1
                bb = state.bids.best()
                ba = state.asks.best()
                if bb is not None and ba is not None and float(bb) >= float(ba):
                    n_crossed_after_snapshot += 1

    print("Diffs applied after the re-anchor snapshot: " + str(n_events_after_snapshot))
    print("Of those, crossed: " + str(n_crossed_after_snapshot))


if __name__ == "__main__":
    main()
