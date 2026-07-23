import json
from compute_volatility import State, build_local_path, build_cloud_path, stream_reconstruct

def noop(ts,uid,mid): pass

state = State()
gaps = []
for day in ["07-05","07-06","07-07","07-08"]:
    stream_reconstruct(build_local_path(day), state, gaps, noop)

TARGET_BID = "62174.39000000"

path = build_local_path("07-09")
history = []
cross_transitions = []
prev_crossed = None
n_events = 0
last_bb = last_ba = None

with open(path, encoding="utf-8") as f:
    for line in f:
        if '"aggTrade"' in line:
            continue
        rec = json.loads(line)
        rtype = rec.get("type")
        ts = rec.get("local_timestamp")
        if rtype == "snapshot":
            snap = rec["data"]
            state.bids.load_snapshot(snap["bids"])
            state.asks.load_snapshot(snap["asks"])
            state.last_update_id = snap["lastUpdateId"]
            state.last_ts = ts
            continue
        if rtype != "diff":
            continue
        data = rec.get("data")
        if not data or data.get("e") != "depthUpdate":
            continue
        if state.last_update_id is None or data["u"] <= state.last_update_id:
            continue
        for p,q in data["b"]:
            if p == TARGET_BID:
                history.append(("BID", ts, data["u"], q))
        state.bids.apply(data["b"])
        state.asks.apply(data["a"])
        state.last_update_id = data["u"]
        state.last_ts = ts
        n_events += 1
        bb, ba = state.bids.best(), state.asks.best()
        crossed = bb is not None and ba is not None and float(bb) >= float(ba)
        if prev_crossed is not None and crossed != prev_crossed:
            cross_transitions.append((ts, data["u"], "-> CROSSED" if crossed else "-> uncrossed", bb, ba))
        prev_crossed = crossed

print("TARGET_BID", TARGET_BID, "full-day touch history (07-09):")
for h in history:
    print(" ", h)

print("\nTotal crossed<->uncrossed transitions during 07-09:", len(cross_transitions))
print("First 20 transitions:")
for t in cross_transitions[:20]:
    print(" ", t)
print("Last 10 transitions:")
for t in cross_transitions[-10:]:
    print(" ", t)

print("\nEnd of 07-09: last_update_id=", state.last_update_id, "bb=", state.bids.best(), "ba=", state.asks.best())
print("Book size at end of 07-09: bids=", len(state.bids.book), "asks=", len(state.asks.book))
