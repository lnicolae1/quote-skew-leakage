import json
from compute_volatility import Side, State, build_local_path, stream_reconstruct

def noop(ts,uid,mid): pass

state = State()
gaps = []
for day in ["07-05","07-06","07-07","07-08"]:
    stream_reconstruct(build_local_path(day), state, gaps, noop)
print("Entering 07-09 at last_update_id=", state.last_update_id)

TARGET_BID = "62174.39000000"
TARGET_ASK = "62173.99000000"
FIRST_CROSS_U = 4491244963

path = build_local_path("07-09")
history = []

with open(path, encoding="utf-8") as f:
    for line in f:
        if '"aggTrade"' in line:
            continue
        rec = json.loads(line)
        rtype = rec.get("type")
        ts = rec.get("local_timestamp")
        if rtype == "snapshot":
            snap = rec["data"]
            snap_uid = snap["lastUpdateId"]
            bid_match = [x for x in snap["bids"] if x[0] == TARGET_BID]
            ask_match = [x for x in snap["asks"] if x[0] == TARGET_ASK]
            print("SNAPSHOT uid=", snap_uid, "ts=", ts)
            print("  TARGET_BID in snapshot bids?", bid_match)
            print("  TARGET_ASK in snapshot asks?", ask_match)
            state.bids.load_snapshot(snap["bids"])
            state.asks.load_snapshot(snap["asks"])
            state.last_update_id = snap_uid
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
                history.append(("BID", ts, data["U"], data["u"], q))
        for p,q in data["a"]:
            if p == TARGET_ASK:
                history.append(("ASK", ts, data["U"], data["u"], q))
        state.bids.apply(data["b"])
        state.asks.apply(data["a"])
        state.last_update_id = data["u"]
        state.last_ts = ts
        if data["u"] >= FIRST_CROSS_U:
            break

print("\nFull touch-history of", TARGET_BID, "(bid) and", TARGET_ASK, "(ask) from snapshot to first crossing (u=%d):" % FIRST_CROSS_U)
for h in history:
    print(" ", h)
print("\nTotal touches:", len(history))
