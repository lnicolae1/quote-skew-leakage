import json
from compute_volatility import build_local_path

TARGET = "62196.77000000"

for day in ["07-09", "07-10"]:
    path = build_local_path(day)
    print("=== scanning", path, "for all touches of", TARGET, "===")
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if TARGET not in line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "snapshot":
                snap = rec["data"]
                in_bids = [x for x in snap["bids"] if x[0] == TARGET]
                in_asks = [x for x in snap["asks"] if x[0] == TARGET]
                if in_bids or in_asks:
                    print("  SNAPSHOT", rec["local_timestamp"], "lastUpdateId=", snap["lastUpdateId"],
                          "in_bids=", in_bids, "in_asks=", in_asks)
                    count += 1
                continue
            if rec.get("type") != "diff":
                continue
            data = rec.get("data", {})
            if data.get("e") != "depthUpdate":
                continue
            for p, q in data.get("b", []):
                if p == TARGET:
                    print("  bid touch", rec["local_timestamp"], "U=", data["U"], "u=", data["u"], "qty=", q)
                    count += 1
            for p, q in data.get("a", []):
                if p == TARGET:
                    print("  ask touch", rec["local_timestamp"], "U=", data["U"], "u=", data["u"], "qty=", q)
                    count += 1
    print("  total touches in", day, ":", count)
