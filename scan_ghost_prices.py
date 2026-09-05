# scan_ghost_prices.py: read-only: are the stuck ask levels at the 08-17 re-anchor ever referenced again?

import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.getcwd())

DATA_DIR = "market_data_droplet"
SCRATCH = os.path.dirname(os.path.abspath(__file__))

STUCK_FROM = "2026-08-17T00:06:10"
STUCK_TO = "2026-08-23"

targets = []
with open(os.path.join(SCRATCH, "ghost_target.txt"), encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            targets.append(parts[1])

print("Targets (%d stuck ask levels from the 08-17 00:06:10 snapshot):" % len(targets))
for t in targets:
    print("   " + t)
print("")
print("Scanning 2026-07-15 .. 2026-08-29 for any reference to these exact price strings.")
print("(A reference after the 08-23 re-anchor is expected and harmless; the market")
print(" legitimately returns to these prices. What matters is references during the")
print(" stuck window, which would mean the apply path skipped a delivered removal.)")
print("")

tset = tuple(targets)
hits = {t: [] for t in targets}
per_day_hits = {}

d = date(2026, 7, 15)
end = date(2026, 8, 29)
while d <= end:
    p = os.path.join(DATA_DIR, "btcusd_" + d.isoformat() + ".jsonl")
    if not os.path.exists(p):
        d += timedelta(days=1)
        continue
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not any(t in line for t in tset):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ts = rec.get("local_timestamp")
            if rec.get("type") == "snapshot":
                snap = rec["data"]
                for side_key in ("bids", "asks"):
                    for price, qty in snap[side_key]:
                        if price in hits:
                            hits[price].append((d.isoformat(), ts, "SNAPSHOT",
                                                snap["lastUpdateId"], side_key, qty))
                            n += 1
                continue
            data = rec.get("data")
            if not data or data.get("e") != "depthUpdate":
                continue
            for side_key, letter in (("b", "bid"), ("a", "ask")):
                for price, qty in data.get(side_key, []):
                    if price in hits:
                        hits[price].append((d.isoformat(), ts, "diff",
                                            (data["U"], data["u"]), letter, qty))
                        n += 1
    if n:
        per_day_hits[d.isoformat()] = n
        print("  %s : %d references" % (d.isoformat(), n))
    sys.stdout.flush()
    d += timedelta(days=1)

print("")
print("=" * 78)
print("Per-price result")
print("=" * 78)
for t in targets:
    hs = hits[t]
    during = [h for h in hs if (h[0] >= "2026-08-17" and h[0] < STUCK_TO
                                and not (h[0] == "2026-08-17" and h[1] < STUCK_FROM))]
    print("")
    print("price %s : %d total references in the window, %d during the stuck period"
          % (t, len(hs), len(during)))
    if during:
        print("   *** references during the stuck window; apply path would be suspect ***")
        for h in during[:20]:
            print("     %s %s %s %s %s qty=%s" % h)
    firsts = [h for h in hs if h[0] >= "2026-08-17"]
    if firsts:
        print("   first reference at or after 08-17: %s %s %s %s qty=%s"
              % (firsts[0][0], firsts[0][1], firsts[0][2], firsts[0][4], firsts[0][5]))
    else:
        print("   no reference at or after 08-17 anywhere in the window.")
    before = [h for h in hs if h[0] < "2026-08-17"]
    print("   references before 08-17: %d" % len(before))

print("")
print("=" * 78)
print("Summary")
print("=" * 78)
tot_during = 0
for t in targets:
    hs = hits[t]
    during = [h for h in hs if (h[0] >= "2026-08-17" and h[0] < STUCK_TO
                                and not (h[0] == "2026-08-17" and h[1] < STUCK_FROM))]
    tot_during += len(during)
print("Total references to any stuck price during 08-17 00:06:10 .. 08-23: %d" % tot_during)
if tot_during == 0:
    print("=> The removals never arrived. The levels are genuinely orphaned: the feed did")
    print("   not deliver them. This is a data-delivery gap, NOT a bug in the apply path.")
else:
    print("=> Removals did arrive during the stuck window. The apply path skipped them.")
    print("   This would be a reconstruction bug, not a feed gap.")
