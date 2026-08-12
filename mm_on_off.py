"""mm_on_off.py: MM on vs off; seed touch share, spreads, seed-order fills by aggressor.
MM marketable requotes caught via an add_limit_order hook.
"""
import os
import statistics
import sys
import time

_D = r"C:\Users\lnico\Desktop"
if _D not in sys.path:
    sys.path.insert(0, _D)
os.chdir(_D)

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from clipped_mm_check import MM_ID

T = 86400.0
LIFE = 720.0
LAM, P_MARKET, DISP = 1.804, 0.0126, 0.0055
K, GAMMA = 0.17763, 1.3391837316766703e-06
SEEDS = [0, 1, 2, 3, 4]
SEED_TAG = "seed"


def one(seed, use_mm):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = (MarketMaker(horizon=T, k=K, gamma=GAMMA,
                      quote_size=DEFAULT_QUOTE_SIZE) if use_mm else None)

    # fills, BTC taken from seed orders
    kill = {"noise": [0, 0.0], "informed": [0, 0.0], "mm": [0, 0.0]}

    _orig_add = eng.add_limit_order

    def _add(side, price, size, agent_id):
        res = _orig_add(side, price, size, agent_id)
        if agent_id == MM_ID:
            for f in res.fills:
                if f.counterparty_id == SEED_TAG:
                    kill["mm"][0] += 1
                    kill["mm"][1] += f.size
        return res
    eng.add_limit_order = _add

    n_sec = seed_touch = mm_touch = 0
    raw, both = [], []
    t = 0.0
    while t < T:
        t += 1.0
        ti = int(t)
        for recs, tag in ((noise.run_until(eng, t), "noise"),
                          (informed.run_until(eng, t, vf), "informed")):
            for r in recs:
                fl = getattr(r, "fills", None)
                if not fl:
                    continue
                for f in fl:
                    if f.counterparty_id == SEED_TAG:
                        kill[tag][0] += 1
                        kill[tag][1] += f.size
                if mm is not None:
                    mm.on_fills(fl, r.side)
        if mm is not None:
            mm.requote(eng, ti)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        n_sec += 1
        qb, qa = eng.bids[bb], eng.asks[ba]
        if any(o.id in seed_ids for o in qb) or any(o.id in seed_ids for o in qa):
            seed_touch += 1
        if (any(o.agent_id == MM_ID for o in qb)
                or any(o.agent_id == MM_ID for o in qa)):
            mm_touch += 1
        raw.append(1e4 * (ba - bb) / (0.5 * (bb + ba)))

        if ti % 60:
            continue
        nb, na = [], []
        for book, store in ((eng.bids, nb), (eng.asks, na)):
            for p, q in book.items():
                for o in q:
                    if o.id in seed_ids or o.agent_id == MM_ID:
                        continue
                    store.append(p)
                    break
        if nb and na:
            rb, ra = max(nb), min(na)
            if ra > rb:
                both.append(1e4 * (ra - rb) / (0.5 * (rb + ra)))

    alive = sum(1 for oid in seed_ids if oid in eng.orders)
    return {
        "seed_touch": 100.0 * seed_touch / max(1, n_sec),
        "mm_touch": 100.0 * mm_touch / max(1, n_sec),
        "alive": alive,
        "raw": statistics.median(raw),
        "excl": statistics.median(both),
        "kn": kill["noise"][0], "kn_btc": kill["noise"][1],
        "ki": kill["informed"][0], "ki_btc": kill["informed"][1],
        "km": kill["mm"][0], "km_btc": kill["mm"][1],
    }


t0 = time.time()
KEYS = ("seed_touch", "mm_touch", "alive", "raw", "excl")
out = {}
print("  %-8s %5s %13s %11s %7s %11s %12s"
      % ("arm", "seed", "seed@touch %", "MM@touch %", "alive", "raw med bp",
         "seed-excl bp"))
for use_mm, tag in ((True, "MM ON"), (False, "MM OFF")):
    rows = []
    for s in SEEDS:
        r = one(s, use_mm)
        rows.append(r)
        print("  %-8s %5d %13.4f %11.4f %7d %11.4f %12.4f"
              % (tag, s, r["seed_touch"], r["mm_touch"], r["alive"],
                 r["raw"], r["excl"]), flush=True)
    print("  %-8s %5s %13.4f %11.4f %7.1f %11.4f %12.4f"
          % (tag, "MEAN", *[statistics.mean(r[k] for r in rows) for k in KEYS]))
    out[tag] = rows
    print()

print("=" * 78)
print("seed-order fills by aggressor")
print("=" * 78)
print("  %-8s %5s %10s %10s %10s %14s %8s"
      % ("arm", "seed", "by noise", "by inf", "by MM", "total BTC", "alive"))
for tag in ("MM ON", "MM OFF"):
    for s, r in zip(SEEDS, out[tag]):
        print("  %-8s %5d %10d %10d %10d %14.5f %8d"
              % (tag, s, r["kn"], r["ki"], r["km"],
                 r["kn_btc"] + r["ki_btc"] + r["km_btc"], r["alive"]))
    rows = out[tag]
    print("  %-8s %5s %10d %10d %10d %14.5f %8d"
          % (tag, "TOTAL", sum(r["kn"] for r in rows), sum(r["ki"] for r in rows),
             sum(r["km"] for r in rows),
             sum(r["kn_btc"] + r["ki_btc"] + r["km_btc"] for r in rows),
             sum(r["alive"] for r in rows)))
    print("  %-8s seed BTC at start = %.5f per run (10 x 0.02)" % (tag, 0.2))
    print()
print("  total %.0fs" % (time.time() - t0))
