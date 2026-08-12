# resolve_width.py: resolves the 0.2798 / 0.3956 width discrepancy

import os
import statistics
import sys
import time

_D = r"C:\Users\lnico\Desktop"
if _D not in sys.path:
    sys.path.insert(0, _D)
os.chdir(_D)

from matching_engine import MatchingEngine
import sim_run
from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from clipped_mm_check import MM_ID, _resid_best


def detector(eng, seed_ids):
    """Today's detector, verbatim in behaviour"""
    bb, ba = eng.best_bid(), eng.best_ask()
    if bb is None or ba is None:
        return None
    qb, qa = eng.bids[bb], eng.asks[ba]
    return (any(o.id in seed_ids for o in qb),
            any(o.id in seed_ids for o in qa),
            all(o.id in seed_ids for o in qb),
            all(o.id in seed_ids for o in qa))


print("=" * 84)
print("Item 5; detector self-test (must pass before anything below means anything)")
print("=" * 84)

eng0 = MatchingEngine()
sim_run._seed_book(eng0)
sids0 = set(eng0.orders.keys())
print("  seed-book-only engine: %d orders, bb=%.2f ba=%.2f"
      % (len(eng0.orders), eng0.best_bid(), eng0.best_ask()))
print("  detector (present_bid, present_ask, only_bid, only_ask) =",
      detector(eng0, sids0))
print("  expect (True, True, True, True)")

eng0.add_limit_order("buy", 61999.75, 0.001, "noise")
eng0.add_limit_order("sell", 62000.25, 0.001, "noise")
print("\n  after adding non-seed inside: bb=%.2f ba=%.2f"
      % (eng0.best_bid(), eng0.best_ask()))
print("  detector =", detector(eng0, sids0))
print("  expect (False, False, False, False)")

eng2 = MatchingEngine()
sim_run._seed_book(eng2)
s2 = set(eng2.orders.keys())
eng2.add_limit_order("buy", 61999.50, 0.001, "noise")
print("\n  non-seed joining the best seed bid level: detector =", detector(eng2, s2))
print("  expect (True, True, False, True)  -- present but no longer seed-only on the bid")
print()


T = 86400.0
LIFE = 720.0
LAM, P_MARKET, DISP = 1.804, 0.0126, 0.0055
K, GAMMA = 0.17763, 1.3391837316766703e-06
SCAN_EVERY = 60
BOOK_EVERY = 600
SEEDS = [0, 1, 2, 3, 4]


def med(v):
    return statistics.median(v) if v else float("nan")


def one_seed(seed):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA, quote_size=DEFAULT_QUOTE_SIZE)

    a_1s, a_60, a_600 = [], [], []
    b_1s, b_60, b_600 = [], [], []
    c_60, c_600 = [], []
    n_sec = seed_touch = seed_only = mm_touch = 0
    last = 62000.0

    t = 0.0
    while t < T:
        t += 1.0
        ti = int(t)
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fl = getattr(r, "fills", None)
                if fl:
                    mm.on_fills(fl, r.side)
        mm.requote(eng, ti)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        last = 0.5 * (bb + ba)
        n_sec += 1
        qb, qa = eng.bids[bb], eng.asks[ba]

        d = detector(eng, seed_ids)
        if d[0] or d[1]:
            seed_touch += 1
        if d[2] or d[3]:
            seed_only += 1
        if (any(o.agent_id == MM_ID for o in qb)
                or any(o.agent_id == MM_ID for o in qa)):
            mm_touch += 1

        w_a = 1e4 * (ba - bb) / last
        a_1s.append(w_a)

        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        w_b = (1e4 * (ra - rb) / (0.5 * (rb + ra))
               if rb is not None and ra is not None and ra > rb else None)
        if w_b is not None:
            b_1s.append(w_b)

        if ti % SCAN_EVERY:
            continue

        nb, na = [], []
        for book, store in ((eng.bids, nb), (eng.asks, na)):
            for p, q in book.items():
                for o in q:
                    if o.id in seed_ids or o.agent_id == MM_ID:
                        continue
                    store.append(p)
                    break
        w_c = None
        if nb and na:
            rb2, ra2 = max(nb), min(na)
            if ra2 > rb2:
                w_c = 1e4 * (ra2 - rb2) / (0.5 * (rb2 + ra2))

        a_60.append(w_a)
        if w_b is not None:
            b_60.append(w_b)
        if w_c is not None:
            c_60.append(w_c)
        if ti % BOOK_EVERY == 0:
            a_600.append(w_a)
            if w_b is not None:
                b_600.append(w_b)
            if w_c is not None:
                c_600.append(w_c)

    alive = sum(1 for oid in seed_ids if oid in eng.orders)
    return {
        "seed": seed, "n_sec": n_sec,
        "seed_touch_pct": 100.0 * seed_touch / max(1, n_sec),
        "seed_only_pct": 100.0 * seed_only / max(1, n_sec),
        "mm_touch_pct": 100.0 * mm_touch / max(1, n_sec),
        "alive": alive,
        "a_1s": med(a_1s), "a_60": med(a_60), "a_600": med(a_600),
        "b_1s": med(b_1s), "b_60": med(b_60), "b_600": med(b_600),
        "c_60": med(c_60), "c_600": med(c_600),
        "n_a1": len(a_1s), "n_b1": len(b_1s), "n_c60": len(c_60),
        "n_c600": len(c_600),
        "bc_same_60": (b_60 == c_60),
        "ab_same_1s": (a_1s == b_1s),
    }


print("=" * 84)
print("Items 1-4; MM-present, 1 day, life=720, K=0.17763, gamma=1.339e-06, "
      "%d seeds" % len(SEEDS))
print("=" * 84)
rows = []
t0 = time.time()
for s in SEEDS:
    r = one_seed(s)
    rows.append(r)
    print("  seed %d done (%.0fs elapsed)" % (s, time.time() - t0), flush=True)
print()

print("  touch occupancy")
print("  %6s %12s %14s %14s %12s" % ("seed", "two-sided s", "seed@touch %",
                                     "seed-only %", "MM@touch %"))
for r in rows:
    print("  %6d %12d %14.4f %14.4f %12.4f"
          % (r["seed"], r["n_sec"], r["seed_touch_pct"], r["seed_only_pct"],
             r["mm_touch_pct"]))
print("  %6s %12d %14.4f %14.4f %12.4f"
      % ("pool", sum(r["n_sec"] for r in rows),
         sum(r["seed_touch_pct"] * r["n_sec"] for r in rows) / sum(r["n_sec"] for r in rows),
         sum(r["seed_only_pct"] * r["n_sec"] for r in rows) / sum(r["n_sec"] for r in rows),
         sum(r["mm_touch_pct"] * r["n_sec"] for r in rows) / sum(r["n_sec"] for r in rows)))
print("  seed orders still resting at end: %s (of 10 each)"
      % ", ".join(str(r["alive"]) for r in rows))
print()

print("  median width bp by convention and sampling rule")
print("  %6s | %-24s | %-24s | %-17s" %
      ("seed", "(a) Raw, no filter", "(b) MM-excl, seed in", "(c) MM+seed excl"))
print("  %6s | %7s %7s %7s | %7s %7s %7s | %7s %7s"
      % ("", "1s", "60s", "600s", "1s", "60s", "600s", "60s", "600s"))
for r in rows:
    print("  %6d | %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f"
          % (r["seed"], r["a_1s"], r["a_60"], r["a_600"],
             r["b_1s"], r["b_60"], r["b_600"], r["c_60"], r["c_600"]))
mn = lambda k: statistics.mean(r[k] for r in rows)
print("  %6s | %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f"
      % ("mean", mn("a_1s"), mn("a_60"), mn("a_600"),
         mn("b_1s"), mn("b_60"), mn("b_600"), mn("c_60"), mn("c_600")))
print()
print("  identity checks (exact list equality, per seed):")
print("    (b) == (c) at the 60s instants : %s"
      % ", ".join("%d:%s" % (r["seed"], r["bc_same_60"]) for r in rows))
print("    (a) == (b) every second        : %s"
      % ", ".join("%d:%s" % (r["seed"], r["ab_same_1s"]) for r in rows))
print()
print("  sample counts seed 0: a_1s=%d b_1s=%d c_60=%d c_600=%d"
      % (rows[0]["n_a1"], rows[0]["n_b1"], rows[0]["n_c60"], rows[0]["n_c600"]))
print("  total %.0fs" % (time.time() - t0))
