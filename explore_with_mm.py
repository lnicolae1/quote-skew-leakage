# explore_with_mm.py: exploratory run: two-sidedness with the market maker present

import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker

T = 86400.0
dt = 1.0
n_steps = int(T / dt)
SEED_V, SEED_NOISE, SEED_INF = 31415, 1001, 2002


def seed_book(eng, mid=62000.0):
    for i in range(5):
        eng.add_limit_order("buy", round(mid - (i + 1) * 0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(mid + (i + 1) * 0.5, 2), 0.02, "seed")


def run(with_mm):
    path = generate_value_path(0.3721, 62000.0, dt, n_steps, seed=SEED_V)
    value_fn = path_value(path, dt)

    eng = MatchingEngine()
    seed_book(eng)

    noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=SEED_NOISE,
                            value_fn=value_fn, reference_price=62000.0)
    informed = InformedTraderFlow(seed=SEED_INF)
    mm = MarketMaker(horizon=T) if with_mm else None

    track = []
    recs = []
    t = 0.0
    while t < T:
        t += 1.0
        for r in noise.run_until(eng, t):
            if r.fills and mm is not None:
                mm.on_fills(r.fills, r.side)
        for r in informed.run_until(eng, t, value_fn):
            recs.append(r)
            if r.fills and mm is not None:
                mm.on_fills(r.fills, r.side)
        if mm is not None:
            mm.requote(eng, t)
        track.append((t, eng.best_bid(), eng.best_ask(), eng.mid(),
                      value_fn(t), mm.q if mm is not None else 0.0))

    return path, eng, noise, informed, mm, recs, track


def summarise(tag, path, eng, noise, informed, mm, recs, track):
    print("=" * 74)
    print(tag)
    print("=" * 74)

    n = len(track)
    two = sum(1 for r in track if r[1] is not None and r[2] is not None)
    no_bid = sum(1 for r in track if r[1] is None)
    no_ask = sum(1 for r in track if r[2] is None)
    devs = [abs(r[3] - r[4]) for r in track if r[3] is not None]

    print("(1) Two-sidedness")
    print("    samples: %d   two-sided: %d (%.2f%%)" % (n, two, 100.0 * two / n))
    print("    no bid: %d   no ask: %d" % (no_bid, no_ask))
    print("")

    print("(2) Tracking error")
    print("    mean   |mid - V| = $%.2f" % statistics.mean(devs))
    print("    median |mid - V| = $%.2f" % statistics.median(devs))
    print("    final  |mid - V| = $%.2f" % devs[-1])
    print("")

    print("(3) Market maker")
    if mm is None:
        print("    (no MM in this run)")
    else:
        print("    requotes=%d  buy_fills=%d  sell_fills=%d  total=%d"
              % (mm.n_requotes, mm.n_buy_fills, mm.n_sell_fills,
                 mm.n_buy_fills + mm.n_sell_fills))
        qs = [r[5] for r in track]
        crossings = sum(1 for i in range(len(qs) - 1)
                        if (qs[i] > 0) != (qs[i + 1] > 0))
        print("    q: mean %+.5f  sd %.5f  min %+.5f  max %+.5f  final %+.5f BTC"
              % (statistics.mean(qs), statistics.pstdev(qs), min(qs), max(qs),
                 qs[-1]))
        print("    q sign changes: %d   |q| never moved off 0: %s"
              % (crossings, "yes (dead)" if max(abs(x) for x in qs) == 0
                 else "no"))
        s = qs[::60]
        if len(s) > 10:
            mu = statistics.mean(s)
            x = [v - mu for v in s]
            num = sum(x[i] * x[i + 1] for i in range(len(x) - 1))
            den = sum(v * v for v in x[:-1])
            if den > 0:
                phi = num / den
                print("    AR(1) phi on 60s-sampled q = %.4f" % phi)
                if 0 < phi < 1:
                    import math
                    print("    => mean-reverting, half-life %.1f min"
                          % (math.log(0.5) / math.log(phi)))
                elif phi >= 1:
                    print("    => NOT mean-reverting (random walk or drifting)")
        print("    cash %.2f   mark-to-market PnL %.2f"
              % (mm.cash, mm.mark_to_market(track[-1][3] or path[-1])))
    print("")

    print("(4) Book vs V across the run")
    print("    %-7s %-7s %11s %11s %11s %11s %9s %11s %s"
          % ("pct", "t(s)", "best_bid", "best_ask", "mid", "V", "mid-V",
             "MM q", "two-sided?"))
    for pct, idx in (("start", 0), ("25%", n // 4), ("50%", n // 2),
                     ("75%", 3 * n // 4), ("end", n - 1)):
        tt, bb, ba, m, v, q = track[idx]
        ok = bb is not None and ba is not None
        print("    %-7s %-7.0f %11s %11s %11s %11.2f %9s %+11.5f %s"
              % (pct, tt,
                 "%.2f" % bb if bb is not None else "None",
                 "%.2f" % ba if ba is not None else "None",
                 "%.2f" % m if m is not None else "None", v,
                 "%.2f" % (m - v) if m is not None else "n/a", q,
                 "yes" if ok else "*** no ***"))
    print("    V:   %.2f -> %.2f (moved %+.2f)"
          % (path[0], path[-1], path[-1] - path[0]))
    mids = [r[3] for r in track if r[3] is not None]
    print("    mid: %.2f -> %.2f (moved %+.2f)"
          % (mids[0], mids[-1], mids[-1] - mids[0]))
    print("")

    print("(5) Resting orders at end (agent_id prefix, case-insensitive)")
    out = {}
    for o in eng.orders.values():
        a = str(o.agent_id).lower()
        key = ("noise" if a.startswith("noise") else
               "informed" if a.startswith("informed") else
               "mm" if a.startswith("mm") else
               "seed" if a.startswith("seed") else "other")
        out[key] = out.get(key, 0) + 1
    print("    " + "  ".join("%s=%d" % (k, v) for k, v in sorted(out.items()))
          + "   (total %d)" % len(eng.orders))
    spreads = [r[2] - r[1] for r in track
               if r[1] is not None and r[2] is not None]
    ss = sorted(spreads)
    print("    median touch spread $%.2f (%.2f bp)   p25 $%.2f  p75 $%.2f"
          % (ss[len(ss) // 2], 1e4 * ss[len(ss) // 2] / path[-1],
             ss[len(ss) // 4], ss[3 * len(ss) // 4]))
    print("")
    return 100.0 * two / n, statistics.mean(devs)


if __name__ == "__main__":
    ts_no, dev_no = summarise("NO-MM CONTROL (baseline for comparison)",
                              *run(with_mm=False))
    ts_mm, dev_mm = summarise("WITH MARKET MAKER (the real configuration)",
                              *run(with_mm=True))
    print("=" * 74)
    print("Side by side")
    print("=" * 74)
    print("    %%two-sided : no-MM %.2f%%   with-MM %.2f%%" % (ts_no, ts_mm))
    print("    mean|mid-V|: no-MM $%.2f   with-MM $%.2f" % (dev_no, dev_mm))
