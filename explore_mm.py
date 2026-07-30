# explore_mm.py: exploratory run: price formation without a market maker

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
    """Thin bootstrap book so a mid exists at t=0"""
    for i in range(5):
        eng.add_limit_order("buy", round(mid - (i + 1) * 0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(mid + (i + 1) * 0.5, 2), 0.02, "seed")


def resting_counts(eng):
    """Resting orders at end, split by who owns them"""
    out = {"noise": 0, "informed": 0, "mm": 0, "seed": 0, "other": 0}
    for o in eng.orders.values():
        a = str(o.agent_id)
        if a.startswith("noise"):
            out["noise"] += 1
        elif a.startswith("informed"):
            out["informed"] += 1
        elif a.startswith("mm"):
            out["mm"] += 1
        elif a.startswith("seed"):
            out["seed"] += 1
        else:
            out["other"] += 1
    return out


def run(with_mm):
    path = generate_value_path(0.3721, 62000.0, dt, n_steps, seed=SEED_V)
    value_fn = path_value(path, dt)

    eng = MatchingEngine()
    seed_book(eng)

    noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=SEED_NOISE,
                            value_fn=value_fn, reference_price=62000.0)
    informed = InformedTraderFlow(seed=SEED_INF)
    mm = MarketMaker(horizon=T) if with_mm else None

    recs = []
    track = []
    t = 0.0
    while t < T:
        t += 1.0
        noise.run_until(eng, t)
        recs.extend(informed.run_until(eng, t, value_fn))
        if mm is not None:
            mm.requote(eng, t)
        track.append((t, eng.best_bid(), eng.best_ask(), eng.mid(), value_fn(t)))

    return path, eng, noise, informed, mm, recs, track


def report(tag, path, eng, noise, informed, mm, recs, track):
    print("=" * 70)
    print(tag)
    print("=" * 70)

    print("informed config: lambda=%.6f/s (toxicity %.2f)  threshold=$%.2f  "
          "size=%.5f BTC  take_fraction=%.2f"
          % (informed.lam_informed, informed.toxicity_ratio,
             informed.threshold, informed.size, informed.take_fraction))
    print("")

    devs = [abs(m - v) for (_t, _bb, _ba, m, v) in track if m is not None]
    print("(1) Tracking error")
    print("    mean   |mid - V| = $%.2f      [take-only baseline: $1155.90]"
          % statistics.mean(devs))
    print("    median |mid - V| = $%.2f      [take-only baseline: $1270.42]"
          % statistics.median(devs))
    print("    final  |mid - V| = $%.2f      [take-only baseline: $1799.28]"
          % devs[-1])
    print("")

    print("(2) Book vs V across the run")
    print("    %-8s %-6s %12s %12s %12s %12s %10s  %s"
          % ("pct", "t(s)", "best_bid", "best_ask", "mid", "V", "mid-V",
             "two-sided?"))
    n = len(track)
    two_sided_all = True
    for pct, idx in (("start", 0), ("25%", n // 4), ("50%", n // 2),
                     ("75%", 3 * n // 4), ("end", n - 1)):
        tt, bb, ba, m, v = track[idx]
        ok = (bb is not None) and (ba is not None)
        two_sided_all = two_sided_all and ok
        print("    %-8s %-6.0f %12s %12s %12s %12.2f %10s  %s"
              % (pct, tt,
                 "%.2f" % bb if bb is not None else "None",
                 "%.2f" % ba if ba is not None else "None",
                 "%.2f" % m if m is not None else "None",
                 v,
                 "%.2f" % (m - v) if m is not None else "n/a",
                 "yes" if ok else "*** No; empty side ***"))
    report.two_sided_all = two_sided_all
    print("    V:   %.2f -> %.2f  (moved %+.2f)"
          % (path[0], path[-1], path[-1] - path[0]))
    mids = [m for (_t, _bb, _ba, m, _v) in track if m is not None]
    print("    mid: %.2f -> %.2f  (moved %+.2f)   [over samples where a mid "
          "existed]" % (mids[0], mids[-1], mids[-1] - mids[0]))
    n_no_mid = sum(1 for (_t, _bb, _ba, m, _v) in track if m is None)
    n_no_ask = sum(1 for (_t, _bb, ba, _m, _v) in track if ba is None)
    n_no_bid = sum(1 for (_t, bb, _ba, _m, _v) in track if bb is None)
    print("    samples with no mid: %d/%d (%.1f%%)   no ask: %d   no bid: %d"
          % (n_no_mid, len(track), 100.0 * n_no_mid / len(track),
             n_no_ask, n_no_bid))
    bids = [bb for (_t, bb, _ba, _m, _v) in track if bb is not None]
    asks = [ba for (_t, _bb, ba, _m, _v) in track if ba is not None]
    print("    best_bid range over run: %.2f .. %.2f" % (min(bids), max(bids)))
    print("    best_ask range over run: %.2f .. %.2f" % (min(asks), max(asks)))
    print("")

    traded = [r for r in recs if r.side is not None]
    passed = [r for r in recs if r.side is None]
    takes = [r for r in traded if r.action == "take"]
    posts = [r for r in traded if r.action == "post"]
    print("(3) Informed activity")
    print("    noise arrivals=%d   informed arrivals=%d"
          % (noise.n_arrivals, informed.n_arrivals))
    print("    acted=%d  passed=%d   -> trade rate %.1f%%   "
          "[take-only baseline: 100.0%%]"
          % (len(traded), len(passed),
             100.0 * len(traded) / max(1, len(recs))))
    print("    of those acting: %d took, %d posted" % (len(takes), len(posts)))
    if traded:
        print("    mean edge when acting: $%.2f   max $%.2f   "
              "[take-only baseline: mean $1150.47]"
              % (sum(r.edge for r in traded) / len(traded),
                 max(r.edge for r in traded)))
    if posts:
        rested = [r for r in posts if r.resting_size > 0]
        crossed = [r for r in posts if r.fills]
        print("    posts that crossed on arrival: %d/%d (%.1f%%)"
              % (len(crossed), len(posts), 100.0 * len(crossed) / len(posts)))
        print("    posts that rested any size    : %d/%d (%.1f%%)"
              % (len(rested), len(posts), 100.0 * len(rested) / len(posts)))
    print("")

    rc = resting_counts(eng)
    print("(4) Resting orders at end")
    print("    noise=%d  informed=%d  mm=%d  seed=%d  other=%d  (total %d)"
          % (rc["noise"], rc["informed"], rc["mm"], rc["seed"], rc["other"],
             len(eng.orders)))
    print("")

    v_end = path[-1]
    print("(5) Stale liquidity BELOW V at end  (V = %.2f)" % v_end)
    if eng.bids:
        lowest_bid = min(eng.bids)
        highest_bid = max(eng.bids)
        stale = [o for o in eng.orders.values()
                 if o.side == "buy" and o.price < v_end - 50.0]
        stale_size = sum(o.size for o in stale)
        print("    lowest  resting bid : %.2f   ($%.2f BELOW V)"
              % (lowest_bid, v_end - lowest_bid))
        print("    highest resting bid : %.2f   ($%.2f from V)"
              % (highest_bid, v_end - highest_bid))
        print("    resting bids more than $50 below V: %d orders, %.5f BTC"
              % (len(stale), stale_size))
        print("    -> the touch is pinned by stale liquidity only if the "
              "highest bid is far below V; low bids just sit harmlessly "
              "beneath it.")
    else:
        print("    bid side empty")
    if eng.asks:
        stale_a = [o for o in eng.orders.values()
                   if o.side == "sell" and o.price < v_end - 50.0]
        print("    lowest resting ask  : %.2f   (stale asks >$50 below V: %d)"
              % (min(eng.asks), len(stale_a)))
    else:
        print("    ask side empty")
    print("")

    spreads = [ba - bb for (_t, bb, ba, _m, _v) in track
               if bb is not None and ba is not None]
    if spreads:
        ss = sorted(spreads)
        med = ss[len(ss) // 2]
        med_bp = 1e4 * med / v_end
        print("(6) Touch spread  (disp = %.5f)" % noise.disp)
        print("    median spread $%.2f (%.2f bp)   p25 $%.2f   p75 $%.2f"
              % (med, med_bp, ss[len(ss) // 4], ss[3 * len(ss) // 4]))
        print("    Phase 3 real median relative spread: 0.29 bp (~$1.80) "
              "-- disp is a 4.5 calibration target, NOT tuned yet.")
    print("")

    if mm is not None:
        print("    MM: requotes=%d buy_fills=%d sell_fills=%d q=%.6f cash=%.2f"
              % (mm.n_requotes, mm.n_buy_fills, mm.n_sell_fills, mm.q, mm.cash))
        print("")

    return statistics.mean(devs), 100.0 * len(traded) / max(1, len(recs)), track


if __name__ == "__main__":
    args = run(with_mm=False)
    mean_dev, trade_rate, track = report("no-MM CONTROL; price formation "
                                         "test (informed now take + post)",
                                         *args)

    print("=" * 70)
    print("PASS criteria (no-MM control)")
    print("=" * 70)
    a = mean_dev < 50.0
    print("  (a) mean |mid - V| collapses to spread scale (<$50): %s  "
          "-- $%.2f vs $1155.90 before" % ("PASS" if a else "FAIL", mean_dev))
    b = report.two_sided_all
    print("  (b) book two-sided at every checkpoint: %s"
          % ("PASS" if b else "FAIL"))
    print("      (informed trade fraction this run: %.1f%%)" % trade_rate)
    pts = [(m, v) for (_t, _bb, _ba, m, v) in track if m is not None]
    if len(pts) < 2:
        pts = [(0.0, 0.0), (0.0, 0.0)]
    dm = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
    dv = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
    total_mid_move = abs(pts[-1][0] - pts[0][0])
    total_v_move = abs(pts[-1][1] - pts[0][1])
    frac = total_mid_move / total_v_move if total_v_move else 0.0
    c = frac > 0.8
    print("  (c) mid visibly tracks V: %s  -- mid moved %.2f vs V %.2f "
          "(%.1f%% of V's move)"
          % ("PASS" if c else "FAIL", total_mid_move, total_v_move,
             100.0 * frac))
    print("")

    print("(for continuity) with-MM run:")
    report("With MM", *run(with_mm=True))
