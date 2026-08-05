# clipped_mm_sensitivity.py: MM sensitivity to k at the clipped working point (Option 1 k)

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world, TARGET
from clipped_depth_sweep import LAM
from clipped_mm_check import _resid_best, MM_ID, ABSENT
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_SIGMA

T = 86400.0
SEEDS = list(range(15))
LIFE = 720.0
DISP = 0.0055
P_MARKET = 0.0126
SAMPLE_EVERY = 60

K_BRACKET = [(0.17763, "Option 1 near-touch $0.25-5, R2=0.9929  [PRIMARY]"),
             (0.16230, "Option 1 near-touch $0.25-10, R2=0.9954  [check]"),
             (0.12429, "Option 1 full swept range $0.25-30, R2=0.9584")]

ONE_TICK_USD = 0.01
ONE_TICK_BP = 1e4 * ONE_TICK_USD / 62000.0

K06747 = {"full_median_bp": 0.3110, "resid_median_bp": 0.3110,
          "mm_passive_share": 0.95, "mm_near_share": 0.00,
          "mm_fill_share": 2.71, "mm_vol_share": 10.04,
          "mm_fills": 143.0, "mm_q": -0.05275}

OLD_BOOK = {"fills_per_day": 4630 / 3.0, "terminal_q": -0.49915,
            "sd_q": 0.23454, "vol_share": 65.84, "orders_side": 17.5}


def run(seed, k):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=k)

    n_fill = n_mm_fill = 0
    vol_total = vol_mm = 0.0
    full_bp, resid_bp = [], []
    q_series, skew_usd, skew_bp = [], [], []
    mm_passive_share, mm_near_share = [], []

    t = 0.0
    while t < T:
        t += 1.0
        for flow_recs in (noise.run_until(eng, t),
                          informed.run_until(eng, t, vf)):
            for r in flow_recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                n_fill += len(fills)
                for f in fills:
                    vol_total += f.size
                    if f.counterparty_id == MM_ID:
                        n_mm_fill += 1
                        vol_mm += f.size
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mid = 0.5 * (bb + ba)
        full_bp.append(1e4 * (ba - bb) / mid)
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is not None and ra is not None and ra > rb:
            resid_bp.append(1e4 * (ra - rb) / (0.5 * (rb + ra)))

        q_series.append(mm.q)
        sk = mm.inventory_skew(mid, t)
        skew_usd.append(sk)
        skew_bp.append(1e4 * sk / mid)

        if t % SAMPLE_EVERY:
            continue
        sz_all = sz_mm = near_all = near_mm = 0.0
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        for book, is_bid in ((eng.bids, True), (eng.asks, False)):
            for p, q in book.items():
                near = (p >= lo) if is_bid else (p <= hi)
                for o in q:
                    if o.id in seed_ids:
                        continue
                    sz_all += o.size
                    if o.agent_id == MM_ID:
                        sz_mm += o.size
                    if near:
                        near_all += o.size
                        if o.agent_id == MM_ID:
                            near_mm += o.size
        if sz_all > 0:
            mm_passive_share.append(100.0 * sz_mm / sz_all)
        if near_all > 0:
            mm_near_share.append(100.0 * near_mm / near_all)

    fs, rs = sorted(full_bp), sorted(resid_bp)
    return {
        "full_median_bp": fs[len(fs) // 2] if fs else float("nan"),
        "resid_median_bp": rs[len(rs) // 2] if rs else float("nan"),
        "mm_passive_share": statistics.mean(mm_passive_share) if mm_passive_share else 0.0,
        "mm_near_share": statistics.mean(mm_near_share) if mm_near_share else 0.0,
        "mm_fill_share": 100.0 * n_mm_fill / max(1, n_fill),
        "mm_vol_share": 100.0 * vol_mm / vol_total if vol_total else 0.0,
        "mm_fills": float(mm.n_buy_fills + mm.n_sell_fills),
        "terminal_q": mm.q,
        "sd_q": statistics.stdev(q_series) if len(q_series) > 1 else 0.0,
        "max_abs_q": max(abs(x) for x in q_series) if q_series else 0.0,
        "mean_skew_usd": statistics.mean(skew_usd),
        "sd_skew_usd": statistics.stdev(skew_usd) if len(skew_usd) > 1 else 0.0,
        "mean_skew_bp": statistics.mean(skew_bp),
        "sd_skew_bp": statistics.stdev(skew_bp) if len(skew_bp) > 1 else 0.0,
        "max_abs_skew_usd": max(abs(x) for x in skew_usd) if skew_usd else 0.0,
    }


def measure(k, seeds=SEEDS):
    per = None
    t0 = time.time()
    for s in seeds:
        x = run(s, k)
        if per is None:
            per = {key: [] for key in x}
        for key, v in x.items():
            per[key].append(v)
    n = len(seeds)
    out = {"k": k, "secs": time.time() - t0}
    for key, v in per.items():
        out[key] = statistics.mean(v)
        out[key + "_se"] = (statistics.stdev(v) / (n ** 0.5)) if n > 1 else 0.0
    return out


def as_decomposition(k):
    """Which as term sets the MM's width? Uses the MM's own methods so the"""
    mm = MarketMaker(horizon=T, k=k)
    mid = 62000.0
    rows = []
    for t in (0.0, T / 2.0, T):
        sig = mm.sigma_absolute(mid)
        tau = mm.time_remaining_years(t)
        inv = mm.gamma * sig * sig * tau
        liq = mm.total_spread(mid, t) - inv
        rows.append((t, inv, liq, mm.total_spread(mid, t)))
    return rows


def main():
    print("=" * 124)
    print("k sensitivity; not A CALIBRATION. No k is chosen; a bracket is "
          "reported.")
    print("=" * 124)
    print("Working point: lam=%.4f  p_market=%.4f  mean_lifetime=%.0fs  "
          "disp=%.4f  JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x %.0fs. gamma=DEFAULT_GAMMA=%g (not chosen). DEFAULT_K "
          "never edited." % (len(SEEDS), T, DEFAULT_GAMMA))
    print("")
    print("k values are the section 3.6 Option 1 measurements (the MM's own "
          "quotes, its own fills,")
    print("distance set not inferred). The earlier measure_k_clipped.py values "
          "are NOT used: that")
    print("estimator is unresolvable near the touch; not flat; because it "
          "re-attributes distance")
    print("once per second while the mid's 1-second sd is $4.11, blurring "
          "every fill by ~+/-$4.")
    print("Measured properly the same region fits at R2 = 0.9929. The "
          "log-hazard is convex, so no")
    print("single k describes the curve: this is a bracket and cannot become "
          "a calibration.")
    print("")

    print("=" * 124)
    print("As spread decomposition; does k or inventory risk set the MM's "
          "width?")
    print("=" * 124)
    print("  total_spread = gamma*sigma^2*(T-t)   +   (2/gamma)*ln(1+gamma/k)")
    print("                 [inventory term]          [liquidity term ~ 2/k]")
    print("  at mid=$62,000, gamma=%g, sigma=%.4f, horizon=%.0fs"
          % (DEFAULT_GAMMA, DEFAULT_SIGMA, T))
    print("")
    print("  %-10s %6s %14s %14s %14s %12s"
          % ("k", "t", "inventory $", "liquidity $", "total $", "liq share"))
    for k, _lab in K_BRACKET + [(0.06747, "old book")]:
        for t, inv, liq, tot in as_decomposition(k):
            print("  %-10.5f %6.0f %14.4f %14.4f %14.4f %11.2f%%"
                  % (k, t, inv, liq, tot, 100.0 * liq / tot if tot else 0.0))
        print("")

    results = {}
    for k, label in K_BRACKET:
        print("=" * 124)
        print("k = %.5f  -- %s" % (k, label))
        print("=" * 124)
        r = measure(k)
        results[k] = r
        print("  measured in %.0fs" % r["secs"], flush=True)
        print("")

    print("=" * 124)
    print("Bracket vs k=0.06747 vs the old 17.5-order book")
    print("=" * 124)
    ks = [0.06747] + [k for k, _ in K_BRACKET]

    def get(k, key, default=None):
        if k == 0.06747:
            return K06747.get(key, default)
        return results[k].get(key, default)

    rows = [
        ("full-book touch (bp)", "full_median_bp", "%10.4f"),
        ("residual touch (bp)", "resid_median_bp", "%10.4f"),
        ("MM passive share, all (%)", "mm_passive_share", "%10.2f"),
        ("MM passive share <=1bp (%)", "mm_near_share", "%10.2f"),
        ("MM fill-count share (%)", "mm_fill_share", "%10.2f"),
        ("MM volume share (%)", "mm_vol_share", "%10.2f"),
        ("MM fills per day", "mm_fills", "%10.1f"),
        ("terminal q (BTC)", "terminal_q", "%10.5f"),
        ("sd of q over the run (BTC)", "sd_q", "%10.5f"),
        ("max |q| (BTC)", "max_abs_q", "%10.5f"),
        ("mean skew ($)", "mean_skew_usd", "%10.5f"),
        ("sd of skew ($)", "sd_skew_usd", "%10.5f"),
        ("sd of skew (bp)", "sd_skew_bp", "%10.6f"),
        ("max |skew| ($)", "max_abs_skew_usd", "%10.5f"),
    ]
    print("  %-30s %s"
          % ("", " ".join("%12s" % ("k=%.5f" % k) for k in ks)))
    print("  " + "-" * (30 + 13 * len(ks)))
    for label, key, fmt in rows:
        cells = []
        for k in ks:
            v = get(k, key)
            cells.append("-" if v is None else (fmt % v).strip())
        print("  %-30s %s" % (label, " ".join("%12s" % c for c in cells)))
    print("")
    print("  one tick = $%.2f = %.6f bp; the floor below which the quote "
          "centre does not move" % (ONE_TICK_USD, ONE_TICK_BP))
    print("  a full price increment, and no observer can read it however "
          "sophisticated.")
    print("  %-30s %s"
          % ("sd(skew) in ticks", " ".join(
              "%12s" % ("-" if get(k, "sd_skew_usd") is None
                        else "%.3f" % (get(k, "sd_skew_usd") / ONE_TICK_USD))
              for k in ks)))

    print("")
    print("  old 17.5-orders/side book (gamma sweep, k=0.06747, 3 days):")
    print("    fills/day %.0f    terminal q %+.5f    sd(q) %.5f    volume "
          "share %.1f%%"
          % (OLD_BOOK["fills_per_day"], OLD_BOOK["terminal_q"],
             OLD_BOOK["sd_q"], OLD_BOOK["vol_share"]))
    print("  MM-ABSENT baseline for the touch (847a7fa): %.4f bp"
          % ABSENT["median_bp"])

    print("")
    print("=" * 124)
    print("Signal check; is there enough inventory variation for a sniffer "
          "to read?")
    print("=" * 124)
    print("  The leak is skew = -q*gamma*sigma^2*(T-t). Its scale is set by "
          "sd(q).")
    print("  Old book sd(q) = %.5f BTC. Bracket:" % OLD_BOOK["sd_q"])
    for k in ks:
        sdq = get(k, "sd_q")
        sdsk = get(k, "sd_skew_usd")
        if sdq is None:
            print("    k=%.5f  sd(q) not measured in that run" % k)
            continue
        print("    k=%.5f  sd(q) = %.5f BTC (%.3fx the old book)   sd(skew) = "
              "$%.5f = %.6f bp"
              % (k, sdq, sdq / OLD_BOOK["sd_q"], sdsk, get(k, "sd_skew_bp")))
    print("")
    print("  the floor that matters: one tick = $%.2f = %.6f bp. If sd(skew) "
          "is below a tick the" % (ONE_TICK_USD, ONE_TICK_BP))
    print("  quote centre does not move a full price increment, so no "
          "observer; however")
    print("  sophisticated; can read it, and a null Phase 6 result would be "
          "uninterpretable.")
    print("")
    print("  %-14s %14s %14s %12s" % ("k", "sd(skew) $", "in ticks", "verdict"))
    for k in ks:
        sdsk = get(k, "sd_skew_usd")
        if sdsk is None:
            print("  %-14.5f %14s %14s %12s" % (k, "-", "-", "not measured"))
            continue
        ticks = sdsk / ONE_TICK_USD
        print("  %-14.5f %14.5f %14.3f %12s"
              % (k, sdsk, ticks,
                 "ABOVE floor" if ticks >= 1.0 else "BELOW floor"))
    print("")
    print("  Reported, not adjudicated; but the tick floor is a hard "
          "observability limit, not a")
    print("  modelling preference: a sub-tick quote-centre movement is not "
          "representable on the book.")


if __name__ == "__main__":
    main()
