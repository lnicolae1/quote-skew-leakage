# resting_size.py: near-touch resting size per price level, simulator vs real

import array
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_depth_sweep import run_measure
from compute_volatility import State, stream_reconstruct, build_cloud_path
from emergent_book import (make_world_emergent, WARMUP, SAMPLE_EVERY, T_DAY,
                           SEEDS, LAM, P_MARKET, LIFE, DISP, CLIP)
from noise_traders import DEFAULT_SIZES_PATH

REAL_DAYS = ["07-11", "07-12", "07-13"]
REAL_SAMPLE_EVERY = 2000
BAND = 1e-4

REAL_NEAR_BTC_BID = 0.0452
REAL_NEAR_BTC_ASK = 0.0247


def pct(sorted_vals, p):
    """lifetime_sweep.pct's convention"""
    if not sorted_vals:
        return float("nan")
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def describe(vals):
    """median/mean/p25/p75/p95/n for one pooled sample of a quantity"""
    if not vals:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "p95": float("nan")}
    s = sorted(vals)
    return {"n": len(s), "median": pct(s, 0.50), "mean": statistics.mean(s),
            "p25": pct(s, 0.25), "p75": pct(s, 0.75), "p95": pct(s, 0.95)}


def safe_ratio(a, b):
    return a / b if b else float("nan")


class SizeSampler:
    """Scans the book every SAMPLE_EVERY seconds after the warm-up, collecting"""

    def __init__(self, eng, noise, informed, warmup):
        self.eng = eng
        self.warmup = warmup
        self.level_sizes = {"bid": [], "ask": []}
        self.orders_per_level = {"bid": [], "ask": []}
        self.order_sizes = {"bid": [], "ask": []}
        self.n_scans = 0
        self._orig_inf = informed.run_until
        informed.run_until = self._hook

    def scan(self, engine):
        """One book scan"""
        bb, ba = engine.best_bid(), engine.best_ask()
        if bb is None or ba is None:
            return False
        mid = 0.5 * (bb + ba)
        lo, hi = mid * (1 - BAND), mid * (1 + BAND)
        self.n_scans += 1
        for side, book, is_bid in (("bid", engine.bids, True),
                                   ("ask", engine.asks, False)):
            for p, q in book.items():
                if not ((p >= lo) if is_bid else (p <= hi)):
                    continue
                live = list(q)
                if not live:
                    continue
                self.level_sizes[side].append(sum(o.size for o in live))
                self.orders_per_level[side].append(len(live))
                for o in live:
                    self.order_sizes[side].append(o.size)
        return True

    def _hook(self, engine, t_end, value_fn):
        recs = self._orig_inf(engine, t_end, value_fn)
        if t_end > self.warmup and not (t_end % SAMPLE_EVERY):
            self.scan(engine)
        return recs


def run_sim_seed(seed):
    t_total = WARMUP + T_DAY
    _p, vf, eng, noise, informed = make_world_emergent(
        seed, t_total, LAM, P_MARKET, LIFE, DISP, CLIP, seed_book=False)
    smp = SizeSampler(eng, noise, informed, WARMUP)
    run_measure(vf, eng, noise, informed, T_DAY, SAMPLE_EVERY, warmup=WARMUP)
    return smp


def real_scan(state, acc):
    """One real-book scan"""
    bb, ba = state.bids.best(), state.asks.best()
    if bb is None or ba is None:
        return False
    bbf, baf = float(bb), float(ba)
    if bbf >= baf:
        return False
    mid = 0.5 * (bbf + baf)
    lo, hi = mid * (1 - BAND), mid * (1 + BAND)
    for side, book, is_bid in (("bid", state.bids.book, True),
                               ("ask", state.asks.book, False)):
        for p_str, q_str in book.items():
            pf = float(p_str)
            if (pf >= lo) if is_bid else (pf <= hi):
                acc[side].append(float(q_str))
    return True


def run_real():
    state = State()
    gaps = []
    acc = {"bid": [], "ask": []}
    ctr = [0, 0]

    def on_event(ts, uid, mid_ignored):
        ctr[0] += 1
        if ctr[0] % REAL_SAMPLE_EVERY:
            return
        if real_scan(state, acc):
            ctr[1] += 1

    nsnap = ndiff = 0
    for day in REAL_DAYS:
        counts = stream_reconstruct(build_cloud_path(day), state, gaps,
                                    on_event)
        nsnap += counts["snapshots"]
        ndiff += counts["diffs"]

    if ctr[1] == 0:
        raise RuntimeError(
            "real side produced ZERO samples from %s -- refusing to report "
            "nan. Most likely no snapshot anchor was reached; the chain must "
            "begin at a day containing one." % (REAL_DAYS,))
    return {"bid": acc["bid"], "ask": acc["ask"], "n_events": ctr[0],
            "n_samples": ctr[1], "snapshots": nsnap, "diffs": ndiff,
            "gaps": len(gaps)}


def load_source_sizes():
    a = array.array("d")
    with open(DEFAULT_SIZES_PATH, "rb") as fh:
        a.frombytes(fh.read())
    return list(a)


def self_test():
    from matching_engine import MatchingEngine
    print("=" * 100)
    print("Self-test; constructed positive and negative cases for every "
          "extractor")
    print("=" * 100)
    ok = True

    v = list(range(1, 11))
    got = [pct(v, p) for p in (0.25, 0.50, 0.75, 0.95)]
    want = [v[int(p * 9)] for p in (0.25, 0.50, 0.75, 0.95)]
    print("  1. pct()                  : %r  expect %r  -> %s"
          % (got, want, "ok" if got == want else "FAIL"))
    ok &= (got == want)
    nan = pct([], 0.5)
    print("     negative (empty)       : %r, is nan -> %s" % (nan, nan != nan))
    ok &= (nan != nan)

    d = describe([1.0, 2.0, 3.0, 4.0])
    good = (d["n"] == 4 and d["mean"] == 2.5 and d["median"] == 2.0)
    print("  2. describe()             : n=%d mean=%.2f median=%.2f  "
          "expect n=4 mean=2.50 median=2.00  -> %s"
          % (d["n"], d["mean"], d["median"], "ok" if good else "FAIL"))
    ok &= good
    e = describe([])
    print("     negative (empty)       : n=%d, median is nan -> %s"
          % (e["n"], e["median"] != e["median"]))
    ok &= (e["n"] == 0 and e["median"] != e["median"])

    eng = MatchingEngine()
    eng.add_limit_order("buy", 61999.00, 0.30, "a")
    eng.add_limit_order("buy", 61999.00, 0.10, "b")
    eng.add_limit_order("buy", 61995.00, 0.05, "c")
    eng.add_limit_order("buy", 61990.00, 9.90, "d")
    eng.add_limit_order("sell", 62001.00, 0.20, "e")
    smp = SizeSampler.__new__(SizeSampler)
    smp.eng = eng
    smp.warmup = 0.0
    smp.level_sizes = {"bid": [], "ask": []}
    smp.orders_per_level = {"bid": [], "ask": []}
    smp.order_sizes = {"bid": [], "ask": []}
    smp.n_scans = 0
    smp.scan(eng)
    good = (sorted(smp.level_sizes["bid"]) == [0.05, 0.40]
            and smp.level_sizes["ask"] == [0.20]
            and sorted(smp.orders_per_level["bid"]) == [1, 2]
            and smp.orders_per_level["ask"] == [1]
            and sorted(smp.order_sizes["bid"]) == [0.05, 0.10, 0.30]
            and smp.order_sizes["ask"] == [0.20])
    print("  3. Sim scan (sides split) : bid levels=%r ask levels=%r"
          % (sorted(smp.level_sizes["bid"]), smp.level_sizes["ask"]))
    print("     bid orders/level=%r ask orders/level=%r  bid order sizes=%r"
          % (sorted(smp.orders_per_level["bid"]), smp.orders_per_level["ask"],
             sorted(smp.order_sizes["bid"])))
    print("     expect bid [0.05,0.4] / ask [0.2] / bid [1,2] / ask [1] / bid "
          "[0.05,0.1,0.3]  -> %s" % ("ok" if good else "FAIL"))
    ok &= good
    print("     negative: the 9.90 BTC out-of-band level is absent from every "
          "population -> %s"
          % (9.90 not in smp.order_sizes["bid"]
             and 9.90 not in smp.level_sizes["bid"]))
    ok &= (9.90 not in smp.order_sizes["bid"])
    print("     negative: sides did NOT get pooled; bid n=%d, ask n=%d, "
          "distinct -> %s"
          % (len(smp.level_sizes["bid"]), len(smp.level_sizes["ask"]),
             smp.level_sizes["bid"] != smp.level_sizes["ask"]))
    ok &= (smp.level_sizes["bid"] != smp.level_sizes["ask"])

    st = State()
    st.bids.load_snapshot([("61999.00", "0.40000000"),
                           ("61995.00", "0.05000000"),
                           ("61990.00", "9.90000000")])
    st.asks.load_snapshot([("62001.00", "0.20000000")])
    racc = {"bid": [], "ask": []}
    fired = real_scan(st, racc)
    good = (fired and sorted(racc["bid"]) == [0.05, 0.40]
            and racc["ask"] == [0.20])
    print("  4. Real scan              : bid=%r ask=%r  expect [0.05,0.4] / "
          "[0.2]  -> %s" % (sorted(racc["bid"]), racc["ask"],
                            "ok" if good else "FAIL"))
    ok &= good

    stx = State()
    stx.bids.load_snapshot([("62002.00", "1.0")])
    stx.asks.load_snapshot([("62001.00", "1.0")])
    xacc = {"bid": [], "ask": []}
    rejected = (real_scan(stx, xacc) is False)
    print("     negative (crossed book): real_scan returned False and "
          "collected %d values -> %s"
          % (len(xacc["bid"]) + len(xacc["ask"]),
             "ok" if (rejected and not xacc["bid"] and not xacc["ask"])
             else "FAIL"))
    ok &= (rejected and not xacc["bid"] and not xacc["ask"])

    st1 = State()
    st1.bids.load_snapshot([("61999.00", "1.0")])
    oacc = {"bid": [], "ask": []}
    print("     negative (one-sided)   : real_scan returned %s (expect False)"
          % real_scan(st1, oacc))
    ok &= (real_scan(st1, oacc) is False)

    m_sim = 0.5 * (eng.best_bid() + eng.best_ask())
    m_real = 0.5 * (float(st.bids.best()) + float(st.asks.best()))
    print("  5. window convention      : sim mid %.4f -> [%.4f, %.4f], real "
          "mid %.4f -> [%.4f, %.4f]  -> %s"
          % (m_sim, m_sim * (1 - BAND), m_sim * (1 + BAND),
             m_real, m_real * (1 - BAND), m_real * (1 + BAND),
             "ok" if abs(m_sim - m_real) < 1e-9 else "FAIL"))
    ok &= (abs(m_sim - m_real) < 1e-9)

    d = describe(load_source_sizes())
    good = (d["n"] == 39184)
    print("  6. trade_sizes.bin        : n=%d (expect 39184)  median=%.5f "
          "(Phase 3 records 0.00076)  mean=%.5f (records 0.00347)  -> %s"
          % (d["n"], d["median"], d["mean"], "ok" if good else "FAIL"))
    ok &= good

    print("  7. safe_ratio             : 1/0 -> %r is nan -> %s"
          % (safe_ratio(1.0, 0.0), safe_ratio(1.0, 0.0) != safe_ratio(1.0, 0.0)))
    ok &= (safe_ratio(1.0, 0.0) != safe_ratio(1.0, 0.0))
    ok &= (safe_ratio(1.0, 2.0) == 0.5)

    print("\n  self-test: %s" % ("all PASS" if ok else "failure; stopping"))
    return ok


HDR = ("  %-34s %9s %10s %10s %10s %10s %10s"
       % ("", "n", "p25", "median", "p75", "p95", "mean"))


def row(label, d):
    return ("  %-34s %9d %10.5f %10.5f %10.5f %10.5f %10.5f"
            % (label, d["n"], d["p25"], d["median"], d["p75"], d["p95"],
               d["mean"]))


def ratio_row(label, num, den):
    return ("  %-34s %10.2f %10.2f %10.2f %10.2f %10.2f"
            % (label,
               safe_ratio(num["p25"], den["p25"]),
               safe_ratio(num["median"], den["median"]),
               safe_ratio(num["p75"], den["p75"]),
               safe_ratio(num["p95"], den["p95"]),
               safe_ratio(num["mean"], den["mean"])))


def main():
    print("=" * 100)
    print("Resting size; is the sim's per-level depth uniformly ~2x real, "
          "or is the shape wrong?")
    print("=" * 100)
    print("Sim: emergent book (no seed book), W=%.0fs, MM-ABSENT, 1 day, %d "
          "seeds, sampled every %ds." % (WARMUP, len(SEEDS), SAMPLE_EVERY))
    print("     lam=%.4f p_market=%.4f mean_lifetime=%.0fs disp=%.4f %s "
          "clipping. Nothing retuned."
          % (LAM, P_MARKET, LIFE, DISP, CLIP.upper()))
    print("Real: cloud chain %s, every %d applied events, crossed and "
          "one-sided samples skipped."
          % ("+".join(REAL_DAYS), REAL_SAMPLE_EVERY))
    print("Both: within 1bp of mid, mid*(1 +/- %g); run_measure's own "
          "convention." % BAND)
    print("")
    print("!! Binance publishes [price, AGGREGATE_QUANTITY]; two fields. "
          "Verified at the raw feed.")
    print("!! Size per order and orders per level therefore have no real "
          "counterpart and never will;")
    print("!! the information was never transmitted. They are reported "
          "sim-only and are never scored")
    print("!! or ratioed against anything real. A real level aggregates an "
          "unknown number of makers")
    print("!! while a sim level aggregates a known number of orders, so "
          "per-level is apples-to-apples")
    print("!! and per-order is not comparable even in principle.")
    print("")
    print("!! Sides are never pooled in Section A. The real book is 1.83:1 "
          "asymmetric near the touch")
    print("!! (bid %.4f / ask %.4f BTC), so pooling the sim and ratioing it "
          "against each real side"
          % (REAL_NEAR_BTC_BID, REAL_NEAR_BTC_ASK))
    print("!! separately would manufacture a ~1.8x spread between the two "
          "ratio rows out of the")
    print("!! asymmetry alone. Bid is compared to bid, ask to ask.")
    print("")
    print("!! n counts level observations, not distinct levels. A level "
          "persisting across many scans")
    print("!! is counted once per scan, on both sides equally; a consistent "
          "time-weighted convention.")
    print("")
    print("!! Percentiles: sorted[int(p*(n-1))], no interpolation, applied "
          "identically to both sides.")
    print("!! Real figures are pooled over samples and carry no error bar; "
          "there is one real book,")
    print("!! not fifteen draws of one. Do not read their digits as precision.")
    print("")

    if not self_test():
        return 1
    print("")

    t0 = time.time()
    sim_lvl = {"bid": [], "ask": []}
    sim_opl = {"bid": [], "ask": []}
    sim_ord = {"bid": [], "ask": []}
    per_seed_med = {"bid": [], "ask": []}
    for s in SEEDS:
        smp = run_sim_seed(s)
        for side in ("bid", "ask"):
            sim_lvl[side].extend(smp.level_sizes[side])
            sim_opl[side].extend(smp.orders_per_level[side])
            sim_ord[side].extend(smp.order_sizes[side])
            per_seed_med[side].append(describe(smp.level_sizes[side])["median"])
        print("  ran sim seed %d (%d scans, %d bid / %d ask levels, %.0fs "
              "elapsed)" % (s, smp.n_scans, len(smp.level_sizes["bid"]),
                            len(smp.level_sizes["ask"]), time.time() - t0),
              flush=True)
    print("")
    print("  streaming the real chain ...", flush=True)
    real = run_real()
    print("  real: %d events, %d samples, %d snapshots, %d diffs, %d gaps "
          "(%.0fs total)" % (real["n_events"], real["n_samples"],
                             real["snapshots"], real["diffs"], real["gaps"],
                             time.time() - t0))
    print("")

    d_sim_b = describe(sim_lvl["bid"])
    d_sim_a = describe(sim_lvl["ask"])
    d_real_b = describe(real["bid"])
    d_real_a = describe(real["ask"])

    print("=" * 100)
    print("A. Size per level within 1bp; the headline. Bid to bid, ask to "
          "ask, nothing pooled.")
    print("=" * 100)
    print(HDR)
    print(row("Sim bid", d_sim_b))
    print(row("sim ask", d_sim_a))
    print(row("real bid", d_real_b))
    print(row("real ask", d_real_a))
    print("")
    print("  ratio sim/real at each percentile; this is the discriminator:")
    print("  %-34s %10s %10s %10s %10s %10s"
          % ("", "p25", "median", "p75", "p95", "mean"))
    print(ratio_row("bid  sim/real", d_sim_b, d_real_b))
    print(ratio_row("ask  sim/real", d_sim_a, d_real_a))
    print("")
    print("  A roughly FLAT ratio row = scale error: the sizes drawn are "
          "simply too big.")
    print("  A ratio row that climbs with percentile = shape error: the "
          "typical level is about right")
    print("  and the right tail is too heavy. These point at different fixes "
          "and nothing else here")
    print("  distinguishes them.")
    print("")
    print("  Side asymmetry, reported because pooling would have hidden it:")
    print("    real ask/bid median ratio = %.3f     sim ask/bid median ratio "
          "= %.3f" % (safe_ratio(d_real_a["median"], d_real_b["median"]),
                      safe_ratio(d_sim_a["median"], d_sim_b["median"])))
    print("    A sim near 1.000 against a real ratio far from it is a finding "
          "in its own right:")
    print("    the simulator has no mechanism that would make one side "
          "systematically heavier.")
    print("")
    print("  per-seed spread on the sim median (SE across %d seeds, the only "
          "error bar available):" % len(SEEDS))
    for side in ("bid", "ask"):
        m = per_seed_med[side]
        print("    %s  %.5f +/- %.5f"
              % (side, statistics.mean(m),
                 statistics.stdev(m) / (len(m) ** 0.5) if len(m) > 1 else 0.0))
    print("")

    d_src = describe(load_source_sizes())
    d_ord_pooled = describe(sim_ord["bid"] + sim_ord["ask"])

    print("=" * 100)
    print("B. Where the size comes from; sim per-resting-order against the "
          "bootstrap source")
    print("=" * 100)
    print("  Amended after stacking_source.py: this is the near-touch resting "
          "Population, not what is")
    print("  drawn. The drawn population was measured separately and matches "
          "the source 1.00 at every")
    print("  percentile over 2,406,478 orders. The 1.69x below is survivorship "
          "in what rests.")
    print("")
    print("  Pooled across sides here on purpose: trade_sizes.bin is an "
          "unsigned population (Verified")
    print("  Facts), so the comparable sim quantity is also unsided. Section "
          "A's rule is the opposite")
    print("  for the same reason; match the convention of the thing being "
          "compared against.")
    print("")
    print(HDR)
    print(row("trade_sizes.bin (39,184 taker)", d_src))
    print(row("sim order sizes resting within 1bp", d_ord_pooled))
    print("")
    print("  %-34s %10s %10s %10s %10s %10s"
          % ("", "p25", "median", "p75", "p95", "mean"))
    print(ratio_row("sim/source", d_ord_pooled, d_src))
    print("")
    print("  branch 1; confirmed by stacking_source.py. The sizes drawn are "
          "faithful and the excess is")
    print("    orders stacking on levels, which points at the JOIN clipping "
          "rule. That rule is recorded")
    print("    as measured but not adopted; nothing in the collected data "
          "validates it as what the real")
    print("    venue does. It was chosen because it made the book behave. "
          "Measured split: unclipped")
    print("    orders rest median 1 / mean 1.00216 / p95 1 per level; clipped "
          "orders median 8 / mean 21.3")
    print("    / p95 87. Every multi-order level in this book is a clipped "
          "level.")
    print("  Branch 2; dead. per-order does NOT run high at the draw: "
          "submitted/source is 1.00 at")
    print("    every percentile over 2,406,478 orders. trade_sizes.bin is not "
          "implicated by this number.")
    print("    (Open limitation #3; taker sizes standing in for maker sizes "
          "with no maker-size data")
    print("    anywhere behind them; is untouched by this and still "
          "stands.)")
    print("")

    print("=" * 100)
    print("C. Orders per level; sim only. No real counterpart exists.")
    print("=" * 100)
    print(HDR)
    print(row("Sim orders per level, bid", describe([float(x) for x in sim_opl["bid"]])))
    print(row("sim orders per level, ask", describe([float(x) for x in sim_opl["ask"]])))
    print("")
    print("  Never scored and never ratioed against real: Binance published "
          "no per-order detail, so")
    print("  there is nothing to compare this to. It locates the defect "
          "inside the simulator. If orders")
    print("  per level runs materially above 1, levels are being shared, "
          "which is branch 1's stacking.")
    print("")
    print("  Nothing fixed, swept or retuned. No committed default changed.")
    print("  wall clock %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
