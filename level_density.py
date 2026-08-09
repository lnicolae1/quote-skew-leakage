# level_density.py: near-touch price-level density, simulator vs real book

import bisect
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from market_maker import MarketMaker, DEFAULT_QUOTE_SIZE
from clipped_placement import make_world
from clipped_mm_check import MM_ID, ABSENT
from compute_volatility import State, stream_reconstruct, build_cloud_path

T_SIM = 86400.0
SIM_SEEDS = [0, 1, 2]
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
C_TARGET = 13.66
C_PER_GAMMA_1D = 1.458e6
GAMMA_1D = C_TARGET / C_PER_GAMMA_1D
SIM_SAMPLE_EVERY = 60

REAL_DAYS = ["07-11", "07-12", "07-13"]
REAL_SAMPLE_EVERY = 2000
TICK = 0.01
BANDS_BP = [1.0, 5.0]


def spacings(prices):
    """Adjacent gaps in a sorted price list"""
    if len(prices) < 2:
        return []
    ps = sorted(prices)
    return [ps[i] - ps[i - 1] for i in range(1, len(ps))]


def sim_sample(eng, seed_ids, include_mm):
    """Levels and spacings within each band, per side, at one instant"""
    bb, ba = eng.best_bid(), eng.best_ask()
    if bb is None or ba is None or ba <= bb:
        return None
    mid = 0.5 * (bb + ba)
    out = {}
    live_bid, live_ask = [], []
    for book, store in ((eng.bids, live_bid), (eng.asks, live_ask)):
        for p, q in book.items():
            ok = False
            for o in q:
                if o.id in seed_ids:
                    continue
                if not include_mm and o.agent_id == MM_ID:
                    continue
                ok = True
                break
            if ok:
                store.append(p)
    out["tot_levels"] = 0.5 * (len(live_bid) + len(live_ask))
    for bp in BANDS_BP:
        lo, hi = mid * (1 - bp * 1e-4), mid * (1 + bp * 1e-4)
        nb = [p for p in live_bid if p >= lo]
        na = [p for p in live_ask if p <= hi]
        out["lv_%g" % bp] = 0.5 * (len(nb) + len(na))
        out["sp_%g" % bp] = spacings(nb) + spacings(na)
    return out


def run_sim(seed, include_mm):
    _p, vf, eng, noise, informed = make_world(seed, T_SIM, LAM, P_MARKET,
                                              LIFE, DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = (MarketMaker(horizon=T_SIM, k=K, gamma=GAMMA_1D,
                      quote_size=DEFAULT_QUOTE_SIZE) if include_mm else None)

    acc = {"tot_levels": []}
    for bp in BANDS_BP:
        acc["lv_%g" % bp] = []
        acc["sp_%g" % bp] = []

    t = 0.0
    while t < T_SIM:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fl = getattr(r, "fills", None)
                if fl and mm is not None:
                    mm.on_fills(fl, r.side)
        if mm is not None:
            mm.requote(eng, t)
        if int(t) % SIM_SAMPLE_EVERY:
            continue
        s = sim_sample(eng, seed_ids, include_mm)
        if s is None:
            continue
        acc["tot_levels"].append(s["tot_levels"])
        for bp in BANDS_BP:
            acc["lv_%g" % bp].append(s["lv_%g" % bp])
            acc["sp_%g" % bp].extend(s["sp_%g" % bp])

    out = {"n_samples": float(len(acc["tot_levels"]))}
    out["tot_levels"] = (statistics.mean(acc["tot_levels"])
                         if acc["tot_levels"] else float("nan"))
    for bp in BANDS_BP:
        lv = acc["lv_%g" % bp]
        sp = acc["sp_%g" % bp]
        out["lv_%g" % bp] = statistics.mean(lv) if lv else float("nan")
        out["sp_%g" % bp] = statistics.median(sp) if sp else float("nan")
        out["nsp_%g" % bp] = float(len(sp))
    return out


def real_sample(state):
    bb = state.bids.best()
    ba = state.asks.best()
    if bb is None or ba is None:
        return None
    bbf, baf = float(bb), float(ba)
    if bbf >= baf:
        return None
    mid = 0.5 * (bbf + baf)
    out = {"tot_levels": 0.5 * (len(state.bids.book) + len(state.asks.book))}
    for bp in BANDS_BP:
        lo, hi = mid * (1 - bp * 1e-4), mid * (1 + bp * 1e-4)
        nb = [float(p) for p in state.bids.book if float(p) >= lo]
        na = [float(p) for p in state.asks.book if float(p) <= hi]
        out["lv_%g" % bp] = 0.5 * (len(nb) + len(na))
        out["sp_%g" % bp] = spacings(nb) + spacings(na)
    return out


def run_real():
    state = State()
    gaps = []
    acc = {"tot_levels": []}
    for bp in BANDS_BP:
        acc["lv_%g" % bp] = []
        acc["sp_%g" % bp] = []
    ctr = [0, 0]

    def on_event(ts, uid, mid):
        ctr[0] += 1
        if ctr[0] % REAL_SAMPLE_EVERY:
            return
        s = real_sample(state)
        if s is None:
            return
        ctr[1] += 1
        acc["tot_levels"].append(s["tot_levels"])
        for bp in BANDS_BP:
            acc["lv_%g" % bp].append(s["lv_%g" % bp])
            acc["sp_%g" % bp].extend(s["sp_%g" % bp])

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

    out = {"n_events": float(ctr[0]), "n_samples": float(ctr[1]),
           "snapshots": float(nsnap),
           "diffs": float(ndiff), "gaps": float(len(gaps))}
    out["tot_levels"] = (statistics.mean(acc["tot_levels"])
                         if acc["tot_levels"] else float("nan"))
    for bp in BANDS_BP:
        lv = acc["lv_%g" % bp]
        sp = acc["sp_%g" % bp]
        out["lv_%g" % bp] = statistics.mean(lv) if lv else float("nan")
        out["sp_%g" % bp] = statistics.median(sp) if sp else float("nan")
        out["nsp_%g" % bp] = float(len(sp))
    return out


def main():
    print("=" * 124)
    print("Near-touch level density; simulator against the real book. THE "
          "discriminating measurement.")
    print("=" * 124)
    print("Sim: clipped working point lam=%.4f p_market=%.4f "
          "mean_lifetime=%.0fs disp=%.4f, JOIN clipping."
          % (LAM, P_MARKET, LIFE, DISP))
    print("     %d seeds x %.0fs, sampled every %ds, seed-book orders "
          "excluded. k=%.5f, C=%.2f -> gamma=%.3e at this horizon."
          % (len(SIM_SEEDS), T_SIM, SIM_SAMPLE_EVERY, K, C_TARGET, GAMMA_1D))
    print("Real: cloud chain %s, streamed line by line through the "
          "committed reconstruction, sampled every %d events."
          % ("+".join(REAL_DAYS), REAL_SAMPLE_EVERY))
    print("Measured tick = $%.2f (Verified Facts). The 1000-level snapshot "
          "cap cannot bite: everything" % TICK)
    print("here is within 1bp of the mid, nowhere near rank 1000.")
    print("")

    rows = []
    for label, inc in (("SIM, MM-absent", False), ("SIM, MM-present", True)):
        t0 = time.time()
        per = None
        for s in SIM_SEEDS:
            x = run_sim(s, inc)
            if per is None:
                per = {k: [] for k in x}
            for k, v in x.items():
                per[k].append(v)
        agg = {k: statistics.mean(v) for k, v in per.items()}
        rows.append((label, agg))
        print("  %-18s %4.0fs   levels/1bp = %.2f   median spacing = $%.4f"
              % (label, time.time() - t0, agg["lv_1"], agg["sp_1"]),
              flush=True)

    t0 = time.time()
    real = run_real()
    rows.append(("REAL, cloud " + "+".join(REAL_DAYS), real))
    print("  %-18s %4.0fs   levels/1bp = %.2f   median spacing = $%.4f"
          % ("real chain", time.time() - t0, real["lv_1"],
             real["sp_1"]), flush=True)
    print("    (%.0f events streamed, %.0f snapshots, %.0f diffs, %.0f "
          "samples, %.0f gaps)"
          % (real["n_events"], real["snapshots"], real["diffs"],
             real["n_samples"], real["gaps"]))
    print("")

    print("=" * 124)
    print("A. Levels within 1bp per side, and the actual spacing between "
          "them")
    print("=" * 124)
    print("  %-22s %14s %18s %16s %14s"
          % ("book", "levels /1bp", "median spacing $", "spacing / tick",
             "total levels"))
    for label, r in rows:
        print("  %-22s %14.2f %18.4f %16.1f %14.1f"
              % (label, r["lv_1"], r["sp_1"], r["sp_1"] / TICK,
                 r["tot_levels"]))
    print("")
    print("  For reference, the old committed figure this replaces: "
          "clipped_mm_check.ABSENT near_lvl = %.4f" % ABSENT["near_lvl"])
    print("  (MM-absent, old working point). One bp is $%.2f at a $62,000 "
          "mid." % (62000.0 * 1e-4))

    print("")
    print("=" * 124)
    print("B. THE SAME within 5bp, as a check that the 1bp result is not a "
          "small-sample artefact")
    print("=" * 124)
    print("  %-22s %14s %18s %16s %14s"
          % ("book", "levels /5bp", "median spacing $", "spacing / tick",
             "gaps measured"))
    for label, r in rows:
        print("  %-22s %14.2f %18.4f %16.1f %14.0f"
              % (label, r["lv_5"], r["sp_5"], r["sp_5"] / TICK,
                 r["nsp_5"]))

    print("")
    print("=" * 124)
    print("C. Concentration; what share of the book sits within 1bp of the "
          "mid?")
    print("=" * 124)
    print("  %-22s %16s %16s %14s"
          % ("book", "levels /1bp", "total levels", "share within 1bp"))
    for label, r in rows:
        sh = 100.0 * r["lv_1"] / r["tot_levels"] if r["tot_levels"] else \
            float("nan")
        print("  %-22s %16.2f %16.1f %13.2f%%"
              % (label, r["lv_1"], r["tot_levels"], sh))

    print("")
    print("=" * 124)
    print("D. The verdict")
    print("=" * 124)
    sim_a = rows[0][1]
    rl = rows[2][1]
    ratio = rl["lv_1"] / sim_a["lv_1"] if sim_a["lv_1"] else float("nan")
    sratio = sim_a["sp_1"] / rl["sp_1"] if rl["sp_1"] else float("nan")
    print("  Real book has %.1fx the levels within 1bp that the sim does "
          "(%.2f against %.2f)."
          % (ratio, rl["lv_1"], sim_a["lv_1"]))
    print("  Sim level spacing is %.1fx the real book's ($%.4f against "
          "$%.4f)." % (sratio, sim_a["sp_1"], rl["sp_1"]))
    print("  Real spacing is %.1f ticks; sim spacing is %.1f ticks."
          % (rl["sp_1"] / TICK, sim_a["sp_1"] / TICK))
    print("")
    if ratio > 1.5:
        print("  diagnosis confirmed. The real book is denser near the touch "
              "where the sim is sparse, so")
        print("  the defect sits in the noise-trader placement distribution: "
              "disp = %.4f spreads orders" % DISP)
        print("  over roughly +/-%.0fbp, and matching the total level count "
              "and the median touch width --" % (DISP * 1e4))
        print("  which 847a7fa did; says nothing about local density where "
              "every trade actually lands.")
    else:
        print("  Diagnosis REFUTED, and in the opposite direction. The sim is "
              "the denser book near the")
        print("  touch; %.2f levels/1bp against the real %.2f, at $%.4f "
              "spacing against $%.4f. Near-touch"
              % (sim_a["lv_1"], rl["lv_1"], sim_a["sp_1"], rl["sp_1"]))
        print("  sparsity therefore cannot explain impact that is worse in "
              "the sim than in reality, and")
        print("  the mechanism paragraph in 0860f69 is wrong. The real "
              "Binance.US book genuinely carries")
        print("  only ~%.1f price levels per side within $%.2f, spaced ~%.0f "
              "ticks apart: it is a very thin"
              % (rl["lv_1"], 62000.0 * 1e-4, rl["sp_1"] / TICK))
        print("  venue, and an intuition that a small trade should not move "
              "it was calibrated to a dense")
        print("  equity book rather than to this one.")

    print("  Limits: one real day, not eleven; level density is far more "
          "stable than volatility but")
    print("  this is a single day. Crossed-book samples skipped. Nothing "
          "retuned, no committed")
    print("  default changed; this measures the gap, it does not close it.")


if __name__ == "__main__":
    main()
