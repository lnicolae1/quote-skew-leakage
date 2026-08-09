# damage_ab.py: paired-seed damage A/B (Phase 6 step 7)
# - Run A (control): sniffer observes only; Run B (treatment): sniffer trades; same seed, same world
# - sniffer: estimates q from the maker's skew (gamma, sigma, tau public); MM long -> sniffer sells, MM short -> buys
# - one open position at a time, at most one action per requote; theta 0 (primary), 0.02, 0.04 BTC;
#   hold 10800 s (headline), 300 s, 60 s; sniffer size 0.02 BTC
# - cluster B: 7 days, 8 seeds, k 0.17763, gamma derived from C = 13.66 at this horizon
# - metrics: d_PnL, MM edge/unit (execution edge), sniffer PnL; SE across seeds only
# - also checks Phase 5 item 3: does the MM make money with no adversary?

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import (MarketMaker, DEFAULT_GAMMA, DEFAULT_SIGMA,
                          DEFAULT_QUOTE_SIZE)
from clipped_mm_check import _resid_best, MM_ID

T = 604800.0
SEEDS = list(range(8))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
REF_MID = 62000.0

C_TARGET = 13.66
TAU_FLOOR_FRAC = 0.02
SNIFF_SIZE = DEFAULT_QUOTE_SIZE
MTM_EVERY = 3600
SEC_PER_YEAR = 365.25 * 86400.0

HOLD_MID = 10800
HOLDS = [HOLD_MID, 300, 60]
THETAS = [0.0, 0.02, 0.04]


def gamma_for_C(c_target, horizon):
    """gamma from the C invariant: C = gamma * (sigma * mid)^2 * tau."""
    tau0 = horizon / SEC_PER_YEAR
    per_gamma = DEFAULT_SIGMA * DEFAULT_SIGMA * REF_MID * REF_MID * tau0
    return c_target / per_gamma, per_gamma


GAMMA, C_PER_GAMMA = gamma_for_C(C_TARGET, T)


def sharpe(series, dt):
    """Annualised Sharpe from a mark-to-market series sampled every dt."""
    if len(series) < 3:
        return float("nan")
    d = [series[i] - series[i - 1] for i in range(1, len(series))]
    sd = statistics.stdev(d)
    if sd <= 0:
        return float("nan")
    return (statistics.mean(d) / sd) * math.sqrt(SEC_PER_YEAR / dt)


def run(seed, trades, theta, hold):
    """One simulation; trades=False is Run A, True is Run B."""
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = MarketMaker(horizon=T, k=K, gamma=GAMMA, quote_size=SNIFF_SIZE)

    pos = 0.0
    sn_cash = 0.0
    entry_t = None
    n_open = 0
    sn_vol = 0.0

    tot_vol = 0.0
    mm_edge_num = 0.0
    mm_edge_den = 0.0
    q_all = []
    mtm = []
    last_mid = REF_MID
    tau0 = mm.time_remaining_years(0.0)

    t = 0.0
    while t < T:
        t += 1.0

        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                for f in fills:
                    tot_vol += f.size
                    if f.counterparty_id == MM_ID:
                        # MM was passive: its side is opposite the aggressor
                        sgn = 1.0 if r.side == "buy" else -1.0
                        mm_edge_num += sgn * (f.price - last_mid) * f.size
                        mm_edge_den += f.size
                mm.on_fills(fills, r.side)

        mm.requote(eng, t)
        q_all.append(mm.q)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        last_mid = 0.5 * (bb + ba)
        if int(t) % MTM_EVERY == 0:
            mtm.append(mm.mark_to_market(last_mid))

        if not trades:
            continue

        # sniffer's restricted view
        mmb = mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mmb = o.price if mmb is None else max(mmb, o.price)
            else:
                mma = o.price if mma is None else min(mma, o.price)
        if mmb is None or mma is None:
            continue
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is None or ra is None or ra <= rb:
            continue
        resid_mid = 0.5 * (rb + ra)
        tau = mm.time_remaining_years(t)
        if tau < TAU_FLOOR_FRAC * tau0:
            continue

        # maximally informed inversion: gamma, sigma, tau public
        skew = 0.5 * (mmb + mma) - resid_mid
        sig = mm.sigma_absolute(resid_mid)
        c_t = GAMMA * sig * sig * tau
        if c_t <= 0:
            continue
        q_hat = -skew / c_t

        # at most one action per requote
        if pos != 0.0:
            if t - entry_t >= hold:
                side = "sell" if pos > 0 else "buy"
                fl = eng.submit_market_order(side, abs(pos))
                got = sum(f.size for f in fl)
                cashflow = sum(f.price * f.size for f in fl)
                sn_cash += cashflow if pos > 0 else -cashflow
                sn_vol += got
                mm.on_fills(fl, side)
                pos = pos - got if pos > 0 else pos + got
                if abs(pos) < 1e-12:
                    pos = 0.0
                    entry_t = None
            continue

        # MM long (q_hat > 0) -> sniffer sells
        if abs(q_hat) <= theta:
            continue
        side = "sell" if q_hat > 0 else "buy"
        fl = eng.submit_market_order(side, SNIFF_SIZE)
        got = sum(f.size for f in fl)
        if got <= 0:
            continue
        cashflow = sum(f.price * f.size for f in fl)
        sn_cash += cashflow if side == "sell" else -cashflow
        sn_vol += got
        mm.on_fills(fl, side)
        pos = -got if side == "sell" else got
        entry_t = t
        n_open += 1

    # flatten the sniffer at the end so its PnL is realised
    if trades and pos != 0.0:
        side = "sell" if pos > 0 else "buy"
        fl = eng.submit_market_order(side, abs(pos))
        cashflow = sum(f.price * f.size for f in fl)
        sn_cash += cashflow if pos > 0 else -cashflow
        sn_vol += sum(f.size for f in fl)
        mm.on_fills(fl, side)
        pos = 0.0

    pnl = mm.mark_to_market(last_mid)
    # PnL = spread + INV: spread = edge against the mid at each fill; INV = residual
    spread_pnl = mm_edge_num
    return {
        "mm_pnl": pnl,
        "mm_spread_pnl": spread_pnl,
        "mm_inv_pnl": pnl - spread_pnl,
        "mm_sharpe": sharpe(mtm, MTM_EVERY),
        "mm_inv_var": statistics.variance(q_all) if len(q_all) > 1 else 0.0,
        "mm_edge": (mm_edge_num / mm_edge_den) if mm_edge_den > 0 else 0.0,
        "mm_vol": mm_edge_den,
        "sn_pnl": sn_cash,
        "sn_vol": sn_vol,
        "sn_share": (100.0 * sn_vol / tot_vol) if tot_vol > 0 else 0.0,
        "n_open": float(n_open),
    }


def mean_se(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    m = statistics.mean(v)
    se = (statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, se, len(v)


def main():
    print("=" * 128)
    print("Paired-seed damage A/B (Phase 6 step 7)")
    print("=" * 128)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping." % (LAM, P_MARKET, LIFE, DISP))
    print("Run length %.0fs (%.0f days). k=%.5f. C = %.2f (gate-2 viability "
          "cap), derived gamma = %.4e"
          % (T, T / 86400.0, K, C_TARGET, GAMMA))
    print("  (C_per_gamma at this horizon = %.4g vs 1.458e6 at 1 day.)" % C_PER_GAMMA)
    print("%d seeds. Sniffer size %.3f BTC. One open position at a time, at "
          "most one action per requote."
          % (len(SEEDS), SNIFF_SIZE))
    print("DEFAULT_GAMMA=%g and DEFAULT_QUOTE_SIZE=%g untouched; both are "
          "constructor arguments."
          % (DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE))
    print("")
    print("Expected ceiling (before the run): ~0.066 bp/unit and "
          "~$1.79/day at this cap,")
    print("~$12.5 over 7 days. MM quotes ~1.46bp from mid vs a")
    print("~0.156bp market touch, so its inventory barely moves the mid, and "
          "a 0.02 BTC clip cannot")
    print("reach its quotes through 0.1244 BTC/side of near-touch "
          "depth. A null here means")
    print("the leak is economically negligible here, not that the "
          "sniffer failed (inference is exact).")
    print("")

    # Run A does not depend on theta or hold: one control per seed
    print("Running control arm (Run A, sniffer observes only)...", flush=True)
    t0 = time.time()
    ctrl = {}
    for s in SEEDS:
        ctrl[s] = run(s, False, 0.0, HOLD_MID)
    print("  %d control runs in %.0fs" % (len(SEEDS), time.time() - t0),
          flush=True)
    print("")

    # Phase 5 item 3: MM PnL with no adversary
    cpnl = [ctrl[s]["mm_pnl"] for s in SEEDS]
    cshp = [ctrl[s]["mm_sharpe"] for s in SEEDS]
    cspr = [ctrl[s]["mm_spread_pnl"] for s in SEEDS]
    cinv = [ctrl[s]["mm_inv_pnl"] for s in SEEDS]
    m_pnl, se_pnl, _n = mean_se(cpnl)
    m_shp, se_shp, _n = mean_se(cshp)
    m_spr, se_spr, _n = mean_se(cspr)
    m_inv, se_inv, _n = mean_se(cinv)
    mult = abs(m_pnl / se_pnl) if se_pnl > 0 else float("nan")

    print("=" * 128)
    print("0. Phase 5 item 3: does the MM make money with no adversary present?")
    print("=" * 128)
    print("  Criterion: with no sniffer, a well-parameterized MM")
    print("  should make money.")
    print("")
    print("  Control-arm MM PnL   : %+10.2f +/- %-9.2f  (|mean|/SE = %.2f, "
          "%d seeds, 7 days)" % (m_pnl, se_pnl, mult, len(SEEDS)))
    print("  Control-arm Sharpe   : %+10.4f +/- %-9.4f  (annualised, from "
          "hourly mark-to-market)" % (m_shp, se_shp))
    print("")
    print("  per-seed control PnL : %s"
          % "  ".join("%+.2f" % v for v in cpnl))
    print("  negative on %d of %d seeds."
          % (sum(1 for v in cpnl if v < 0), len(cpnl)))
    print("")
    print("  Decomposition, PnL = spread + INV (exact):")
    print("    Spread (edge captured quoting) : %+10.2f +/- %-9.2f"
          % (m_spr, se_spr))
    print("    INV    (price vs inventory)    : %+10.2f +/- %-9.2f"
          % (m_inv, se_inv))
    tot = abs(m_spr) + abs(m_inv)
    if tot > 0:
        print("    INV is %.1f%% of the gross magnitude (old book: 88-95%%)." % (100.0 * abs(m_inv) / tot))
    print("")
    if m_pnl > 0 and mult >= 2.0:
        print("  VERDICT: PASS. MM makes money with no adversary, "
              "distinguishably from zero.")
    elif m_pnl < 0 and mult >= 2.0:
        print("  VERDICT: FAILED. MM loses money with no adversary present, at")
        print("  %.2f SE. Phase 5 item 3 not satisfied on this book;" % mult)
        print("  every damage number below is a change in a baseline that is itself")
        print("  negative,")
        print("  its own.")
    else:
        print("  VERDICT: indistinguishable from zero at %.2f SE (not a "
              "demonstrated profit);" % mult)
        print("  treated as unresolved:")
        print("  the damage numbers below rest on a baseline of unknown sign.")
    print("")

    configs = [("theta=0  hold=%ds (HEADLINE)" % HOLD_MID, 0.0, HOLD_MID)]
    for h in HOLDS[1:]:
        configs.append(("theta=0  hold=%ds" % h, 0.0, h))
    for th in THETAS[1:]:
        configs.append(("theta=%.3f BTC  hold=%ds" % (th, HOLD_MID), th,
                        HOLD_MID))

    results = []
    for label, th, hold in configs:
        t0 = time.time()
        per = {k: [] for k in ("d_pnl", "d_sharpe", "d_invvar", "d_edge",
                               "sn_pnl", "sn_share", "n_open", "mm_vol")}
        for s in SEEDS:
            b = run(s, True, th, hold)
            a = ctrl[s]
            per["d_pnl"].append(b["mm_pnl"] - a["mm_pnl"])
            per["d_sharpe"].append(b["mm_sharpe"] - a["mm_sharpe"])
            per["d_invvar"].append(b["mm_inv_var"] - a["mm_inv_var"])
            per["d_edge"].append(b["mm_edge"] - a["mm_edge"])
            per["sn_pnl"].append(b["sn_pnl"])
            per["sn_share"].append(b["sn_share"])
            per["n_open"].append(b["n_open"])
            per["mm_vol"].append(b["mm_vol"])
        row = {"label": label, "theta": th, "hold": hold,
               "secs": time.time() - t0, "raw": per}
        for k, v in per.items():
            m, se, n = mean_se(v)
            row[k], row[k + "_se"] = m, se
        results.append(row)
        print("  %-34s %4.0fs   d_PnL = %+9.2f +/- %-8.2f  sniffer share "
              "%.3f%%" % (label, row["secs"], row["d_pnl"], row["d_pnl_se"],
                          row["sn_share"]), flush=True)
    print("")

    print("=" * 128)
    print("1. Damage: paired difference Run B minus Run A, SE across seeds")
    print("=" * 128)
    print("  %-34s %20s %8s %20s %20s"
          % ("configuration", "d MM PnL ($)", "SE mult", "d MM Sharpe",
             "d inventory var"))
    for r in results:
        mult = abs(r["d_pnl"] / r["d_pnl_se"]) if r["d_pnl_se"] > 0 else \
            float("nan")
        print("  %-34s %9.2f +/- %-7.2f %8.2f %9.4f +/- %-7.4f "
              "%9.2e +/- %-8.2e"
              % (r["label"], r["d_pnl"], r["d_pnl_se"], mult,
                 r["d_sharpe"], r["d_sharpe_se"],
                 r["d_invvar"], r["d_invvar_se"]))
    print("")
    print("  SE mult = |mean| / SE; below ~2 not distinguishable from "
          "zero at %d seeds." % len(SEEDS))
    print("  Holding windows overlap, so SEs are")
    print("  across seeds, never across windows.")

    print("")
    print("=" * 128)
    print("2. Execution quality (less sensitive to an extra "
          "participant in the book)")
    print("=" * 128)
    print("  %-34s %22s %14s %16s"
          % ("configuration", "d MM edge/unit ($)", "as bp of mid",
             "MM volume (BTC)"))
    for r in results:
        bp = 1e4 * r["d_edge"] / REF_MID
        bp_se = 1e4 * r["d_edge_se"] / REF_MID
        print("  %-34s %10.5f +/- %-8.5f %6.4f+-%-6.4f %16.2f"
              % (r["label"], r["d_edge"], r["d_edge_se"], bp, bp_se,
                 r["mm_vol"]))
    print("")
    print("  MM edge/unit = size-weighted mean of (fill price - mid) "
          "signed by the MM's side:")
    print("  edge per BTC; the difference is the cost of the sniffer's presence.")

    print("")
    print("=" * 128)
    print("3. Extra-participant confound; does the attack pay?")
    print("=" * 128)
    print("  %-34s %16s %16s %16s"
          % ("configuration", "sniffer share %", "round trips",
             "sniffer PnL ($)"))
    for r in results:
        print("  %-34s %8.4f+-%-6.4f %16.0f %8.2f+-%-6.2f"
              % (r["label"], r["sn_share"], r["sn_share_se"], r["n_open"],
                 r["sn_pnl"], r["sn_pnl_se"]))
    print("")
    print("  measured at ~4.4% (smoke seed). Run B has a participant Run A")
    print("  lacks, at roughly a third of the MM's footprint, so the "
          "raw PnL difference in")
    print("  section 1 is not pure leakage damage; section 2's "
          "execution edge is primary.")
    print("  Sniffer PnL ~ 0: the attack does not pay")
    print("  regardless of the maker; sniffer PnL much larger than the MM's loss:")
    print("  profit comes from the rest of the book, not the maker")
    print("  at all.")

    print("")
    print("=" * 128)
    print("4. Distribution across seeds")
    print("=" * 128)
    for r in results:
        print("  %s" % r["label"])
        print("    per-seed d MM PnL: %s"
              % "  ".join("%+.2f" % v for v in r["raw"]["d_pnl"]))
    print("")
    print("  Sign agreement across seeds: a difference that flips sign")
    print("  seed to seed is noise.")


if __name__ == "__main__":
    main()
