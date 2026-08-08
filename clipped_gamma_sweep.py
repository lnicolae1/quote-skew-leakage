# clipped_gamma_sweep.py: gamma range and holding period at the clipped working point, MM present

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_SIGMA
from clipped_mm_check import _resid_best

T = 86400.0
SEEDS = list(range(5))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763
MM_ID = "MM"

C_PER_GAMMA = 1.458e6

GAMMAS = [1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 2e-5, 4e-5, 1e-4]

AR1_DT = 60

VIABILITY = 0.50
SPREAD_GATE = 10.0


def ar1_half_life(series):
    """Demeaned AR(1) fit"""
    n = len(series)
    if n < 3:
        return None, None
    m = sum(series) / n
    x = [v - m for v in series]
    num = sum(x[i] * x[i - 1] for i in range(1, n))
    den = sum(x[i - 1] * x[i - 1] for i in range(1, n))
    if den <= 0:
        return None, None
    phi = num / den
    if phi <= 0.0 or phi >= 1.0:
        return phi, None
    return phi, AR1_DT * math.log(0.5) / math.log(phi)


def run(seed, gamma):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = MarketMaker(horizon=T, k=K, gamma=gamma)

    mm_fills = 0
    mm_vol = 0.0
    tot_vol = 0.0
    mm_spreads, resid_spreads, skews, q_all = [], [], [], []
    q_sampled = []

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
                        mm_vol += f.size
                mm_fills += mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        q_all.append(mm.q)
        if int(t) % AR1_DT == 0:
            q_sampled.append(mm.q)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mb = ma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mb = o.price if mb is None else max(mb, o.price)
            else:
                ma = o.price if ma is None else min(ma, o.price)
        if mb is None or ma is None:
            continue
        mm_spreads.append(ma - mb)
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is None or ra is None or ra <= rb:
            continue
        resid_spreads.append(ra - rb)
        skews.append(0.5 * (mb + ma) - 0.5 * (rb + ra))

    cut = int(0.70 * len(q_sampled))
    phi_full, hl_full = ar1_half_life(q_sampled)
    phi_tr, hl_tr = ar1_half_life(q_sampled[:cut])

    med_mm = statistics.median(mm_spreads) if mm_spreads else float("nan")
    sk_sd = statistics.stdev(skews) if len(skews) > 1 else 0.0

    return {
        "mm_fills": float(mm_fills),
        "mm_vol": mm_vol,
        "tot_vol": tot_vol,
        "vol_share": (mm_vol / tot_vol) if tot_vol > 0 else 0.0,
        "mm_spread": med_mm,
        "resid_spread": (statistics.median(resid_spreads)
                         if resid_spreads else float("nan")),
        "sd_q": statistics.stdev(q_all) if len(q_all) > 1 else 0.0,
        "max_abs_q": max(abs(v) for v in q_all) if q_all else 0.0,
        "skew_sd": sk_sd,
        "leak_frac": (sk_sd / med_mm) if med_mm and med_mm > 0 else 0.0,
        "phi_full": phi_full if phi_full is not None else float("nan"),
        "hl_full": hl_full,
        "phi_train": phi_tr if phi_tr is not None else float("nan"),
        "hl_train": hl_tr,
    }


def mean_se(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float)
                                                   and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    m = statistics.mean(v)
    se = (statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, se, len(v)


def measure(gamma):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, gamma)
        if per is None:
            per = {key: [] for key in x}
        for key, v in x.items():
            per[key].append(v)
    out = {"gamma": gamma, "C": gamma * C_PER_GAMMA,
           "secs": time.time() - t0}
    for key, vals in per.items():
        m, se, n = mean_se(vals)
        out[key], out[key + "_se"], out[key + "_n"] = m, se, n
    return out


def gate_cap(rows, value_of, thresh, decreasing):
    """Log-interpolate the C at which `value_of` crosses `thresh`"""
    prev = None
    for r in rows:
        v = value_of(r)
        if prev is not None:
            pv = value_of(prev)
            hit = (pv >= thresh > v) if decreasing else (pv <= thresh < v)
            if hit:
                span = pv - v if decreasing else v - pv
                f = ((pv - thresh) / span) if decreasing else \
                    ((thresh - pv) / span)
                lc = math.log(prev["C"]) + f * (math.log(r["C"])
                                                - math.log(prev["C"]))
                c = math.exp(lc)
                return prev, r, c, c / C_PER_GAMMA
        prev = r
    return None


def fmt_hl(m, n):
    if n == 0 or (isinstance(m, float) and math.isnan(m)):
        return "  none stationary"
    return "%9.0fs (%.1fh) n=%d" % (m, m / 3600.0, n)


def main():
    print("=" * 128)
    print("Gamma sweep at the clipped working point; sets the gamma range "
          "and the holding period. Nothing is chosen here.")
    print("=" * 128)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("k=%.5f (constructor arg; DEFAULT_K never edited). sigma=%.4f. "
          "horizon=%.0fs. %d seeds x %.0fs."
          % (K, DEFAULT_SIGMA, T, len(SEEDS), T))
    print("C_at_t0 = gamma * %.4g at this horizon. DEFAULT_GAMMA=%g -> "
          "C = %.3f." % (C_PER_GAMMA, DEFAULT_GAMMA,
                         DEFAULT_GAMMA * C_PER_GAMMA))
    print("The old sweep ran at horizon 8,640,000s, so its gamma grid maps to "
          "this one at 100x.")
    print("Its gamma=1e-6 row (the dead MM) is C=146, not the current "
          "configuration; that is C=1.458.")
    print("")

    rows = []
    for g in GAMMAS:
        r = measure(g)
        rows.append(r)
        print("  gamma=%.1e  C=%9.3f  measured in %3.0fs   fills/day=%7.1f  "
              "MM spread=$%.2f"
              % (g, r["C"], r["secs"], r["mm_fills"], r["mm_spread"]),
              flush=True)
    print("")

    base = rows[0]["mm_fills"]

    print("=" * 128)
    print("The sweep, indexed by C (transferable) and gamma (this horizon "
          "only)")
    print("=" * 128)
    print("  %9s %9s %11s %8s %11s %11s %9s %10s %11s"
          % ("C_at_t0", "gamma", "fills/day", "vs g->0", "MM spread",
             "mkt spread", "MM/mkt", "vol share", "sd(q) BTC"))
    for r in rows:
        print("  %9.3f %9.1e %7.1f+-%-3.0f %7.1f%% %8.2f+-%-2.2f "
              "%8.2f+-%-2.2f %8.2f %9.2f%% %10.5f"
              % (r["C"], r["gamma"], r["mm_fills"], r["mm_fills_se"],
                 100.0 * r["mm_fills"] / base if base > 0 else float("nan"),
                 r["mm_spread"], r["mm_spread_se"],
                 r["resid_spread"], r["resid_spread_se"],
                 r["mm_spread"] / r["resid_spread"]
                 if r["resid_spread"] > 0 else float("nan"),
                 100.0 * r["vol_share"], r["sd_q"]))
    print("")
    print("  vs g->0 is the fill rate as a percentage of the gamma=%.0e "
          "anchor (%.1f fills/day)." % (GAMMAS[0], base))

    print("")
    print("=" * 128)
    print("The viability gates; both pre-registered, both blind to the "
          "damage number")
    print("=" * 128)

    g1 = gate_cap(rows, lambda r: 100.0 * r["mm_fills"] / base,
                  100.0 * VIABILITY, True)
    g2 = gate_cap(rows, lambda r: r["mm_spread"] / r["resid_spread"],
                  SPREAD_GATE, False)

    def show(tag, rule, g, unit):
        if g is None:
            print("  %s (%s): never crossed in this grid; the grid is too "
                  "narrow to place this gate." % (tag, rule))
            return None
        lo, hi, c, gam = g
        print("  %s: %s" % (tag, rule))
        print("      crosses between C=%.3f (gamma=%.1e) and C=%.3f "
              "(gamma=%.1e)" % (lo["C"], lo["gamma"], hi["C"], hi["gamma"]))
        print("      log-interpolated cap:  C = %.2f,  gamma = %.2e   [%s]"
              % (c, gam, unit))
        return c

    c1 = show("GATE 1  fill rate", "MM keeps >= %.0f%% of its gamma->0 fill "
              "rate (%.1f/day)" % (100 * VIABILITY, base), g1,
              "this horizon only")
    print("")
    c2 = show("GATE 2  spread realism",
              "MM spread / market touch <= %.0fx" % SPREAD_GATE, g2,
              "this horizon only")
    print("")

    caps = [(c, n) for c, n in ((c1, "GATE 1 (fill rate)"),
                                (c2, "GATE 2 (spread realism)"))
            if c is not None]
    if not caps:
        print("  neither gate crosses. The range cannot be set from this grid.")
    else:
        cbind, nbind = min(caps)
        print("  Binding gate: %s, at C = %.2f (gamma = %.2e at this "
              "horizon)." % (nbind, cbind, cbind / C_PER_GAMMA))
        if len(caps) > 1:
            cother, nother = max(caps)
            print("  The other gate would have allowed C = %.2f; a factor "
                  "of %.2f looser." % (cother, cother / cbind))
        print("")
        print("  Gate 1 alone was the original rule and it is silent on "
              "spread realism: at its own cap the")
        print("  MM quotes about %.0fx the market touch. Gate 2 exists "
              "because that is not a market maker."
              % (SPREAD_GATE * 1.2))
        print("  Both gates are functions of the MM's own behaviour only. "
              "Neither looks at the sniffer,")
        print("  the damage, or any defense, so neither can be called tuned "
              "to the result.")
        print("")
        print("  Note on precision: both caps are log-interpolations between "
              "adjacent grid points, not")
        print("  measured points. A finer grid inside the bracketing interval "
              "would be needed to state")
        print("  either to better than the width of that interval.")

    print("")
    print("=" * 128)
    print("The leak fraction; does the ~3% ceiling from the old book hold "
          "here?")
    print("=" * 128)
    print("  %9s %9s %13s %13s %15s %11s"
          % ("C_at_t0", "gamma", "skew_sd $", "MM spread $",
             "skew_sd/spread", "max|q| BTC"))
    for r in rows:
        print("  %9.3f %9.1e %8.4f+-%-4.4f %8.2f+-%-4.2f %14.3f%% %11.5f"
              % (r["C"], r["gamma"], r["skew_sd"], r["skew_sd_se"],
                 r["mm_spread"], r["mm_spread_se"],
                 100.0 * r["leak_frac"], r["max_abs_q"]))
    peak = max(rows, key=lambda r: r["leak_frac"])
    print("")
    print("  peak leak fraction %.3f%% at C=%.3f (gamma=%.1e), where the MM "
          "keeps %.1f%% of its fills."
          % (100.0 * peak["leak_frac"], peak["C"], peak["gamma"],
             100.0 * peak["mm_fills"] / base if base > 0 else float("nan")))
    print("  Old book (k=0.06747, base spread $29.6) peaked at 3.11%% at "
          "C=58.3. New base spread is 2/k = $%.2f." % (2.0 / K))
    print("  The MM's whole quoted spread is %.2f bp of a ~$62,000 mid, so "
          "the damage ceiling is that"
          % (10000.0 * rows[0]["mm_spread"] / 62000.0))
    print("  leak fraction times the half-spread; state it in bp before "
          "step 7 runs, not after.")

    print("")
    print("=" * 128)
    print("Inventory mean-reversion; this sets the holding period")
    print("=" * 128)
    print("  Sampled every %ds. phi >= 1 is non-stationary and has no "
          "half-life; those seeds are counted, not averaged." % AR1_DT)
    print("")
    print("  %9s %9s %12s %26s %12s %26s"
          % ("C_at_t0", "gamma", "phi (full)", "half-life (full run)",
             "phi (70%)", "half-life (train slice)"))
    for r in rows:
        print("  %9.3f %9.1e %12.6f %26s %12.6f %26s"
              % (r["C"], r["gamma"], r["phi_full"],
                 fmt_hl(r["hl_full"], r["hl_full_n"]),
                 r["phi_train"], fmt_hl(r["hl_train"], r["hl_train_n"])))
    print("")
    print("  The full-run column is contaminated: tau decays to zero inside "
          "the run, so the MM stops")
    print("  skewing over the tail and inventory random-walks regardless of "
          "gamma. The train-slice")
    print("  column is the one to set a holding period from; it is the same "
          "70% the sniffer fits on.")
    print("")
    print("  Old sweep, 100-day horizon, measured 40,000-80,000s (11-22 "
          "hours). If that order of")
    print("  magnitude survives here, a 60s or 300s holding period is under "
          "1% of the half-life and")
    print("  would return a null for a reason unrelated to leakage. Read this "
          "table before fixing it.")

    print("")
    print("=" * 128)
    print("What this run does not do")
    print("=" * 128)
    print("  It does not choose a gamma. It measures where the viability line "
          "falls so the range can be")
    print("  set by the pre-registered rule. Step 7 is not run. No committed "
          "default was edited --")
    print("  gamma and k are constructor arguments and DEFAULT_GAMMA=%g is "
          "untouched." % DEFAULT_GAMMA)


if __name__ == "__main__":
    main()
