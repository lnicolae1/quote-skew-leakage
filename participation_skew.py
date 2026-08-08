# participation_skew.py: does quote skew control inventory, and does it depend on the maker's share of flow?

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from clipped_mm_check import _resid_best, MM_ID

T = 86400.0
SEEDS = list(range(5))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
K = 0.17763

GAMMA_CAP = 9.37e-6
FLAT_GAMMA = 1e-9
C_PER_GAMMA = 1.458e6

QUOTE_SIZES = [0.005, 0.010, 0.020]

AR1_DT = 60
H = 3600.0
TRAIN_FRAC = 0.70

ARMS = ("skew", "flat", "gamma0")


class FlatQuoteMM(MarketMaker):
    """gamma_cap in every respect except that the reservation price is the mid"""

    def reservation_price(self, mid, t):
        return mid

    def inventory_skew(self, mid, t):
        return 0.0


def make_mm(arm, qsize):
    if arm == "skew":
        return MarketMaker(horizon=T, k=K, gamma=GAMMA_CAP, quote_size=qsize)
    if arm == "flat":
        return FlatQuoteMM(horizon=T, k=K, gamma=GAMMA_CAP, quote_size=qsize)
    if arm == "gamma0":
        return MarketMaker(horizon=T, k=K, gamma=FLAT_GAMMA, quote_size=qsize)
    raise ValueError("unknown arm %r" % (arm,))


def series_stats(q_sampled):
    """phi, half-life, R_direct and R_ar1 on the train slice only"""
    n = len(q_sampled)
    cut = int(TRAIN_FRAC * n)
    x = q_sampled[:cut]
    if len(x) < 10:
        return None
    m = statistics.mean(x)
    xd = [v - m for v in x]

    den = sum(xd[i - 1] * xd[i - 1] for i in range(1, len(xd)))
    if den <= 0:
        return None
    phi = sum(xd[i] * xd[i - 1] for i in range(1, len(xd))) / den
    hl = (AR1_DT * math.log(0.5) / math.log(phi)) if 0.0 < phi < 1.0 else None

    lag = int(H / AR1_DT)
    if len(xd) <= lag + 2:
        return None
    d2 = sum(xd[i] * xd[i] for i in range(len(xd) - lag))
    n2 = sum(xd[i] * (xd[i + lag] - xd[i]) for i in range(len(xd) - lag))
    r_direct = -(n2 / d2) if d2 > 0 else float("nan")
    r_ar1 = (1.0 - phi ** lag) if phi > 0 else float("nan")

    return {"phi": phi, "hl": hl, "r_direct": r_direct, "r_ar1": r_ar1}


def run(seed, qsize, arm):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = make_mm(arm, qsize)

    n_fill = 0
    n_mm_fill = 0
    vol_total = 0.0
    vol_mm = 0.0
    mm_spreads = []
    q_all = []
    q_sampled = []

    t = 0.0
    while t < T:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
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

        q_all.append(mm.q)
        if int(t) % AR1_DT == 0:
            q_sampled.append(mm.q)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mmb = None
        mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mmb = o.price if mmb is None else max(mmb, o.price)
            else:
                mma = o.price if mma is None else min(mma, o.price)
        if mmb is not None and mma is not None:
            mm_spreads.append(mma - mmb)

    st = series_stats(q_sampled)
    out = {
        "fill_share": 100.0 * n_mm_fill / n_fill if n_fill else 0.0,
        "vol_share": 100.0 * vol_mm / vol_total if vol_total else 0.0,
        "mm_fills": float(n_mm_fill),
        "mm_spread": (statistics.median(mm_spreads) if mm_spreads
                      else float("nan")),
        "sd_q": statistics.stdev(q_all) if len(q_all) > 1 else 0.0,
        "max_abs_q": max(abs(v) for v in q_all) if q_all else 0.0,
    }
    out.update(st if st else {"phi": float("nan"), "hl": None,
                              "r_direct": float("nan"),
                              "r_ar1": float("nan")})
    return out


def mean_se(vals):
    v = [x for x in vals if x is not None
         and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    m = statistics.mean(v)
    se = (statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, se, len(v)


def measure(qsize):
    """Three paired arms on identical seeds: skew, FLAT (primary control)"""
    per = {a: [] for a in ARMS}
    diffs = {"diff": [], "diff_ar1": [], "confound": []}
    t0 = time.time()
    for s in SEEDS:
        got = {a: run(s, qsize, a) for a in ARMS}
        for a in ARMS:
            per[a].append(got[a])
        diffs["diff"].append(got["skew"]["r_direct"] - got["flat"]["r_direct"])
        diffs["diff_ar1"].append(got["skew"]["r_ar1"] - got["flat"]["r_ar1"])
        diffs["confound"].append(got["flat"]["r_direct"]
                                 - got["gamma0"]["r_direct"])

    out = {"qsize": qsize, "secs": time.time() - t0}
    for arm in ARMS:
        rows = per[arm]
        for key in rows[0]:
            m, se, n = mean_se([r[key] for r in rows])
            out["%s_%s" % (arm, key)] = m
            out["%s_%s_se" % (arm, key)] = se
            out["%s_%s_n" % (arm, key)] = n
    for tag, vals in diffs.items():
        m, se, n = mean_se(vals)
        out[tag], out[tag + "_se"], out[tag + "_n"] = m, se, n
    return out


def fmt_hl(m, n):
    if n == 0 or (isinstance(m, float) and math.isnan(m)):
        return "  none stationary"
    return "%8.0fs (%.1fh)" % (m, m / 3600.0)


def main():
    print("=" * 126)
    print("Does skewing control inventory?; three paired arms at fixed C, "
          "over the clean lever range")
    print("=" * 126)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping."
          % (LAM, P_MARKET, LIFE, DISP))
    print("k=%.5f. gamma_cap=%.2e (C=%.2f, the binding gate-2 viability cap)."
          % (K, GAMMA_CAP, GAMMA_CAP * C_PER_GAMMA))
    print("%d seeds x %.0fs, paired on identical seeds. H = %.0fs fixed. "
          "Train slice = first %d%%."
          % (len(SEEDS), T, H, int(100 * TRAIN_FRAC)))
    print("DEFAULT_GAMMA=%g and DEFAULT_QUOTE_SIZE=%g untouched; both are "
          "constructor arguments."
          % (DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE))
    print("")
    print("  Skew    gamma_cap, normal AS quoting.")
    print("  FLAT    gamma_cap, reservation price forced to the mid. Same "
          "width, same flow, no skew.")
    print("          Primary control; R_skew - R_flat is computed against "
          "this.")
    print("  GAMMA0  gamma=%.0e, the literal 'set gamma = 0'. Removes the "
          "skew and narrows the quote," % FLAT_GAMMA)
    print("          so it is NOT a valid control. Retained because the "
          "FLAT-GAMMA0 gap is the spread")
    print("          confound, measured rather than argued.")
    print("")
    print("Scope: quote_size is clean only to 0.02 BTC, so this spans 6.1% -> "
          "13.8% volume share --")
    print("entirely minority. The minority-vs-majority contrast is "
          "unreachable on a calibrated book")
    print("(non-MM liquidity is the calibration) and is dropped as a claim. "
          "R_skew - R_flat below is a")
    print("bound over a 2.3x span, not evidence of a participation gradient "
          "in either direction.")
    print("")

    rows = []
    for qs in QUOTE_SIZES:
        r = measure(qs)
        rows.append(r)
        print("  quote_size=%.3f  %3.0fs   vol share S=%5.2f%% F=%5.2f%% "
              "G0=%5.2f%%   phi_flat=%.6f   R_skew-R_flat=%+.4f"
              % (qs, r["secs"], r["skew_vol_share"], r["flat_vol_share"],
                 r["gamma0_vol_share"], r["flat_phi"], r["diff"]), flush=True)
    print("")

    print("=" * 126)
    print("0. Control validity; FLAT must match skew on width and flow, and "
          "GAMMA0 must not")
    print("=" * 126)
    print("  %11s %26s %26s %26s"
          % ("quote_size", "MM spread $  S / F / G0",
             "vol share %  S / F / G0", "fills/day  S / F / G0"))
    for r in rows:
        print("  %11.3f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.0f %8.0f %8.0f"
              % (r["qsize"], r["skew_mm_spread"], r["flat_mm_spread"],
                 r["gamma0_mm_spread"], r["skew_vol_share"],
                 r["flat_vol_share"], r["gamma0_vol_share"],
                 r["skew_mm_fills"], r["flat_mm_fills"],
                 r["gamma0_mm_fills"]))
    print("")
    print("  If skew and FLAT agree on spread and volume share, the control "
          "isolates the skew and")
    print("  nothing else. If GAMMA0 differs from FLAT, that difference is "
          "the spread confound --")
    print("  the reason gamma=0 alone cannot be used as the control.")

    print("")
    print("=" * 126)
    print("1. Is inventory reverting because of skewing, or because of flow "
          "mixing?  -- the load-bearing result")
    print("=" * 126)
    print("  %11s %11s %12s %12s %12s %19s %19s"
          % ("quote_size", "vol share", "phi_skew", "phi_flat", "phi_gamma0",
             "half-life skew", "half-life FLAT"))
    for r in rows:
        print("  %11.3f %10.2f%% %12.6f %12.6f %12.6f %19s %19s"
              % (r["qsize"], r["skew_vol_share"], r["skew_phi"],
                 r["flat_phi"], r["gamma0_phi"],
                 fmt_hl(r["skew_hl"], r["skew_hl_n"]),
                 fmt_hl(r["flat_hl"], r["flat_hl_n"])))
    print("")
    print("  phi_flat is the lag-1 inventory autocorrelation with the MM "
          "quoting the same width but")
    print("  not skewing. phi_flat ~ 1 means inventory is a random walk with "
          "no restoring force, so")
    print("  whatever reversion the skewing MM shows is attributable to "
          "skewing. phi_flat materially")
    print("  below 1 means flow mixing reverts inventory on its own, and the "
          "sd(q)-vs-gamma")
    print("  insensitivity reported in d5d134c has nothing to do with "
          "participation.")

    print("")
    print("=" * 126)
    print("2. Skew effectiveness; R_skew - R_flat at H = %.0fs, paired "
          "across identical seeds" % H)
    print("=" * 126)
    print("  %11s %11s %17s %17s %20s"
          % ("quote_size", "vol share", "R_skew", "R_flat",
             "R_skew - R_flat"))
    for r in rows:
        print("  %11.3f %10.2f%% %8.4f+-%-7.4f %8.4f+-%-7.4f "
              "%10.4f+-%-8.4f"
              % (r["qsize"], r["skew_vol_share"],
                 r["skew_r_direct"], r["skew_r_direct_se"],
                 r["flat_r_direct"], r["flat_r_direct_se"],
                 r["diff"], r["diff_se"]))
    print("")
    print("  R_direct = -slope of (q[t+H] - q[t]) on demeaned q[t]. "
          "Assumption-free, primary.")
    print("  The difference column is a paired statistic: computed per seed, "
          "then averaged, so its")
    print("  standard error is the SE of the paired differences, not of the "
          "two levels separately.")
    print("")
    print("  Same quantity via the committed AR(1) convention, "
          "R_ar1 = 1 - phi^(H/%ds):" % AR1_DT)
    print("  %11s %17s %17s %20s"
          % ("quote_size", "R_skew (ar1)", "R_flat (ar1)", "difference"))
    for r in rows:
        print("  %11.3f %8.4f+-%-7.4f %8.4f+-%-7.4f %10.4f+-%-8.4f"
              % (r["qsize"], r["skew_r_ar1"], r["skew_r_ar1_se"],
                 r["flat_r_ar1"], r["flat_r_ar1_se"],
                 r["diff_ar1"], r["diff_ar1_se"]))
    lo = rows[0]
    hi = rows[-1]
    print("")
    print("  bound over the clean range: across volume share %.2f%% -> "
          "%.2f%% (a %.1fx span),"
          % (lo["skew_vol_share"], hi["skew_vol_share"],
             hi["skew_vol_share"] / lo["skew_vol_share"]
             if lo["skew_vol_share"] else float("nan")))
    print("  R_skew - R_flat moves %+.4f -> %+.4f. A 2.3x span cannot "
          "support a general claim about"
          % (lo["diff"], hi["diff"]))
    print("  participation in either direction, and none is made.")

    print("")
    print("=" * 126)
    print("2b. The spread confound, measured; FLAT minus GAMMA0, two arms "
          "that both have ZERO skew")
    print("=" * 126)
    print("  %11s %17s %17s %20s %18s"
          % ("quote_size", "R_flat", "R_gamma0", "FLAT - GAMMA0",
             "naive diff vs G0"))
    for r in rows:
        naive = r["skew_r_direct"] - r["gamma0_r_direct"]
        print("  %11.3f %8.4f+-%-7.4f %8.4f+-%-7.4f %10.4f+-%-8.4f %18.4f"
              % (r["qsize"], r["flat_r_direct"], r["flat_r_direct_se"],
                 r["gamma0_r_direct"], r["gamma0_r_direct_se"],
                 r["confound"], r["confound_se"], naive))
    print("")
    print("  Both arms skew by construction ZERO, so any nonzero FLAT-GAMMA0 "
          "gap is caused purely by")
    print("  the quoted width and the flow it wins; it is the bias that "
          "using gamma=0 as the control")
    print("  would have injected. The last column is what R_skew - R_flat "
          "would have been reported as")
    print("  had GAMMA0 been used as the control; compare it against the "
          "primary column above.")

    print("")
    print("=" * 126)
    print("3. THE MM still functions at each level, and sd(q) for "
          "completeness only")
    print("=" * 126)
    print("  %11s %12s %12s %12s %11s %13s %11s %13s"
          % ("quote_size", "fill share", "fills/day", "MM spread",
             "sd(q) skew", "sd(q)/qsize", "max|q|", "max|q|/qsize"))
    for r in rows:
        print("  %11.3f %11.2f%% %12.1f %12.2f %11.5f %13.3f %11.5f %13.3f"
              % (r["qsize"], r["skew_fill_share"], r["skew_mm_fills"],
                 r["skew_mm_spread"], r["skew_sd_q"],
                 r["skew_sd_q"] / r["qsize"], r["skew_max_abs_q"],
                 r["skew_max_abs_q"] / r["qsize"]))
    print("")
    print("  sd(q) scales near-linearly with quote_size by construction; "
          "each fill moves inventory")
    print("  by quote_size while fill-count share barely moves. The "
          "normalized column is near-flat")
    print("  for that reason and is not evidence of anything. "
          "R_skew - R_flat is scale-free.")

    print("")
    print("=" * 126)
    print("What is dropped, and why it is a result rather than a failure")
    print("=" * 126)
    print("  The minority-participant claim is withdrawn. Majority "
          "participation is unreachable on a")
    print("  calibrated book because non-MM liquidity is the calibration, so "
          "the old book's 64% volume")
    print("  share was a symptom of the mis-scaling the clipped "
          "recalibration fixed. The 1.3x-vs-7x")
    print("  sd(q) contrast conflates participation with CALIBRATION and no "
          "lever decouples them on one")
    print("  book. That is a methodological result about this class of "
          "experiment, and it is the honest")
    print("  thing to record in place of the claim it retires.")


if __name__ == "__main__":
    main()
