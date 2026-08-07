# sniffer_skew.py: skew-based sniffer against the undefended maker (left endpoint of the defense curve)
# - adversary sees: MM bid/ask, residual and full-book mid, clock, (T-t), public prints; no q, cash, PnL or counterparties
# - skew = (bid + ask)/2 - mid = -q * C, C = gamma * sigma^2 * (T - t)
# - models: raw skew (constant slope), normalized skew/(mid^2 * tau), multivariate (+ width, skew velocity, fill volume)
# - time split, out-of-sample R2 against the test-set mean; expected R2 ~ 1.0 for the normalized model

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_placement import make_world
from market_maker import MarketMaker, DEFAULT_GAMMA
from sniffer_fill_only import PublicPrint, ols, TRAIN_FRAC, WINDOWS
from clipped_mm_check import _resid_best

T = 86400.0
N_SEC = int(T)
SEEDS = list(range(15))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
MM_ID = "MM"
TAU_FLOOR_FRAC = 0.02
VEL_LAGS = [60, 300]

K_BRACKET = [(0.17763, "Option 1 near-touch $0.25-5, R2=0.9929  [PRIMARY]"),
             (0.16230, "Option 1 near-touch $0.25-10, R2=0.9954"),
             (0.12429, "Option 1 full swept range, R2=0.9584")]

# fill-only floor (windows-only spec), for comparison
FILL_FLOOR = {0.17763: -5.0789, 0.16230: -6.1863, 0.12429: -7.9250}


class SkewView(object):
    """Adversary's per-second view: MM bid/ask, full-book mid, residual mid, clock, tau."""

    __slots__ = ("t", "mm_bid", "mm_ask", "book_mid", "resid_mid", "tau")

    def __init__(self, t, mm_bid, mm_ask, book_mid, resid_mid, tau):
        self.t = t
        self.mm_bid = mm_bid
        self.mm_ask = mm_ask
        self.book_mid = book_mid
        self.resid_mid = resid_mid
        self.tau = tau


_FORBIDDEN = ("q", "inventory", "cash", "pnl", "counterparty_id",
              "counterparty", "n_buy_fills", "n_sell_fills")
for _f in _FORBIDDEN:
    assert _f not in SkewView.__slots__, \
        "information boundary violated: SkewView must not carry %r" % _f
assert not hasattr(SkewView(0, 0, 0, 0, 0, 0), "__dict__"), \
    "SkewView must use __slots__ so no private field can be attached"
assert "counterparty_id" not in PublicPrint.__slots__, \
    "information boundary violated: PublicPrint must not carry identity"


def build_features(views, prints, n_sec):
    """Features from the public view and prints only. Returns (rows, colnames, keep)."""
    per_sec = [0.0] * (n_sec + 1)
    for p in prints:
        s = 1.0 if p.aggressor_side == "buy" else -1.0
        idx = int(p.t)
        if 0 <= idx <= n_sec:
            per_sec[idx] += -s * p.size
    cum = [0.0] * (n_sec + 1)
    run = 0.0
    for i in range(1, n_sec + 1):
        run += per_sec[i]
        cum[i] = run

    # skew against the residual mid (what requote() prices against); full-book mid kept as a sensitivity
    skew = [0.0] * len(views)
    skew_full = [0.0] * len(views)
    for i, v in enumerate(views):
        qm = 0.5 * (v.mm_bid + v.mm_ask)
        skew[i] = qm - v.resid_mid
        skew_full[i] = qm - v.book_mid

    tau0 = views[0].tau if views else 0.0
    floor = TAU_FLOOR_FRAC * tau0

    rows, keep = [], []
    for i, v in enumerate(views):
        sk = skew[i]
        denom = v.resid_mid * v.resid_mid * v.tau
        norm = sk / denom if denom > 0 else 0.0
        dfull = v.book_mid * v.book_mid * v.tau
        norm_full = skew_full[i] / dfull if dfull > 0 else 0.0
        row = [1.0, norm, sk, v.mm_ask - v.mm_bid, norm_full]
        for L in VEL_LAGS:
            j = i - L
            row.append(sk - (skew[j] if j >= 0 else 0.0))
        idx = int(v.t)
        for w in WINDOWS:
            j = idx - w
            row.append(cum[idx] - (cum[j] if j > 0 else 0.0))
        rows.append(row)
        keep.append(v.tau >= floor)

    names = (["const", "skew_norm", "skew_raw", "spread_width",
              "skew_norm_fullmid"]
             + ["skew_vel_%ds" % L for L in VEL_LAGS]
             + ["fillvol_%ds" % w for w in WINDOWS])
    return rows, names, keep


def fit_score(rows, y, keep, cols):
    """Time split on kept samples: fit on the first TRAIN_FRAC, score out of sample."""
    idx = [i for i in range(len(y)) if keep[i]]
    n = len(idx)
    cut = int(TRAIN_FRAC * n)
    tr, te = idx[:cut], idx[cut:]
    Xtr = [[rows[i][c] for c in cols] for i in tr]
    Xte = [[rows[i][c] for c in cols] for i in te]
    ytr = [y[i] for i in tr]
    yte = [y[i] for i in te]
    beta = ols(Xtr, ytr)
    pred = [sum(b * x for b, x in zip(beta, row)) for row in Xte]
    mte = statistics.mean(yte)
    ss_res = sum((a - b) ** 2 for a, b in zip(yte, pred))
    ss_tot = sum((a - mte) ** 2 for a in yte)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, math.sqrt(ss_res / len(yte))


def run(seed, k):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    mm = MarketMaker(horizon=T, k=k)

    views, prints, q_series = [], [], []
    n_quote_updates = 0
    size_imbalance_nonzero = 0

    t = 0.0
    while t < T:
        t += 1.0
        for recs in (noise.run_until(eng, t), informed.run_until(eng, t, vf)):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                for f in fills:
                    prints.append(PublicPrint(t, f.price, f.size, r.side))
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)
        n_quote_updates += 1

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        # the MM's resting quotes as an observer reads them (quote owners are public, counterparties are not)
        mb = ma = None
        bsz = asz = 0.0
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mb = o.price if mb is None else max(mb, o.price)
                bsz += o.size
            else:
                ma = o.price if ma is None else min(ma, o.price)
                asz += o.size
        if mb is None or ma is None:
            continue
        if abs(bsz - asz) > 1e-12:
            size_imbalance_nonzero += 1
        rb = _resid_best(eng.bids, bb, True)
        ra = _resid_best(eng.asks, ba, False)
        if rb is None or ra is None or ra <= rb:
            continue
        views.append(SkewView(t, mb, ma, 0.5 * (bb + ba), 0.5 * (rb + ra),
                              mm.time_remaining_years(t)))
        q_series.append(mm.q)

    # sniffer sees only views + prints from here on
    rows, names, keep = build_features(views, prints, N_SEC)
    y = q_series
    n = min(len(rows), len(y))
    rows, y, keep = rows[:n], y[:n], keep[:n]

    c = {nm: i for i, nm in enumerate(names)}
    r2_raw, rmse_raw = fit_score(rows, y, keep, [c["const"], c["skew_raw"]])
    r2_nrm, rmse_nrm = fit_score(rows, y, keep, [c["const"], c["skew_norm"]])
    r2_fullmid, _rm = fit_score(rows, y, keep,
                                [c["const"], c["skew_norm_fullmid"]])
    # full-mid column: sensitivity only
    mul_cols = [i for i, nm in enumerate(names) if nm != "skew_norm_fullmid"]
    r2_mul, rmse_mul = fit_score(rows, y, keep, mul_cols)

    idx = [i for i in range(len(y)) if keep[i]]
    cut = int(TRAIN_FRAC * len(idx))
    sd_test = statistics.stdev([y[i] for i in idx[cut:]])

    return {
        "r2_raw": r2_raw, "rmse_raw": rmse_raw,
        "r2_norm": r2_nrm, "rmse_norm": rmse_nrm,
        "r2_mul": r2_mul, "rmse_mul": rmse_mul,
        "r2_fullmid": r2_fullmid,
        "sd_q": statistics.stdev(y),
        "sd_q_test": sd_test,
        "n_kept": float(sum(1 for x in keep if x)),
        "n_total": float(len(keep)),
        "size_imbalance_pct": 100.0 * size_imbalance_nonzero / max(1, len(views)),
        "n_prints": float(len(prints)),
    }


def measure(k):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, k)
        if per is None:
            per = {key: [] for key in x}
        for key, v in x.items():
            per[key].append(v)
    n = len(SEEDS)
    out = {"k": k, "secs": time.time() - t0}
    for key, v in per.items():
        out[key] = statistics.mean(v)
        out[key + "_se"] = (statistics.stdev(v) / math.sqrt(n)) if n > 1 else 0.0
    return out


def main():
    print("=" * 120)
    print("Skew sniffer, undefended baseline (left endpoint)")
    print("=" * 120)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x %.0fs per k. gamma=DEFAULT_GAMMA=%g (not chosen). "
          "DEFAULT_K untouched. horizon=%.0fs."
          % (len(SEEDS), T, DEFAULT_GAMMA, T))
    print("Time split %d/%d, out-of-sample only, R2 against the test-set mean."
          % (int(100 * TRAIN_FRAC), int(100 * (1 - TRAIN_FRAC))))
    print("")
    print("Sniffer sees: MM bid/ask, book mid, clock, (T-t), public prints.")
    print("Not visible to it: inventory, cash, PnL, counterparties, MM "
          "internal state.")
    print("Adversary knows gamma, sigma and (T-t).")
    print("")
    print("skew = (MM_bid + MM_ask)/2 - observed_mid.  C = "
          "gamma*sigma^2*(T-t) varies:")
    print("(T-t) decays within the test slice, which depresses raw-skew OLS;")
    print("normalized feature skew/(mid^2*tau) is the one to compare")
    print("with the predicted R2 = 1.0.")
    print("")

    results = []
    for k, label in K_BRACKET:
        r = measure(k)
        results.append((k, label, r))
        print("  k=%.5f measured in %.0fs  (kept %.0f of %.0f samples after "
              "the tau floor)"
              % (k, r["secs"], r["n_kept"], r["n_total"]), flush=True)
    print("")

    print("=" * 120)
    print("Left endpoint: skew-sniffer out-of-sample R2 against true inventory")
    print("=" * 120)
    print("  %-10s %20s %20s %20s %11s %13s"
          % ("k", "raw skew R2", "normalized R2", "multivariate R2",
             "RMSE BTC", "sd(q) test"))
    for k, _lab, r in results:
        print("  %-10.5f %9.4f +/- %-7.4f %9.6f +/- %-7.6f %9.6f +/- %-7.6f "
              "%11.3e %13.5f"
              % (k, r["r2_raw"], r["r2_raw_se"], r["r2_norm"], r["r2_norm_se"],
                 r["r2_mul"], r["r2_mul_se"], r["rmse_norm"], r["sd_q_test"]))
    print("")
    print("  raw skew     = q ~ skew, constant slope.")
    print("  Normalized   = q ~ skew/(mid^2 * tau), adversary knows "
          "mid and (T-t).")
    print("                 Left endpoint.")
    print("  multivariate = plus spread width, skew velocity (%s s) and "
          "trailing fill volume (%s s)."
          % ("/".join(str(x) for x in VEL_LAGS),
             "/".join(str(w) for w in WINDOWS)))
    print("  RMSE: normalized model, BTC. sd(q): test-slice SD,")
    print("  the scale for an R2 against the test-set mean (full-run")
    print("  sd is larger).")
    print("")
    print("  Mid sensitivity: normalized skew against the full-book "
          "mid instead of the")
    print("  residual mid:")
    for k, _lab, r in results:
        print("    k=%.5f  R2 = %+.4f +/- %.4f   (vs %+.6f with the residual "
              "mid)"
              % (k, r["r2_fullmid"], r["r2_fullmid_se"], r["r2_norm"]))
    print("  The MM cancels before observing the mid "
          "(market_maker.observe_mid), so")
    print("  an adversary using the post-requote full-book mid is reading a "
          "different quantity")
    print("  than the one the MM priced against, and loses most of the signal.")

    print("")
    print("=" * 120)
    print("Skew channel vs fill channel")
    print("=" * 120)
    print("  %-10s %22s %22s %14s"
          % ("k", "skew R2 (normalized)", "fill-only floor", "gap"))
    for k, _lab, r in results:
        f = FILL_FLOOR[k]
        print("  %-10.5f %22.4f %22.4f %14.4f"
              % (k, r["r2_norm"], f, r["r2_norm"] - f))
    print("")
    print("  fill-only floor: windows-only spec, strongest fill-only "
          "adversary measured.")
    print("  Negative floor: fill integration predicts")
    print("  worse than the test-set mean; recoverable inventory information")
    print("  arrives through the quote channel")
    print("  (which a quote-side defense can act on).")

    print("")
    print("=" * 120)
    print("Degenerate features (excluded)")
    print("=" * 120)
    print("  quote_update_rate and quoted_size_imbalance are listed as "
          "observable.")
    print("  Both are degenerate here:")
    print("    quote_update_rate     : MM requotes every second, so")
    print("                             constant.")
    print("    quoted_size_imbalance; nonzero on %.2f%% of samples (the MM "
          "posts DEFAULT_QUOTE_SIZE"
          % results[0][2]["size_imbalance_pct"])
    print("                             on both sides).")
    print("  Neither is in the multivariate model.")


if __name__ == "__main__":
    main()
