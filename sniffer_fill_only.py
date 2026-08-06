# sniffer_fill_only.py: fill-only sniffer (vacuity guard): inventory from integrated public fills, no skew features
# - oracle: q = sum of signed MM fills using counterparty identity; must give R2 = 1.0 (completeness check on prints)
# - sniffer: public prints only (time, price, size, aggressor side); no counterparty field
# - features: signed volume integrated from t=0 plus trailing windows; windows-only model is the floor quoted
# - bands: R2 ~ 1.0 -> attribution clean, do not proceed; R2 ~ 0.3-0.6 -> real ambiguity, proceed;
#   other ranges have no instruction and are flagged

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

T = 86400.0
N_SEC = int(T)
SEEDS = list(range(15))
LAM, P_MARKET, LIFE, DISP = 1.804, 0.0126, 720.0, 0.0055
MM_ID = "MM"
TRAIN_FRAC = 0.70
WINDOWS = [60, 300, 900, 3600]
SAMPLE_EVERY = 60

K_BRACKET = [(0.17763, "Option 1 near-touch $0.25-5, R2=0.9929  [PRIMARY]"),
             (0.16230, "Option 1 near-touch $0.25-10, R2=0.9954"),
             (0.12429, "Option 1 full swept range, R2=0.9584")]


class PublicPrint(object):
    """One public trade print; no counterparty field."""
    __slots__ = ("t", "price", "size", "aggressor_side")

    def __init__(self, t, price, size, aggressor_side):
        self.t = t
        self.price = price
        self.size = size
        self.aggressor_side = aggressor_side


assert "counterparty_id" not in PublicPrint.__slots__, \
    "information boundary violated: PublicPrint must not carry identity"
assert not hasattr(PublicPrint(0, 0, 0, "buy"), "__dict__"), \
    "PublicPrint must use __slots__ so no identity field can be attached"


# the sniffer's input: a list of PublicPrint
def build_features(prints, n_sec):
    """Per-second features from prints only: [1, cum, w_60, w_300, w_900, w_3600]. Returns (rows, cum)."""
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
    rows = []
    for i in range(1, n_sec + 1):
        row = [1.0, cum[i]]
        for w in WINDOWS:
            j = i - w
            row.append(cum[i] - (cum[j] if j > 0 else 0.0))
        rows.append(row)
    return rows, cum


def ols(X, y):
    """OLS via normal equations; rows include a leading 1."""
    p = len(X[0])
    A = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for i in range(len(X)):
        xi, yi = X[i], y[i]
        for a in range(p):
            xa = xi[a]
            rhs[a] += xa * yi
            Aa = A[a]
            for b in range(a, p):
                Aa[b] += xa * xi[b]
    for a in range(p):
        for b in range(a):
            A[a][b] = A[b][a]
    M = [A[r][:] + [rhs[r]] for r in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if M[c][c] == 0:
            continue
        for r in range(p):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for cc in range(c, p + 1):
                M[r][cc] -= f * M[c][cc]
    return [M[r][p] / M[r][r] if M[r][r] else 0.0 for r in range(p)]


def fit_score(rows, y, cols):
    """Time split: fit on the first TRAIN_FRAC, score out of sample."""
    n = len(y)
    cut = int(TRAIN_FRAC * n)
    Xtr = [[r[c] for c in cols] for r in rows[:cut]]
    Xte = [[r[c] for c in cols] for r in rows[cut:]]
    ytr, yte = y[:cut], y[cut:]
    beta = ols(Xtr, ytr)
    pred = [sum(b * x for b, x in zip(beta, row)) for row in Xte]
    mte = statistics.mean(yte)
    ss_res = sum((a - b) ** 2 for a, b in zip(yte, pred))
    ss_tot = sum((a - mte) ** 2 for a in yte)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, math.sqrt(ss_res / len(yte))


def score_oracle(oracle, y):
    """Oracle q = sum of signed MM fills, scored on the test slice; plus worst absolute discrepancy."""
    n = len(y)
    cut = int(TRAIN_FRAC * n)
    yte, ote = y[cut:], oracle[cut:]
    mte = statistics.mean(yte)
    ss_res = sum((a - b) ** 2 for a, b in zip(yte, ote))
    ss_tot = sum((a - mte) ** 2 for a in yte)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    max_abs = max(abs(a - b) for a, b in zip(y, oracle))
    return r2, max_abs


def run(seed, k):
    _p, vf, eng, noise, informed = make_world(seed, T, LAM, P_MARKET, LIFE,
                                              DISP, "join")
    seed_ids = set(eng.orders.keys())
    mm = MarketMaker(horizon=T, k=k)

    prints = []
    q_series = []
    oracle_series = []
    cum_mm = 0.0

    n_noise_limit = n_noise_market = 0
    n_fill_total = n_fill_mm = 0
    vol_total = vol_mm = 0.0
    mm_passive_share, mm_near_share = [], []

    t = 0.0
    while t < T:
        t += 1.0
        for recs, is_noise in ((noise.run_until(eng, t), True),
                               (informed.run_until(eng, t, vf), False)):
            for r in recs:
                if is_noise:
                    if r.order_type == "limit":
                        n_noise_limit += 1
                    else:
                        n_noise_market += 1
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                s = 1.0 if r.side == "buy" else -1.0
                for f in fills:
                    prints.append(PublicPrint(t, f.price, f.size, r.side))
                    n_fill_total += 1
                    vol_total += f.size
                    if f.counterparty_id == MM_ID:
                        cum_mm += -s * f.size
                        n_fill_mm += 1
                        vol_mm += f.size
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)
        q_series.append(mm.q)
        oracle_series.append(cum_mm)

        if t % SAMPLE_EVERY:
            continue
        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        mid = 0.5 * (bb + ba)
        lo, hi = mid * (1 - 1e-4), mid * (1 + 1e-4)
        sz_all = sz_mm = near_all = near_mm = 0.0
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

    # sniffer sees only prints from here on
    rows, _cum = build_features(prints, N_SEC)
    n = min(len(rows), len(q_series))
    rows, y = rows[:n], q_series[:n]

    # [0]=const [1]=cumulative from 0 [2..]=trailing windows
    ncols = len(rows[0])
    r2_uni, rmse_uni = fit_score(rows, y, [0, 1])
    r2_mul, rmse_mul = fit_score(rows, y, list(range(ncols)))
    # windows-only: drops the cumulative term (a drifting random walk while q mean-reverts)
    r2_win, rmse_win = fit_score(rows, y, [0] + list(range(2, ncols)))
    r2_orc, max_gap = score_oracle(oracle_series[:n], y)

    return {
        "r2_oracle": r2_orc, "oracle_max_gap": max_gap,
        "r2_uni": r2_uni, "rmse_uni": rmse_uni,
        "r2_mul": r2_mul, "rmse_mul": rmse_mul,
        "r2_win": r2_win, "rmse_win": rmse_win,
        "sd_q": statistics.stdev(y),
        "noise_limit_pct": 100.0 * n_noise_limit / max(1, n_noise_limit + n_noise_market),
        "mm_fill_share": 100.0 * n_fill_mm / max(1, n_fill_total),
        "mm_vol_share": 100.0 * vol_mm / vol_total if vol_total else 0.0,
        "mm_passive_share": statistics.mean(mm_passive_share) if mm_passive_share else 0.0,
        "mm_near_share": statistics.mean(mm_near_share) if mm_near_share else 0.0,
        "n_prints": len(prints),
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
    print("=" * 118)
    print("Fill-only sniffer (vacuity guard)")
    print("=" * 118)
    print("Working point: lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping. MM present."
          % (LAM, P_MARKET, LIFE, DISP))
    print("%d seeds x %.0fs per k. gamma=DEFAULT_GAMMA=%g (not chosen). "
          "DEFAULT_K untouched." % (len(SEEDS), T, DEFAULT_GAMMA))
    print("Time split %d/%d, out-of-sample metrics only, R2 against the "
          "test-set mean."
          % (int(100 * TRAIN_FRAC), int(100 * (1 - TRAIN_FRAC))))
    print("")
    print("Sniffer input per trade print: time, price, size, aggressor side.")
    print("No counterparty identity (PublicPrint has no such field;")
    print("build_features() receives only the prints list).")
    print("Features: integrated signed volume from t=0, plus trailing-window "
          "integrals at %s s." % "/".join(str(w) for w in WINDOWS))
    print("No skew feature.")
    print("")

    results = []
    for k, label in K_BRACKET:
        r = measure(k)
        results.append((k, label, r))
        print("  k=%.5f measured in %.0fs" % (k, r["secs"]), flush=True)
    print("")

    print("=" * 118)
    print("Completeness check: oracle q = sum of signed MM fills (uses "
          "identity; not the sniffer)")
    print("=" * 118)
    print("  Must be R2 = 1.0 exactly; otherwise prints are dropped and the")
    print("  fill-only numbers below are invalid.")
    print("  %-12s %14s %20s %14s" % ("k", "oracle R2", "max |oracle-q| BTC",
                                      "prints/day"))
    for k, _lab, r in results:
        print("  %-12.5f %14.10f %20.3e %14.0f"
              % (k, r["r2_oracle"], r["oracle_max_gap"], r["n_prints"]))

    print("")
    print("=" * 118)
    print("Floor: fill-only out-of-sample R2 against true inventory")
    print("=" * 118)
    print("  %-10s %20s %20s %20s %11s %11s"
          % ("k", "univariate R2", "multivariate R2", "Windows-only R2",
             "RMSE BTC", "sd(q) BTC"))
    for k, _lab, r in results:
        print("  %-10.5f %9.4f +/- %-7.4f %9.4f +/- %-7.4f %9.4f +/- %-7.4f "
              "%11.5f %11.5f"
              % (k, r["r2_uni"], r["r2_uni_se"], r["r2_mul"], r["r2_mul_se"],
                 r["r2_win"], r["r2_win_se"], r["rmse_win"], r["sd_q"]))
    print("")
    print("  univariate    = q ~ integrated signed volume from t=0")
    print("                  construction, with attribution unavailable.")
    print("  multivariate  = that plus the trailing-window integrals.")
    print("  Windows-only  = trailing windows, no cumulative term (the "
          "floor quoted).")
    print("                  The cumulative term is a random walk while q")
    print("                  mean-reverts, so its train-slice slope "
          "does not generalize;")
    print("                  a negative R2 there is a specification artifact.")
    print("                  Windowed integrals are stationary:")
    print("                  strongest fill-only adversary.")
    print("  Negative out-of-sample R2 = worse than the test-set mean.")
    print("  RMSE: windows-only model, BTC; sd(q) for scale.")

    print("")
    print("=" * 118)
    print("Attribution ambiguity")
    print("=" * 118)
    print("  Limit-order posting by noise traders creates")
    print("  fill-attribution ambiguity.")
    print("  %-12s %16s %16s %16s %18s"
          % ("k", "noise limit %", "MM fill share", "MM vol share",
             "MM passive <=1bp"))
    for k, _lab, r in results:
        print("  %-12.5f %15.2f%% %15.2f%% %15.2f%% %17.2f%%"
              % (k, r["noise_limit_pct"], r["mm_fill_share"],
                 r["mm_vol_share"], r["mm_near_share"]))
    for k, _lab, r in results:
        print("  k=%.5f: MM share of all resting passive size %.2f%%"
              % (k, r["mm_passive_share"]))

    print("")
    print("=" * 118)
    print("Verdict against the bands")
    print("=" * 118)
    print("  Two bands:")
    print("    R2 ~ 1.0     -> attribution clean; nothing downstream means "
          "anything; do not proceed")
    print("    R2 ~ 0.3-0.6 -> real ambiguity; proceed and report the number")
    print("  0.6-1.0 and below 0.3: no instruction; flagged.")
    print("")
    for k, _lab, r in results:
        v = r["r2_win"]
        if v >= 0.90:
            band = "~1.0 band -> DO NOT PROCEED (add attribution noise)"
        elif 0.3 <= v <= 0.6:
            band = "0.3-0.6 band -> PROCEED"
        elif 0.6 < v < 0.90:
            band = "IN THE PLAN'S GAP (0.6-1.0): no instruction given"
        else:
            band = "BELOW 0.3: no instruction given"
        print("  k=%.5f  fill-only R2 (windows-only) = %+.4f  ->  %s"
              % (k, v, band))


if __name__ == "__main__":
    main()
