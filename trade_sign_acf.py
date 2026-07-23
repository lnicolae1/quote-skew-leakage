# trade_sign_acf.py: trade-sign ACF to lag 1000+ and DFA/Hurst: power-law long memory or not?
# - aggTrade only (no book); sign: m=true -> sell-initiated (-1), m=false -> buy-initiated (+1)
# - dedup across the 07-11 local/cloud overlap by aggTrade id "a"

import json
import math
import sys

LOCAL_FILES = [
    r"market_data\btcusd_2026-07-05.jsonl",
    r"market_data\btcusd_2026-07-06.jsonl",
    r"market_data\btcusd_2026-07-07.jsonl",
    r"market_data\btcusd_2026-07-08.jsonl",
    r"market_data\btcusd_2026-07-09.jsonl",
    r"market_data\btcusd_2026-07-10.jsonl",
    r"market_data\btcusd_2026-07-11.jsonl",
]
CLOUD_FILES = [
    r"market_data_cloud\btcusd_2026-07-11.jsonl",
    r"market_data_cloud\btcusd_2026-07-12.jsonl",
    r"market_data_cloud\btcusd_2026-07-13.jsonl",
    r"market_data_cloud\btcusd_2026-07-14.jsonl",
    r"market_data_cloud\btcusd_2026-07-15.jsonl",
]
ALL_FILES = LOCAL_FILES + CLOUD_FILES

LAGS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]


def extract_aggtrades(path):
    """Yield (agg_id, trade_time_ms, sign) per aggTrade record."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' not in line:
                continue
            rec = json.loads(line)
            data = rec.get("data")
            if not data or data.get("e") != "aggTrade":
                continue
            sign = -1 if data["m"] else 1
            out.append((data["a"], data["T"], sign))
    return out


def dedup_and_sort(all_trades):
    seen = {}
    for a, T, sign in all_trades:
        if a not in seen:
            seen[a] = (T, sign)
    items = sorted(seen.items(), key=lambda kv: kv[0])
    return items


def acf(x, lags):
    n = len(x)
    mean = sum(x) / n
    c0 = sum((v - mean) ** 2 for v in x) / n
    result = {}
    for lag in lags:
        if lag >= n:
            result[lag] = None
            continue
        cov = sum((x[i] - mean) * (x[i + lag] - mean) for i in range(n - lag)) / n
        result[lag] = cov / c0
    return result


def dense_acf(x, max_lag):
    """rho(j) for every j = 1..max_lag (needed by the Bartlett SE)."""
    n = len(x)
    mean = sum(x) / n
    centered = [v - mean for v in x]
    c0 = sum(v * v for v in centered) / n
    rho_dense = [0.0] * (max_lag + 1)
    for lag in range(1, max_lag + 1):
        cov = sum(a * b for a, b in zip(centered, centered[lag:])) / n
        rho_dense[lag] = cov / c0
    return rho_dense


def bartlett_se(rho_dense, lags, n):
    """Bartlett SE: sqrt((1 + 2 * sum_{j<k} rho(j)^2) / n)."""
    se = {}
    for k in lags:
        cumsum_sq = sum(rho_dense[j] ** 2 for j in range(1, k))
        se[k] = math.sqrt((1 + 2 * cumsum_sq) / n)
    return se


def linreg(xs, ys):
    """Ordinary least squares slope+intercept for y = a + b*x"""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


def dfa(x, box_sizes):
    """Detrended fluctuation analysis; returns [(n, F(n))]."""
    n_total = len(x)
    mean = sum(x) / n_total
    profile = [0.0] * n_total
    running = 0.0
    for i, v in enumerate(x):
        running += (v - mean)
        profile[i] = running

    results = []
    for n in box_sizes:
        n_segments = n_total // n
        if n_segments < 2:
            continue
        idx = list(range(n))
        sx = sum(idx)
        sxx = sum(i * i for i in idx)
        denom = n * sxx - sx * sx

        sq_fluct_sum = 0.0
        seg_count = 0

        for s in range(n_segments):
            seg = profile[s * n:(s + 1) * n]
            sy = sum(seg)
            sxy = sum(i * v for i, v in zip(idx, seg))
            b = (n * sxy - sx * sy) / denom
            a = (sy - b * sx) / n
            resid_sq = sum((v - (a + b * i)) ** 2 for i, v in zip(idx, seg))
            sq_fluct_sum += resid_sq / n
            seg_count += 1

        # backward segments too, so the tail end is used
        for s in range(n_segments):
            seg = profile[n_total - (s + 1) * n: n_total - s * n]
            sy = sum(seg)
            sxy = sum(i * v for i, v in zip(idx, seg))
            b = (n * sxy - sx * sy) / denom
            a = (sy - b * sx) / n
            resid_sq = sum((v - (a + b * i)) ** 2 for i, v in zip(idx, seg))
            sq_fluct_sum += resid_sq / n
            seg_count += 1

        F_n = math.sqrt(sq_fluct_sum / seg_count)
        results.append((n, F_n))
    return results


def log_spaced_ints(lo, hi, count):
    out = []
    for i in range(count):
        v = lo * (hi / lo) ** (i / (count - 1))
        iv = int(round(v))
        if not out or iv > out[-1]:
            out.append(iv)
    return out


def main():
    print("Reading aggTrade records from 12 btcusd_ spot files...")
    all_trades = []
    for path in ALL_FILES:
        n_before = len(all_trades)
        trades = extract_aggtrades(path)
        all_trades.extend(trades)
        print("  " + path + ": " + str(len(trades)) + " aggTrade records")

    print("\nTotal raw aggTrade records read (pre-dedup): " + str(len(all_trades)))

    items = dedup_and_sort(all_trades)
    n_dup = len(all_trades) - len(items)
    print("Duplicate agg IDs removed (local/cloud 07-11 overlap): " + str(n_dup))
    print("Unique trades after dedup, sorted by agg id: " + str(len(items)))

    signs = [sign for (a, (T, sign)) in items]
    n = len(signs)
    n_pos = sum(1 for s in signs if s == 1)
    n_neg = n - n_pos
    print("\nSanity check; sign balance: +1 (buy-initiated): " + str(n_pos) +
          " (" + str(round(100 * n_pos / n, 2)) + "%), "
          "-1 (sell-initiated): " + str(n_neg) +
          " (" + str(round(100 * n_neg / n, 2)) + "%)")

    # ACF
    print("\n=== Trade-sign ACF, lags 1..1000 ===")
    rho = acf(signs, LAGS)

    print("Computing dense ACF at every integer lag 1.." + str(max(LAGS)) +
          " for Bartlett SE (this is the slow part)...")
    rho_dense = dense_acf(signs, max(LAGS))
    se = bartlett_se(rho_dense, LAGS, n)

    print(f"{'lag':>6} {'rho(lag)':>12} {'Bartlett SE':>12} {'|rho|>1.96*SE':>14} "
          f"{'pairs (N-lag)':>15} {'indep blocks (N//lag)':>22}")
    for lag in LAGS:
        r = rho[lag]
        r_str = "n/a" if r is None else f"{r:.4f}"
        se_str = f"{se[lag]:.4f}"
        sig = "n/a" if r is None else ("yes" if abs(r) > 1.96 * se[lag] else "NO")
        print(f"{lag:>6} {r_str:>12} {se_str:>12} {sig:>14} {n - lag:>15} {n // lag:>22}")

    # local log-log slopes
    print("\n=== Local log-log slope between consecutive measured lags ===")
    print(f"{'interval':>14} {'slope':>10}")
    for (l1, l2) in zip(LAGS[:-1], LAGS[1:]):
        r1, r2 = rho[l1], rho[l2]
        if r1 is None or r2 is None or r1 <= 0 or r2 <= 0:
            print(f"{str(l1)+'->'+str(l2):>14} {'n/a (rho<=0)':>10}")
            continue
        slope = (math.log(r2) - math.log(r1)) / (math.log(l2) - math.log(l1))
        print(f"{str(l1)+'->'+str(l2):>14} {slope:>10.4f}")

    # DFA / Hurst
    print("\n=== DFA on the sign series ===")
    max_box = n // 4
    box_sizes = log_spaced_ints(10, max_box, 20)
    print("Box sizes used: " + str(box_sizes))
    fn = dfa(signs, box_sizes)
    print(f"{'n (box size)':>14} {'F(n)':>14}")
    for n_box, F_n in fn:
        print(f"{n_box:>14} {F_n:>14.6f}")

    print("\n=== DFA local log-log slope between consecutive box sizes ===")
    print("(flat across box sizes = single scaling regime;")
    print(" drift or steps = one global H")
    print(" does not fit)")
    print(f"{'interval':>18} {'local slope':>12}")
    for (n1, F1), (n2, F2) in zip(fn[:-1], fn[1:]):
        local_slope = (math.log(F2) - math.log(F1)) / (math.log(n2) - math.log(n1))
        print(f"{str(n1)+'->'+str(n2):>18} {local_slope:>12.4f}")

    log_ns = [math.log(n_box) for n_box, _ in fn]
    log_Fs = [math.log(F_n) for _, F_n in fn]
    a, H = linreg(log_ns, log_Fs)
    gamma = 2 * (1 - H)
    print("\nDFA fit over all box sizes (single global H; unreliable")
    print("if the local slopes above drift):")
    print("  Hurst exponent H = " + f"{H:.4f}")
    print("  gamma = 2(1-H) = " + f"{gamma:.4f}")


if __name__ == "__main__":
    main()
