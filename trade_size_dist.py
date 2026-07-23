# trade_size_dist.py: empirical trade-size distribution from the real aggTrade stream
# - size = |q| of each aggTrade; read-only, streamed line by line
# - dedup of the local/cloud 07-11 overlap by aggTrade id "a" (the depth splice uid does not apply to trades)
# - saves the raw samples to trade_sizes.bin for the noise traders' bootstrap

import array
import json
import math
import os
import sys

BASE = r"C:\Users\lnico\Desktop"

LOCAL_FILES = [
    "market_data\\btcusd_2026-07-05.jsonl",
    "market_data\\btcusd_2026-07-06.jsonl",
    "market_data\\btcusd_2026-07-07.jsonl",
    "market_data\\btcusd_2026-07-08.jsonl",
    "market_data\\btcusd_2026-07-09.jsonl",
    "market_data\\btcusd_2026-07-10.jsonl",
    "market_data\\btcusd_2026-07-11.jsonl",
]
CLOUD_FILES = [
    "market_data_cloud\\btcusd_2026-07-11.jsonl",
    "market_data_cloud\\btcusd_2026-07-12.jsonl",
    "market_data_cloud\\btcusd_2026-07-13.jsonl",
    "market_data_cloud\\btcusd_2026-07-14.jsonl",
    "market_data_cloud\\btcusd_2026-07-15.jsonl",
]
ALL_FILES = LOCAL_FILES + CLOUD_FILES


def extract_sizes(path, seen):
    """Stream one file into seen (agg_id -> size); returns (records, new)."""
    n_file = 0
    n_new = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '"aggTrade"' not in line:
                continue
            rec = json.loads(line)
            data = rec.get("data")
            if not data or data.get("e") != "aggTrade":
                continue
            n_file += 1
            a = data["a"]
            if a in seen:
                continue
            # "q": absolute traded quantity, as a string
            seen[a] = abs(float(data["q"]))
            n_new += 1
    return n_file, n_new


def percentile(sorted_vals, p):
    """Linear-interpolation percentile (p in [0,100]) on an ascending list"""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def hill_alpha(sorted_desc, k):
    """Hill tail index on the top-k order statistics: alpha = k / sum ln(X_(i) / X_(k+1))."""
    u = sorted_desc[k]
    if u <= 0:
        return None, u
    s = 0.0
    for i in range(k):
        s += math.log(sorted_desc[i] / u)
    if s <= 0:
        return None, u
    return k / s, u


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    lines_out = []

    def emit(s=""):
        print(s)
        lines_out.append(s)

    emit("=== trade_size_dist.py; empirical trade-size distribution ===")
    emit("Streaming aggTrade 'q' from 12 btcusd_ spot files "
         "(local 07-05..07-11 + cloud 07-11..07-15), dedup by aggTrade id.")
    emit("")

    seen = {}
    total_records = 0
    for rel in ALL_FILES:
        path = os.path.join(BASE, rel)
        n_file, n_new = extract_sizes(path, seen)
        total_records += n_file
        emit("  " + rel + ": " + str(n_file) + " aggTrade records, " +
             str(n_new) + " new after dedup")

    sizes = list(seen.values())
    n = len(sizes)
    n_dup = total_records - n
    emit("")
    emit("Total raw aggTrade records (pre-dedup): " + str(total_records))
    emit("Duplicate agg IDs removed (local/cloud 07-11 overlap): " + str(n_dup))
    emit("Unique trades (sample count for the distribution): " + str(n))

    if n == 0:
        emit("No trades found; aborting.")
        return

    sizes.sort()
    total = sum(sizes)
    mean = total / n
    var = sum((x - mean) ** 2 for x in sizes) / n
    std = math.sqrt(var)
    median = percentile(sizes, 50)

    emit("")
    emit("=== Summary statistics (BTC) ===")
    emit("  count : " + str(n))
    emit("  mean  : " + format(mean, ".8f"))
    emit("  std   : " + format(std, ".8f") + "  (population)")
    emit("  min   : " + format(sizes[0], ".8f"))
    emit("  max   : " + format(sizes[-1], ".8f"))
    emit("  median: " + format(median, ".8f"))

    emit("")
    emit("=== Percentiles (BTC) ===")
    pcts = [1, 5, 25, 50, 75, 90, 95, 99, 99.9]
    pvals = {}
    for p in pcts:
        v = percentile(sizes, p)
        pvals[p] = v
        emit("  p" + format(p, ">5") + " : " + format(v, ".8f"))

    # tail
    emit("")
    emit("=== Tail characterization ===")
    k = max(1, int(round(0.01 * n)))
    sorted_desc = sizes[::-1]
    top1_vals = sorted_desc[:k]
    top1_mean = sum(top1_vals) / k
    emit("  top-1% cutoff k (count): " + str(k))
    emit("  p99 / median ratio        : " + format(pvals[99] / median, ".2f"))
    emit("  p99.9 / median ratio      : " + format(pvals[99.9] / median, ".2f"))
    emit("  max / median ratio        : " + format(sizes[-1] / median, ".2f"))
    emit("  mean(top 1%) / median     : " + format(top1_mean / median, ".2f"))
    emit("  mean(top 1%) / overall mean: " + format(top1_mean / mean, ".2f"))

    # Hill estimator on the top 1% (plus nearby k)
    emit("")
    emit("  Hill tail-index estimates (P(X>x) ~ x^-alpha; smaller alpha = fatter):")
    for frac in (0.005, 0.01, 0.02):
        kk = max(2, int(round(frac * n)))
        if kk + 1 >= n:
            continue
        alpha, u = hill_alpha(sorted_desc, kk)
        if alpha is None:
            emit("    top " + format(frac * 100, ".1f") + "% (k=" + str(kk) +
                 "): degenerate (ties at threshold)")
        else:
            emit("    top " + format(frac * 100, ".1f") + "% (k=" + str(kk) +
                 ", threshold u=" + format(u, ".6f") + " BTC): alpha = " +
                 format(alpha, ".3f"))
    emit("")
    emit("  Reference alpha ~= 1.15 for the size tail (1.1-1.2 consistent;")
    emit("  much larger alpha = thinner tail than")
    emit("  assumed).")

    # raw samples (float64 binary)
    samples_path = os.path.join(out_dir, "trade_sizes.bin")
    arr = array.array("d", sizes)
    with open(samples_path, "wb") as fbin:
        arr.tofile(fbin)
    emit("")
    emit("=== Saved samples ===")
    emit("  file  : " + samples_path)
    emit("  format: raw float64 (array('d')), native byte order, no header")
    emit("  count : " + str(len(arr)) + " values (== unique trade count above)")
    emit("  bytes : " + str(len(arr) * arr.itemsize))
    emit("  reload (stdlib): a=array.array('d'); a.fromfile(open(path,'rb'), N)")
    emit("  samples saved sorted ascending; order carries no meaning")
    emit("        for resampling.")

    # results file
    results_path = os.path.join(out_dir, "trade_size_results.txt")
    with open(results_path, "w", encoding="utf-8") as ftxt:
        ftxt.write("\n".join(lines_out) + "\n")
    print("")
    print("Wrote summary to: " + results_path)


if __name__ == "__main__":
    main()
