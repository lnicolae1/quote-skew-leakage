# reconstruct_book.py: rebuilds the order book from a snapshot plus diffs (U/u/pu sequence checks)
# - resumption gap: a diff's U does not continue from the last applied update
# - invisible span: a snapshot's lastUpdateId is ahead of the last applied update
# - both written as intervals to .gaps.json
# - --anchor-out / --anchor-in carry the final book state to the next day's file
# - compares the result with a live snapshot

import argparse
import json
import sys
import urllib.request

SYMBOL = "BTCUSD"
SNAPSHOT_URL = "https://api.binance.us/api/v3/depth?symbol=" + SYMBOL + "&limit=1000"


def fetch_live_snapshot():
    # live snapshot for comparison
    with urllib.request.urlopen(SNAPSHOT_URL, timeout=10) as response:
        return json.loads(response.read())


def load_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_anchor(path):
    with open(path, "r") as f:
        return json.load(f)


def save_anchor(path, anchor):
    with open(path, "w") as f:
        json.dump(anchor, f)


def apply_diff(side_dict, updates):
    for price_str, qty_str in updates:
        if float(qty_str) == 0.0:
            side_dict.pop(price_str, None)
        else:
            side_dict[price_str] = qty_str


def reconstruct(records, anchor=None):
    if anchor is not None:
        bids = dict(anchor["bids"])
        asks = dict(anchor["asks"])
        last_update_id = anchor["last_update_id"]
        last_ts = anchor["final_timestamp"]
        carried_resumption_gaps = anchor.get("cumulative_resumption_gaps", 0)
        carried_resumption_updates_lost = anchor.get("cumulative_resumption_updates_lost", 0)
        carried_invisible_spans = anchor.get("cumulative_invisible_spans", 0)
        carried_invisible_updates_lost = anchor.get("cumulative_invisible_updates_lost", 0)
        source_files = list(anchor.get("source_files", []))
        seeded_from_anchor = True
    else:
        bids = {}
        asks = {}
        last_update_id = None
        last_ts = None
        carried_resumption_gaps = 0
        carried_resumption_updates_lost = 0
        carried_invisible_spans = 0
        carried_invisible_updates_lost = 0
        source_files = []
        seeded_from_anchor = False

    resumption_gaps = []
    invisible_spans = []

    for rec in records:
        rec_ts = rec.get("local_timestamp")

        if rec.get("type") == "snapshot":
            snap = rec["data"]
            snap_uid = snap["lastUpdateId"]

            # not a run start: compare this snapshot with the current position
            if last_update_id is not None:
                if snap_uid > last_update_id:
                    invisible_spans.append({
                        "kind": "invisible_span",
                        "start_ts": last_ts,
                        "end_ts": rec_ts,
                        "start_uid": last_update_id,
                        "end_uid": snap_uid,
                        "updates_lost": snap_uid - last_update_id,
                    })
                    print("[warning] invisible span at snapshot: " +
                          str(snap_uid - last_update_id) +
                          " updates happened between last_update_id=" +
                          str(last_update_id) + " and this snapshot's lastUpdateId=" +
                          str(snap_uid) + ", with no diff ever covering them")
                elif snap_uid < last_update_id:
                    raise RuntimeError(
                        "Backward anchor jump: this snapshot's lastUpdateId (" +
                        str(snap_uid) + ") is BEHIND the current position (" +
                        str(last_update_id) + "). Applying it would move the "
                        "book backwards in time, and the existing gap check would "
                        "then see a clean resumption and report nothing wrong. "
                        "Refusing to continue -- this usually means the wrong "
                        "--anchor-in file was used, or files were processed out "
                        "of order."
                    )
                # same position: no gap

            bids = {p: q for p, q in snap["bids"]}
            asks = {p: q for p, q in snap["asks"]}
            last_update_id = snap_uid
            last_ts = rec_ts
            continue

        data = rec.get("data")
        if not data or data.get("e") != "depthUpdate" or last_update_id is None:
            continue

        if data["u"] <= last_update_id:
            continue

        if data["U"] != last_update_id + 1:
            resumption_gaps.append({
                "kind": "resumption",
                "start_ts": last_ts,
                "end_ts": rec_ts,
                "start_uid": last_update_id,
                "end_uid": data["U"],
                "updates_lost": data["U"] - last_update_id - 1,
            })
            print("[warning] sequence gap at update " + str(data["u"]) +
                  " (expected this event's U to be " + str(last_update_id + 1) +
                  ", got U=" + str(data["U"]) + ")")

        apply_diff(bids, data["b"])
        apply_diff(asks, data["a"])
        last_update_id = data["u"]
        last_ts = rec_ts

    if last_update_id is None:
        raise RuntimeError(
            "No snapshot record was found in this file, and no --anchor-in "
            "was supplied. There is no anchor point to reconstruct from, so "
            "the book cannot be built. Either this file needs to contain a "
            "snapshot, or pass --anchor-in pointing at the --anchor-out file "
            "produced by reconstructing the file immediately before this one."
        )

    return {
        "bids": bids,
        "asks": asks,
        "last_update_id": last_update_id,
        "last_ts": last_ts,
        "resumption_gaps": resumption_gaps,
        "invisible_spans": invisible_spans,
        "seeded_from_anchor": seeded_from_anchor,
        "source_files": source_files,
        "carried_resumption_gaps": carried_resumption_gaps,
        "carried_resumption_updates_lost": carried_resumption_updates_lost,
        "carried_invisible_spans": carried_invisible_spans,
        "carried_invisible_updates_lost": carried_invisible_updates_lost,
    }


def top_levels(book_side, n, reverse):
    prices = sorted(book_side.keys(), key=float, reverse=reverse)
    return [(p, book_side[p]) for p in prices[:n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild the order book from a collected .jsonl file."
    )
    parser.add_argument("path", help="path to the .jsonl file to reconstruct")
    parser.add_argument(
        "--anchor-in",
        help="path to a JSON anchor file (from a prior run's --anchor-out) to "
             "seed the starting book state -- needed for files with no snapshot "
             "of their own",
    )
    parser.add_argument(
        "--anchor-out",
        help="path to write the final book state to, as a JSON anchor file, "
             "for chaining into the next file's --anchor-in",
    )
    args = parser.parse_args()

    path = args.path
    print("Loading " + path + " ...")
    records = load_records(path)
    print("Loaded " + str(len(records)) + " records. Reconstructing...")

    anchor = None
    if args.anchor_in:
        print("Seeding from anchor: " + args.anchor_in)
        anchor = load_anchor(args.anchor_in)

    result = reconstruct(records, anchor=anchor)
    bids = result["bids"]
    asks = result["asks"]
    resumption_gaps = result["resumption_gaps"]
    invisible_spans = result["invisible_spans"]
    total_resumption_lost = sum(g["updates_lost"] for g in resumption_gaps)
    total_invisible_lost = sum(g["updates_lost"] for g in invisible_spans)

    print("")
    print("Reconstruction complete.")
    print("  Resumption gaps this file: " + str(len(resumption_gaps)) +
          " (" + str(total_resumption_lost) + " updates lost)")
    print("  Invisible spans this file: " + str(len(invisible_spans)) +
          " (" + str(total_invisible_lost) + " updates lost)")
    if not resumption_gaps and not invisible_spans:
        print("  (No gaps of either kind detected in this file.)")

    gaps_out_path = path + ".gaps.json"
    with open(gaps_out_path, "w") as f:
        json.dump(resumption_gaps + invisible_spans, f, indent=2)
    print("  Gap intervals (start/end timestamp + update-ID) written to: " + gaps_out_path)

    if args.anchor_out:
        cumulative_resumption_gaps = result["carried_resumption_gaps"] + len(resumption_gaps)
        cumulative_resumption_updates_lost = result["carried_resumption_updates_lost"] + total_resumption_lost
        cumulative_invisible_spans = result["carried_invisible_spans"] + len(invisible_spans)
        cumulative_invisible_updates_lost = result["carried_invisible_updates_lost"] + total_invisible_lost
        chain_has_gaps = (cumulative_resumption_gaps + cumulative_invisible_spans) > 0

        anchor_out = {
            "source_files": result["source_files"] + [path],
            "final_timestamp": result["last_ts"],
            "last_update_id": result["last_update_id"],
            "bids": bids,
            "asks": asks,
            "cumulative_resumption_gaps": cumulative_resumption_gaps,
            "cumulative_resumption_updates_lost": cumulative_resumption_updates_lost,
            "cumulative_invisible_spans": cumulative_invisible_spans,
            "cumulative_invisible_updates_lost": cumulative_invisible_updates_lost,
            "chain_has_gaps": chain_has_gaps,
        }
        save_anchor(args.anchor_out, anchor_out)
        print("  Anchor state written to: " + args.anchor_out)
        if chain_has_gaps:
            print("  Note: this anchor's chain (" + ", ".join(anchor_out["source_files"]) +
                  ") has crossed " + str(cumulative_resumption_gaps) + " resumption gap(s) and " +
                  str(cumulative_invisible_spans) + " invisible span(s) so far (" +
                  str(cumulative_resumption_updates_lost + cumulative_invisible_updates_lost) +
                  " updates lost total). Exclude those windows downstream "
                  "(intervals in the .gaps.json files).")

    print("")
    print("Reconstructed top 5 bids (highest price first):")
    for p, q in top_levels(bids, 5, True):
        print("  " + p + "  size=" + q)

    print("")
    print("Reconstructed top 5 asks (lowest price first):")
    for p, q in top_levels(asks, 5, False):
        print("  " + p + "  size=" + q)

    print("")
    print("Fetching a fresh live snapshot from Binance.US for comparison...")
    live = fetch_live_snapshot()

    print("")
    print("Actual live top 5 bids:")
    for p, q in live["bids"][:5]:
        print("  " + p + "  size=" + q)

    print("")
    print("actual live top 5 asks:")
    for p, q in live["asks"][:5]:
        print("  " + p + "  size=" + q)

    print("")
    print("Compare reconstructed vs live top of book above.")
    print("Small differences expected: the market moves")
    print("between the last recorded diff and")
    print("the live fetch.")
