"""Binance.US spot collector: depth snapshot + incremental depth diffs + trades, with auto-reconnect."""

import asyncio
import json
import signal
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import websockets

CANDIDATE_SYMBOLS = ["btcusd", "btcusdt"]

OUTPUT_DIR = Path("./market_data")
OUTPUT_DIR.mkdir(exist_ok=True)

_shutdown = False
SYMBOL = None


def _resolve_symbol():
    """First candidate symbol Binance.US accepts."""
    for candidate in CANDIDATE_SYMBOLS:
        test_url = f"https://api.binance.us/api/v3/depth?symbol={candidate.upper()}&limit=5"
        try:
            with urllib.request.urlopen(test_url, timeout=10) as response:
                if response.status == 200:
                    print(f"Trading pair confirmed working: {candidate.upper()}")
                    return candidate
        except Exception as e:
            print(f"  {candidate.upper()} did not work ({e}). Trying next option...")
    raise RuntimeError(
        "None of the candidate symbols worked on Binance.US. Open "
        "https://api.binance.us/api/v3/exchangeInfo in a browser and search "
        "for 'BTC' to find the exact pair name Binance.US currently lists, "
        "then send that to me."
    )


def _current_output_file() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return OUTPUT_DIR / f"{SYMBOL}_{day}.jsonl"


def _write_record(record):
    out_path = _current_output_file()
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _snapshot_url():
    return f"https://api.binance.us/api/v3/depth?symbol={SYMBOL.upper()}&limit=1000"


def _ws_url():
    return f"wss://stream.binance.us:9443/stream?streams={SYMBOL}@depth@100ms/{SYMBOL}@aggTrade"


def _fetch_and_record_snapshot():
    print("Fetching initial order book snapshot (anchor point)...")
    with urllib.request.urlopen(_snapshot_url(), timeout=10) as response:
        snapshot = json.loads(response.read())
    record = {
        "local_timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "snapshot",
        "data": snapshot,
    }
    _write_record(record)
    print(f"Snapshot recorded (lastUpdateId={snapshot['lastUpdateId']}).")


async def collect():
    global _shutdown
    reconnect_delay = 1

    while not _shutdown:
        try:
            _fetch_and_record_snapshot()

            async with websockets.connect(_ws_url(), ping_interval=None) as ws:
                print(f"[connected] {datetime.now(timezone.utc).isoformat()}")
                reconnect_delay = 1

                async for raw_message in ws:
                    local_ts = datetime.now(timezone.utc).isoformat()
                    payload = json.loads(raw_message)
                    record = {
                        "local_timestamp": local_ts,
                        "type": "diff",
                        "stream": payload.get("stream"),
                        "data": payload.get("data"),
                    }
                    _write_record(record)

        except Exception as e:
            print(f"[error] {e} — reconnecting in {reconnect_delay}s")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)


def _handle_shutdown(signum, frame):
    global _shutdown
    print("\n[shutdown requested]")
    _shutdown = True


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    SYMBOL = _resolve_symbol()
    print(f"Writing to: {OUTPUT_DIR.resolve()}")
    asyncio.run(collect())