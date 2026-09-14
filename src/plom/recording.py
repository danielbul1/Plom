"""Record several venues' raw market data into one JSONL file, and read it back.

Each line is {"venue", "recv_ms", "msg"}: recv_ms is our local receive time, the only clock shared
across venues. Paths ending in .gz are gzip-compressed. Older Hyperliquid-only recordings, whose
lines are bare messages, still read back as venue "hyperliquid".
"""

import asyncio
import gzip
import json
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO

from plom.venues import VENUES

FLUSH_EVERY_S = 5.0


def open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8", buffering=1 if mode == "a" else -1)


async def record(coin: str, venues: Sequence[str], path: Path, status_every_s: float = 60.0) -> None:
    counts: Counter[str] = Counter()
    with open_text(path, "a") as out:

        async def pump(venue: str) -> None:
            async for message in VENUES[venue].messages(coin):
                line = {"venue": venue, "recv_ms": round(time.time() * 1000, 1), "msg": message}
                out.write(json.dumps(line, separators=(",", ":")) + "\n")
                counts[venue] += 1

        async def report() -> None:
            last_status = time.monotonic()
            while True:
                await asyncio.sleep(FLUSH_EVERY_S)
                out.flush()
                if time.monotonic() - last_status >= status_every_s:
                    last_status = time.monotonic()
                    summary = "  ".join(f"{venue} {counts[venue]:,}" for venue in venues)
                    print(f"{time.strftime('%H:%M:%S')}  messages: {summary}", flush=True)

        await asyncio.gather(report(), *(pump(venue) for venue in venues))


def read(path: Path) -> Iterator[tuple[str, float | None, dict]]:
    """Yield (venue, recv_ms, message), stopping quietly at a line cut off by an unclean stop."""
    with open_text(path, "r") as recording:
        try:
            for line in recording:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return
                if "venue" in entry:
                    yield entry["venue"], entry["recv_ms"], entry["msg"]
                else:
                    yield "hyperliquid", None, entry
        except (EOFError, gzip.BadGzipFile):
            return
