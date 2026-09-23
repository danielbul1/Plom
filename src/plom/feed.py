"""Shared plumbing for venues that stream JSON over one or more WebSocket connections."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

import websockets


@dataclass(frozen=True)
class Source:
    url: str
    subscribe: Sequence[dict] = ()
    """Messages sent after each (re)connect."""
    throttle_ms: float = 0
    """Keep only the first book message per interval, for top-of-book feeds that update far faster
    than we need. Only safe for snapshots, never for deltas."""
    is_book: Callable[[dict], bool] = lambda message: True
    decode: Callable[[str | bytes], dict] = json.loads
    pong: Callable[[dict], dict | None] = lambda message: None
    """The reply to a server keep-alive message, which is then not passed on."""
    keep: Callable[[dict], bool] = lambda message: True
    """Whether to pass a message on, e.g. to drop subscription acknowledgements."""
    heartbeat: Callable[[], dict] | None = None
    """A keep-alive message some venues need from us, sent every heartbeat_s."""
    heartbeat_s: float = 20.0
    sync: Callable[[], Callable[[dict], bool]] | None = None
    """Makes a fresh check per connection that returns False when a message shows a gap in a
    delta stream; we then reconnect, which brings a new snapshot."""


async def merged(sources: Sequence[Source]) -> AsyncIterator[dict]:
    """Yield messages from all sources as they arrive, reconnecting each one if it drops."""
    queue: asyncio.Queue[dict] = asyncio.Queue()
    tasks = [asyncio.create_task(_pump(source, queue)) for source in sources]
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait([getter, *tasks], return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                for task in done:
                    task.result()  # A pump only finishes by raising.
            yield getter.result()
    finally:
        for task in tasks:
            task.cancel()


async def _pump(source: Source, queue: asyncio.Queue[dict]) -> None:
    async for ws in websockets.connect(source.url, max_size=None):
        beat = asyncio.create_task(_heartbeat(ws, source)) if source.heartbeat else None
        try:
            for message in source.subscribe:
                await ws.send(json.dumps(message))
            last_ms = 0.0
            in_sync = source.sync() if source.sync else None
            async for raw in ws:
                message = source.decode(raw)
                reply = source.pong(message)
                if reply is not None:
                    await ws.send(json.dumps(reply))
                    continue
                if not source.keep(message):
                    continue
                if in_sync is not None and not in_sync(message):
                    await ws.close()
                    break
                if source.throttle_ms and source.is_book(message):
                    now_ms = time.monotonic() * 1000
                    if now_ms - last_ms < source.throttle_ms:
                        continue
                    last_ms = now_ms
                queue.put_nowait(message)
        except websockets.ConnectionClosed:
            continue
        finally:
            if beat is not None:
                beat.cancel()


async def _heartbeat(ws: websockets.ClientConnection, source: Source) -> None:
    while True:
        await asyncio.sleep(source.heartbeat_s)
        await ws.send(json.dumps(source.heartbeat()))


def chained(sequence: Callable[[dict], tuple[bool, int, int] | None]) -> Callable[[], Callable[[dict], bool]]:
    """A sync check for delta streams where each update names the sequence number of the one before.

    `sequence` returns (is_snapshot, previous, current) for book messages and None for the rest.
    """

    def make() -> Callable[[dict], bool]:
        last: int | None = None

        def in_sync(message: dict) -> bool:
            nonlocal last
            found = sequence(message)
            if found is None:
                return True
            is_snapshot, previous, current = found
            if not is_snapshot and previous != last:
                return False
            last = current
            return True

        return in_sync

    return make
