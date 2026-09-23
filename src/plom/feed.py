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
        try:
            for message in source.subscribe:
                await ws.send(json.dumps(message))
            last_ms = 0.0
            async for raw in ws:
                message = source.decode(raw)
                reply = source.pong(message)
                if reply is not None:
                    await ws.send(json.dumps(reply))
                    continue
                if not source.keep(message):
                    continue
                if source.throttle_ms and source.is_book(message):
                    now_ms = time.monotonic() * 1000
                    if now_ms - last_ms < source.throttle_ms:
                        continue
                    last_ms = now_ms
                queue.put_nowait(message)
        except websockets.ConnectionClosed:
            continue
