"""Drive a market maker with one venue's books and trades plus a reference venue's books.

The reference is one venue, or the composite of several (see plom.composite).

The market maker runs on the quoting venue's exchange clock. Reference books arrive on another
venue's clock, so we translate them through our local receive time: a reference book received at
local time r is treated as happening at r minus the typical delay between the quoting venue's
exchange timestamps and our receiving its messages.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from plom import composite, recording
from plom.market import Book, Trade
from plom.mm import MarketMaker
from plom.venues import VENUES

OFFSET_SMOOTHING = 0.05

Event = tuple[bool, float | None, Book | Trade]
"""(is_reference, local receive time in ms, event)."""


@dataclass
class Dispatcher:
    mm: MarketMaker
    offset_ms: float | None = None
    """Typical local receive time minus the quoting venue's exchange time."""

    def feed(self, is_reference: bool, recv_ms: float | None, event: Book | Trade) -> None:
        if not is_reference:
            if recv_ms is not None and isinstance(event, Book):
                sample = recv_ms - event.time_ms
                self.offset_ms = sample if self.offset_ms is None else self.offset_ms + OFFSET_SMOOTHING * (sample - self.offset_ms)
            if isinstance(event, Book):
                self.mm.on_book(event)
            else:
                self.mm.on_trade(event)
        elif isinstance(event, Book) and recv_ms is not None and self.offset_ms is not None:
            self.mm.on_reference(replace(event, time_ms=round(recv_ms - self.offset_ms)))


class Router:
    """Parses recorded or live messages into events for the quoting venue and its reference.

    With the composite reference, books from every composite venue except the quoting one feed a
    Composite, and its books are the reference.
    """

    def __init__(self, venue: str, reference: str | None) -> None:
        self.venue = venue
        self.composite = None
        references: tuple[str, ...] = ()
        if reference == composite.NAME:
            self.composite = composite.Composite(tuple(v for v in composite.VENUES if v != venue))
            references = self.composite.venues
        elif reference:
            references = (reference,)
        self.parsers = {name: VENUES[name].Parser() for name in (venue, *references)}

    def route(self, source: str, recv_ms: float | None, message: dict) -> list[Event]:
        parser = self.parsers.get(source)
        if parser is None:
            return []
        events = parser.events(message)
        if source == self.venue:
            return [(False, recv_ms, event) for event in events]
        if self.composite is None:
            return [(True, recv_ms, event) for event in events]
        routed = []
        for event in events:
            if isinstance(event, Book) and recv_ms is not None:
                book = self.composite.on_book(source, recv_ms, event)
                if book is not None:
                    routed.append((True, recv_ms, book))
        return routed


def replay(path: Path, venue: str, reference: str | None) -> Iterator[Event]:
    router = Router(venue, reference)
    for source, recv_ms, message in recording.read(path):
        yield from router.route(source, recv_ms, message)


async def live(venue: str, reference: str | None, coin: str) -> AsyncIterator[Event]:
    """Merge the venue's and the reference's live streams, stamping each event with its receive time."""
    router = Router(venue, reference)
    queue: asyncio.Queue[tuple[str, float, dict]] = asyncio.Queue()

    async def pump(name: str) -> None:
        async for message in VENUES[name].messages(coin):
            queue.put_nowait((name, time.time() * 1000, message))

    tasks = [asyncio.create_task(pump(name)) for name in router.parsers]
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait([getter, *tasks], return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                for task in done:
                    task.result()
            for event in router.route(*getter.result()):
                yield event
    finally:
        for task in tasks:
            task.cancel()
