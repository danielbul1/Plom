"""Drive a market maker with one venue's books and trades plus a reference venue's books.

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

from plom import recording
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


def replay(path: Path, venue: str, reference: str | None) -> Iterator[Event]:
    parsers = {venue: VENUES[venue].Parser()}
    if reference:
        parsers[reference] = VENUES[reference].Parser()
    for recorded_venue, recv_ms, message in recording.read(path):
        parser = parsers.get(recorded_venue)
        if parser is None:
            continue
        for event in parser.events(message):
            yield recorded_venue != venue, recv_ms, event


async def live(venue: str, reference: str | None, coin: str) -> AsyncIterator[Event]:
    """Merge the venue's and the reference's live streams, stamping each event with its receive time."""
    queue: asyncio.Queue[Event] = asyncio.Queue()

    async def pump(name: str, is_reference: bool) -> None:
        parser = VENUES[name].Parser()
        async for message in VENUES[name].messages(coin):
            recv_ms = time.time() * 1000
            for event in parser.events(message):
                queue.put_nowait((is_reference, recv_ms, event))

    tasks = [asyncio.create_task(pump(venue, False))]
    if reference:
        tasks.append(asyncio.create_task(pump(reference, True)))
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait([getter, *tasks], return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                for task in done:
                    task.result()
            yield getter.result()
    finally:
        for task in tasks:
            task.cancel()
