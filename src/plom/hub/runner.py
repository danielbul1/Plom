"""Stream every venue for every symbol into its SymbolState, forever, reconnecting with backoff."""

import asyncio
import logging
import time
from dataclasses import replace

from plom.hub import contracts
from plom.hub.state import SymbolState
from plom.market import Book, Trade
from plom.venues import VENUES

log = logging.getLogger("plom.hub")
MAX_BACKOFF_S = 60.0


class Hub:
    def __init__(self, coins: list[str], venues: list[str]) -> None:
        self.venues = tuple(venues)
        self.states = {symbol_of(coin): SymbolState(symbol_of(coin), self.venues) for coin in coins}
        self.status: dict[tuple[str, str], str] = {}
        """(symbol, venue) -> "connecting", "live", "not listed" or the last error."""
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        for symbol in self.states:
            for venue in self.venues:
                self._tasks.append(asyncio.create_task(self._run(symbol, venue)))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, symbol: str, venue: str) -> None:
        coin, state, key = coin_of(symbol), self.states[symbol], (symbol, venue)
        backoff = 1.0
        while True:
            self.status[key] = "connecting"
            try:
                size = await asyncio.to_thread(contracts.contract_size, venue, coin)
                if size is None:
                    self.status[key] = "not listed"
                    return
                book_scale = size if venue in contracts.BOOK_IN_CONTRACTS else 1.0
                trade_scale = size if venue in contracts.TRADES_IN_CONTRACTS else 1.0
                parser = VENUES[venue].Parser()
                async for message in VENUES[venue].messages(coin):
                    recv_ms = time.time() * 1000
                    for event in parser.events(message):
                        if isinstance(event, Book):
                            state.on_book(venue, recv_ms, scale_book(event, book_scale))
                        elif isinstance(event, Trade):
                            state.on_trade(venue, recv_ms, replace(event, size=event.size * trade_scale))
                    self.status[key] = "live"
                    backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as error:  # A venue failing must never take the others down.
                self.status[key] = f"{type(error).__name__}: {error}"[:200]
                log.warning("%s %s: %s; retrying in %.0fs", symbol, venue, self.status[key], backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)


def scale_book(book: Book, scale: float) -> Book:
    if scale == 1.0:
        return book
    return replace(book, bids=[(p, s * scale) for p, s in book.bids], asks=[(p, s * scale) for p, s in book.asks])


def symbol_of(coin: str) -> str:
    return f"{coin.upper()}-USD"


def coin_of(symbol: str) -> str:
    return symbol.split("-")[0].upper()
