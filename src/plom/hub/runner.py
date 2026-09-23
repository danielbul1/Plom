"""Stream every venue for every symbol into its SymbolState, forever, reconnecting with backoff."""

import asyncio
import logging
import time
from dataclasses import replace

from plom.hub import candles, contracts
from plom.hub.state import SymbolState
from plom.market import Book, Trade
from plom.venues import VENUES

log = logging.getLogger("plom.hub")
MAX_BACKOFF_S = 60.0
BOOK_LEVELS = 200
"""Levels per side kept from venues whose books we rebuild ourselves."""
DEPTH_STREAMS = {"okx", "blofin", "htx_spot", "htx_perps", "aster"}
"""Venues that stream only the top of book unless asked for depth."""
DEEP_PARSERS = {"okx", "blofin", "coinbase", "orderly"}
CANDLE_FLUSH_S = 5.0
REPAIR_EVERY_S = 300.0
DAY_MS = 86_400_000
INITIAL_BACKFILL = {"1m": DAY_MS, "15m": 14 * DAY_MS, "1h": 90 * DAY_MS, "4h": 365 * DAY_MS, "1d": 5 * 365 * DAY_MS}


class Hub:
    def __init__(self, coins: list[str], venues: list[str], store: candles.Store | None = None) -> None:
        self.venues = tuple(venues)
        self.store = store
        started_ms = time.time() * 1000
        self.states = {symbol_of(coin): SymbolState(symbol_of(coin), self.venues, started_ms) for coin in coins}
        self.status: dict[tuple[str, str], str] = {}
        """(symbol, venue) -> "connecting", "live", "not listed" or the last error."""
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        for symbol in self.states:
            for venue in self.venues:
                self._tasks.append(asyncio.create_task(self._run(symbol, venue)))
        self._tasks.append(asyncio.create_task(self._sample()))
        if self.store is not None:
            self._tasks.append(asyncio.create_task(self._flush_candles()))
            self._tasks.append(asyncio.create_task(self._initial_backfill()))
            self._tasks.append(asyncio.create_task(self._repair_partials()))

    async def _repair_partials(self) -> None:
        """Candles that began before we started only saw part of their trades: once they close,
        replace them with backfilled ones."""
        while True:
            await asyncio.sleep(REPAIR_EVERY_S)
            now_ms = int(time.time() * 1000)
            for symbol, state in self.states.items():
                for interval, length in candles.INTERVALS.items():
                    opens = self.store.closed_partials(symbol, interval, now_ms)
                    if not opens:
                        continue
                    try:
                        await asyncio.to_thread(
                            candles.backfill, self.store, symbol, interval, min(opens), max(opens) + length - 1,
                            dict(state.composite.basis),
                        )
                    except Exception as error:
                        log.warning("repairing %s %s candles failed: %s", symbol, interval, error)

    async def _flush_candles(self) -> None:
        while True:
            await asyncio.sleep(CANDLE_FLUSH_S)
            for symbol, state in self.states.items():
                state.candles.flush(symbol, self.store)

    async def _initial_backfill(self) -> None:
        """Give charts some history from the start: fill each window's gaps once."""
        await asyncio.sleep(30)  # Let each venue's basis settle first.
        now_ms = int(time.time() * 1000)
        for symbol, state in self.states.items():
            for interval, span_ms in INITIAL_BACKFILL.items():
                try:
                    count = await asyncio.to_thread(
                        candles.backfill, self.store, symbol, interval, now_ms - span_ms, now_ms, dict(state.composite.basis),
                    )
                    log.info("backfilled %d %s candles for %s", count, interval, symbol)
                except Exception as error:
                    log.warning("backfill %s %s failed: %s", symbol, interval, error)

    async def _sample(self) -> None:
        """Take a heatmap column for every symbol at the top of each second."""
        while True:
            await asyncio.sleep(1 - time.time() % 1)
            now_ms = time.time() * 1000
            for state in self.states.values():
                state.sample_heatmap(now_ms)

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
                module = VENUES[venue]
                parser = module.Parser(depth=BOOK_LEVELS) if venue in DEEP_PARSERS else module.Parser()
                stream = module.messages(coin, depth=True) if venue in DEPTH_STREAMS else module.messages(coin)
                async for message in stream:
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
