"""Everything the hub knows about one symbol: each venue's latest book, the composite price, a merged
order book and the recent trades of every venue.

Times are local receive times in ms. Sizes are in coins, already scaled from contracts.
"""

import math
from collections import deque
from itertools import islice, takewhile
from dataclasses import dataclass

import numpy as np

from plom import composite
from plom.hub.candles import LiveCandles
from plom.market import Book, Trade

FRESH_MS = 5000
"""A venue whose last book is older than this is left out of prices and the merged book."""
TAPE_SIZE = 5000
HEATMAP_SECONDS = 1800
"""How much heatmap history to keep, at one column a second."""
HEATMAP_LEVELS = 150
"""Buckets each side of the composite price in each heatmap column."""


@dataclass(frozen=True)
class Column:
    """One second of the basis-adjusted merged book on a price grid centred on the composite price.

    Bids and asks share the grid: bucket i is the price low + i * bucket. Venues lead and lag each
    other by about a basis point, more than any one venue's spread, so their bids and asks overlap
    a little near the price even after adjusting.
    """

    ts_ms: int
    bucket: float
    price: float
    low: float
    bids: np.ndarray
    asks: np.ndarray

    def to_json(self) -> dict:
        size = lambda values: [float(f"{x:.6g}") for x in values]
        return {
            "ts_ms": self.ts_ms, "bucket": self.bucket, "price": self.price, "low": self.low,
            "bids": size(self.bids), "asks": size(self.asks),
        }


@dataclass(frozen=True)
class Print:
    seq: int
    recv_ms: float
    venue: str
    trade: Trade

    def to_json(self) -> dict:
        t = self.trade
        return {
            "seq": self.seq, "ts_ms": t.time_ms, "recv_ms": round(self.recv_ms), "venue": self.venue,
            "side": t.side, "price": t.price, "size": t.size, "notional": t.price * t.size,
        }


class SymbolState:
    def __init__(self, symbol: str, venues: tuple[str, ...], started_ms: float = 0.0) -> None:
        self.symbol = symbol
        self.candles = LiveCandles(started_ms)
        self.composite = composite.Composite(venues)  # Every venue, so each gets a basis for the adjusted book.
        self.books: dict[str, tuple[float, Book]] = {}
        self.tape: deque[Print] = deque(maxlen=TAPE_SIZE)
        self.seq = 0
        self.price: float | None = None
        self.price_ms: float | None = None
        self.heatmap: deque[Column] = deque(maxlen=HEATMAP_SECONDS)

    def on_book(self, venue: str, recv_ms: float, book: Book) -> None:
        if not book.bids or not book.asks:
            return
        self.books[venue] = (recv_ms, book)
        merged = self.composite.on_book(venue, recv_ms, book)
        if merged is not None:
            self.price, self.price_ms = merged.bids[0][0], recv_ms

    def on_trade(self, venue: str, recv_ms: float, trade: Trade) -> None:
        self.seq += 1
        self.tape.append(Print(self.seq, recv_ms, venue, trade))
        adjusted = trade.price * math.exp(-self.composite.basis.get(venue, 0.0))
        self.candles.on_trade(trade.time_ms, adjusted, trade.size, trade.side)

    def fresh_books(self, now_ms: float) -> dict[str, tuple[float, Book]]:
        return {v: (t, b) for v, (t, b) in self.books.items() if now_ms - t <= FRESH_MS}

    def tick(self, now_ms: float) -> dict:
        fresh = self.fresh_books(now_ms)
        since = now_ms - 1000
        return {
            "symbol": self.symbol,
            "price": self.price,
            "price_source": "composite",
            "ts_ms": round(self.price_ms) if self.price_ms else None,
            "prices": {v: (b.bids[0][0] + b.asks[0][0]) / 2 for v, (_, b) in sorted(fresh.items())},
            "spreads_bps": {v: (b.asks[0][0] / b.bids[0][0] - 1) * 10_000 for v, (_, b) in sorted(fresh.items())},
            "ages_ms": {v: round(now_ms - t) for v, (t, _) in sorted(fresh.items())},
            "basis_bps": {v: b * 10_000 for v, b in sorted(self.composite.basis.items())},
            "volume_1s": sum(p.trade.size for p in takewhile(lambda p: p.recv_ms >= since, reversed(self.tape))),
        }

    def dom(self, now_ms: float, bucket: float | None = None, depth: int = 50, adjusted: bool = False) -> dict:
        """Every fresh venue's levels summed into price buckets: bids rounded down, asks up.

        Venues trade at different levels, so the raw merge can look crossed. `adjusted` first moves
        each venue's prices onto the composite's level by its learned basis.
        """
        fresh = self.fresh_books(now_ms)
        if bucket is None:
            bucket = default_bucket(self.price or next((b.bids[0][0] for _, b in fresh.values()), 1.0))
        sides = {}
        for side in ("bids", "asks"):
            levels: dict[float, dict[str, float]] = {}
            for venue, (_, book) in fresh.items():
                shift = math.exp(-self.composite.basis.get(venue, 0.0)) if adjusted else 1.0
                for price, size in getattr(book, side):
                    price *= shift
                    steps = math.floor(price / bucket + 1e-9) if side == "bids" else math.ceil(price / bucket - 1e-9)
                    key = round(steps * bucket, 10)
                    by_venue = levels.setdefault(key, {})
                    by_venue[venue] = by_venue.get(venue, 0.0) + size
            ordered = sorted(levels.items(), reverse=side == "bids")[:depth]
            sides[side] = [{"price": p, "size": sum(v.values()), "venues": v} for p, v in ordered]
        return {"symbol": self.symbol, "ts_ms": round(now_ms), "bucket": bucket, "adjusted": adjusted, "venues": sorted(fresh), **sides}

    def sample_heatmap(self, now_ms: float) -> Column | None:
        if self.price is None:
            return None
        dom = self.dom(now_ms, depth=HEATMAP_LEVELS, adjusted=True)
        bucket = dom["bucket"]
        low = round((math.floor(self.price / bucket) - HEATMAP_LEVELS) * bucket, 10)
        sides = []
        for side in ("bids", "asks"):
            sizes = np.zeros(2 * HEATMAP_LEVELS, dtype=np.float32)
            for level in dom[side]:
                index = round((level["price"] - low) / bucket)
                if 0 <= index < sizes.size:
                    sizes[index] += level["size"]
            sides.append(sizes)
        column = Column(round(now_ms), bucket, self.price, low, *sides)
        self.heatmap.append(column)
        return column

    def heatmap_since(self, now_ms: float, seconds: float, step: int = 1) -> list[Column]:
        """Columns from the last `seconds`, oldest first, keeping every `step`-th counting back from the newest."""
        since = now_ms - seconds * 1000
        recent = list(takewhile(lambda c: c.ts_ms >= since, reversed(self.heatmap)))
        return recent[::step][::-1]

    def prints_after(self, seq: int, limit: int = 500) -> list[Print]:
        """Trades with a sequence number above seq, oldest first, at most the newest `limit`."""
        newer = list(islice(takewhile(lambda p: p.seq > seq, reversed(self.tape)), limit))
        return newer[::-1]


def default_bucket(price: float) -> float:
    """About one basis point, rounded to a power of ten: 1 for BTC near 85,000, 0.1 for ETH near 3,000."""
    return 10 ** math.floor(math.log10(price * 1e-4)) if price > 0 else 1.0
