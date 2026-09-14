"""Venue-neutral market data: books, trades, and a local book maintained from deltas."""

from dataclasses import dataclass
from typing import Literal, Protocol

Level = tuple[float, float]
"""A book level as (price, size)."""

BOOK_DEPTH = 50
"""Levels per side kept in emitted books; venues with deeper books are truncated."""


@dataclass(frozen=True)
class Book:
    time_ms: int
    bids: list[Level]
    """Best (highest) first."""
    asks: list[Level]
    """Best (lowest) first."""


@dataclass(frozen=True)
class Trade:
    time_ms: int
    side: Literal["buy", "sell"]
    """The aggressor's side: a "sell" trade hit resting bids."""
    price: float
    size: float


class Parser(Protocol):
    """Turns a venue's raw messages into books and trades. May keep state across messages."""

    def events(self, message: dict) -> list[Book | Trade]: ...


class LocalBook:
    """A full order book rebuilt from a snapshot plus deltas, where size 0 deletes a level."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def reset(self, bids: list[Level], asks: list[Level]) -> None:
        self.bids, self.asks = {}, {}
        self.apply(bids, asks)

    def apply(self, bids: list[Level], asks: list[Level]) -> None:
        for levels, side in ((bids, self.bids), (asks, self.asks)):
            for price, size in levels:
                if size > 0:
                    side[price] = size
                else:
                    side.pop(price, None)

    def book(self, time_ms: int, depth: int = BOOK_DEPTH) -> Book:
        bids = sorted(self.bids.items(), reverse=True)[:depth]
        asks = sorted(self.asks.items())[:depth]
        return Book(time_ms, bids, asks)
