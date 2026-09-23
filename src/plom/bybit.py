"""Bybit's public WebSocket for the USDT perpetual: the book and trades (no auth needed).

By default we stream the best bid/ask (`orderbook.1`, every push a snapshot). With depth,
`orderbook.50` sends a snapshot and then deltas whose update id `u` rises by one each time; on a
gap we reconnect for a new snapshot. The server drops connections that don't ping it. Sizes are in
coins.
"""

from collections.abc import AsyncIterator

from plom import feed
from plom.market import BOOK_DEPTH, Book, LocalBook, Trade

NAME = "bybit"
WS_URL = "wss://stream.bybit.com/v5/public/linear"
BBO_EVERY_MS = 50


def _sequence(message: dict) -> tuple[bool, int, int] | None:
    if not message.get("topic", "").startswith("orderbook.50."):
        return None
    u = message["data"]["u"]
    return message["type"] == "snapshot", u - 1, u


def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
    symbol = f"{coin.upper()}USDT"
    book = f"orderbook.{50 if depth else 1}.{symbol}"
    return feed.merged([feed.Source(
        WS_URL, [{"op": "subscribe", "args": [book, f"publicTrade.{symbol}"]}],
        throttle_ms=0 if depth else BBO_EVERY_MS,
        is_book=lambda m: m.get("topic", "").startswith("orderbook.1."),
        keep=lambda m: "topic" in m,
        heartbeat=lambda: {"op": "ping"},
        sync=feed.chained(_sequence) if depth else None,
    )])


class Parser:
    def __init__(self, depth: int = BOOK_DEPTH) -> None:
        self._book = LocalBook(depth)
        self._u: int | None = None

    def events(self, message: dict) -> list[Book | Trade]:
        topic, data = message.get("topic", ""), message.get("data")
        time_ms = message.get("cts") or message.get("ts")
        if topic.startswith("orderbook.1."):
            bids, asks = _levels(data["b"]), _levels(data["a"])
            return [Book(time_ms, bids, asks)] if bids and asks else []
        if topic.startswith("orderbook."):
            if message["type"] == "snapshot":
                self._book.reset(_levels(data["b"]), _levels(data["a"]))
            elif self._u is None or data["u"] != self._u + 1:
                self._u = None  # Out of sync until the next snapshot.
                return []
            else:
                self._book.apply(_levels(data["b"]), _levels(data["a"]))
            self._u = data["u"]
            return [self._book.book(time_ms)]
        if topic.startswith("publicTrade."):
            return [Trade(t["T"], "buy" if t["S"] == "Buy" else "sell", float(t["p"]), float(t["v"])) for t in data]
        return []


def _levels(rows: list[list[str]]) -> list[tuple[float, float]]:
    return [(float(p), float(s)) for p, s in rows]
