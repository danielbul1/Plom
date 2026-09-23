"""Coinbase Exchange's public WebSocket: a 50-level book from batched deltas, and trades (no auth).

level2_batch sends a snapshot on subscribe, then l2update deltas batched every ~50ms. The feed has
no sequence numbers to check, but a reconnect always starts with a fresh snapshot. Spot BTC-USD
trades in real dollars, where most other reference venues trade perps against USDT.
"""

from collections.abc import AsyncIterator
from datetime import datetime

from plom import feed
from plom.market import Book, LocalBook, Trade

NAME = "coinbase"
WS_URL = "wss://ws-feed.exchange.coinbase.com"
KEEP = {"snapshot", "l2update", "match"}


def messages(coin: str) -> AsyncIterator[dict]:
    product = f"{coin.upper()}-USD"
    subscribe = {"type": "subscribe", "product_ids": [product], "channels": ["level2_batch", "matches"]}
    return feed.merged([feed.Source(WS_URL, [subscribe], keep=lambda m: m.get("type") in KEEP)])


class Parser:
    def __init__(self) -> None:
        self._book = LocalBook()
        self._synced = False

    def events(self, message: dict) -> list[Book | Trade]:
        match message.get("type"):
            case "snapshot":
                bids = [(float(p), float(s)) for p, s in message["bids"]]
                asks = [(float(p), float(s)) for p, s in message["asks"]]
                self._book.reset(bids, asks)
                self._synced = True
                return []  # No timestamp; the first update carries one.
            case "l2update":
                if not self._synced:
                    return []
                bids = [(float(p), float(s)) for side, p, s in message["changes"] if side == "buy"]
                asks = [(float(p), float(s)) for side, p, s in message["changes"] if side == "sell"]
                self._book.apply(bids, asks)
                return [self._book.book(_ms(message["time"]))]
            case "match":
                # side is the maker's, so the aggressor took the other side.
                side = "sell" if message["side"] == "buy" else "buy"
                return [Trade(_ms(message["time"]), side, float(message["price"]), float(message["size"]))]
            case _:
                return []


def _ms(iso: str) -> int:
    return round(datetime.fromisoformat(iso).timestamp() * 1000)
