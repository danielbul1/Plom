"""BloFin's public WebSocket for the USDT perpetual: the order book and trades (no auth).

By default we stream five-level snapshots. With depth, the `books` channel instead sends a
400-level snapshot and then deltas, each naming the previous one's seqId; on a gap we reconnect for
a new snapshot. Sizes are in contracts (0.001 BTC each for BTC-USDT).
"""

from collections.abc import AsyncIterator

from plom import feed
from plom.market import BOOK_DEPTH, Book, LocalBook, Trade

NAME = "blofin"
WS_URL = "wss://openapi.blofin.com/ws/public"
BOOK_EVERY_MS = 50


def _channel(message: dict) -> str | None:
    return message.get("arg", {}).get("channel")


def _sequence(message: dict) -> tuple[bool, int, int] | None:
    if _channel(message) != "books":
        return None
    data = message["data"]
    return message["action"] == "snapshot", int(data["prevSeqId"]), int(data["seqId"])


def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
    inst = f"{coin.upper()}-USDT"
    book_channel = "books" if depth else "books5"
    subscribe = {"op": "subscribe", "args": [{"channel": c, "instId": inst} for c in (book_channel, "trades")]}
    return feed.merged([feed.Source(
        WS_URL, [subscribe],
        throttle_ms=0 if depth else BOOK_EVERY_MS,
        is_book=lambda m: _channel(m) == "books5",
        keep=lambda m: "data" in m,
        sync=feed.chained(_sequence) if depth else None,
    )])


class Parser:
    def __init__(self, depth: int = BOOK_DEPTH) -> None:
        self._book = LocalBook(depth)
        self._seq: int | None = None

    def events(self, message: dict) -> list[Book | Trade]:
        data = message["data"]
        match _channel(message):
            case "books5":
                bids = [(float(p), float(s)) for p, s in data["bids"]]
                asks = [(float(p), float(s)) for p, s in data["asks"]]
                return [Book(int(data["ts"]), bids, asks)] if bids and asks else []
            case "books":
                bids = [(float(p), float(s)) for p, s in data["bids"]]
                asks = [(float(p), float(s)) for p, s in data["asks"]]
                if message["action"] == "snapshot":
                    self._book.reset(bids, asks)
                elif self._seq is None or int(data["prevSeqId"]) != self._seq:
                    self._seq = None  # Out of sync until the next snapshot.
                    return []
                else:
                    self._book.apply(bids, asks)
                self._seq = int(data["seqId"])
                return [self._book.book(int(data["ts"]))]
            case "trades":
                return [Trade(int(d["ts"]), d["side"], float(d["price"]), float(d["size"])) for d in data]
            case _:
                return []
