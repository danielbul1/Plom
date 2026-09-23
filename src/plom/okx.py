"""OKX's public WebSocket for the USDT perpetual: best bid/ask or the full book, and trades (no auth).

By default we stream tick-by-tick best bid/ask. With depth, the `books` channel instead sends a
400-level snapshot and then deltas every 100ms, each naming the previous one's seqId; on a gap we
reconnect for a new snapshot.

Sizes are in contracts (0.01 BTC each for BTC-USDT-SWAP), not coins. The documented port 8443 is
refused from some networks, so we use 443.
"""

from collections.abc import AsyncIterator

from plom import feed
from plom.market import BOOK_DEPTH, Book, LocalBook, Trade

NAME = "okx"
WS_URL = "wss://ws.okx.com/ws/v5/public"
BBO_EVERY_MS = 50


def _channel(message: dict) -> str | None:
    return message.get("arg", {}).get("channel")


def _sequence(message: dict) -> tuple[bool, int, int] | None:
    if _channel(message) != "books":
        return None
    data = message["data"][0]
    return message["action"] == "snapshot", data["prevSeqId"], data["seqId"]


def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
    inst = f"{coin.upper()}-USDT-SWAP"
    book_channel = "books" if depth else "bbo-tbt"
    subscribe = {"op": "subscribe", "args": [{"channel": c, "instId": inst} for c in (book_channel, "trades")]}
    return feed.merged([feed.Source(
        WS_URL, [subscribe],
        throttle_ms=0 if depth else BBO_EVERY_MS,
        is_book=lambda m: _channel(m) == "bbo-tbt",
        keep=lambda m: "data" in m,
        sync=feed.chained(_sequence) if depth else None,
    )])


class Parser:
    def __init__(self, depth: int = BOOK_DEPTH) -> None:
        self._book = LocalBook(depth)
        self._seq: int | None = None

    def events(self, message: dict) -> list[Book | Trade]:
        match _channel(message):
            case "bbo-tbt":
                return [
                    Book(int(d["ts"]), [_level(d["bids"][0])], [_level(d["asks"][0])])
                    for d in message["data"] if d["bids"] and d["asks"]
                ]
            case "books":
                data = message["data"][0]
                if message["action"] == "snapshot":
                    self._book.reset([_level(r) for r in data["bids"]], [_level(r) for r in data["asks"]])
                elif self._seq is None or data["prevSeqId"] != self._seq:
                    self._seq = None  # Out of sync until the next snapshot.
                    return []
                else:
                    self._book.apply([_level(r) for r in data["bids"]], [_level(r) for r in data["asks"]])
                self._seq = data["seqId"]
                return [self._book.book(int(data["ts"]))]
            case "trades":
                return [Trade(int(d["ts"]), d["side"], float(d["px"]), float(d["sz"])) for d in message["data"]]
            case _:
                return []


def _level(row: list[str]) -> tuple[float, float]:
    return float(row[0]), float(row[1])
