"""BloFin's public WebSocket for the USDT perpetual: five-level book snapshots and trades (no auth).

Sizes are in contracts (0.001 BTC each for BTC-USDT).
"""

from collections.abc import AsyncIterator

from plom import feed
from plom.market import Book, Trade

NAME = "blofin"
WS_URL = "wss://openapi.blofin.com/ws/public"
BOOK_EVERY_MS = 50


def messages(coin: str) -> AsyncIterator[dict]:
    inst = f"{coin.upper()}-USDT"
    subscribe = {"op": "subscribe", "args": [{"channel": c, "instId": inst} for c in ("books5", "trades")]}
    return feed.merged([feed.Source(
        WS_URL, [subscribe], throttle_ms=BOOK_EVERY_MS,
        is_book=lambda m: m.get("arg", {}).get("channel") == "books5",
        keep=lambda m: "data" in m,
    )])


class Parser:
    def events(self, message: dict) -> list[Book | Trade]:
        data = message["data"]
        match message.get("arg", {}).get("channel"):
            case "books5":
                bids = [(float(p), float(s)) for p, s in data["bids"]]
                asks = [(float(p), float(s)) for p, s in data["asks"]]
                return [Book(int(data["ts"]), bids, asks)] if bids and asks else []
            case "trades":
                return [Trade(int(d["ts"]), d["side"], float(d["price"]), float(d["size"])) for d in data]
            case _:
                return []
