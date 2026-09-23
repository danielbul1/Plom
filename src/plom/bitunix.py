"""Bitunix's public WebSocket for the USDT perpetual: book snapshots and trades (no auth needed).

Books arrive as snapshots about three times a second: the best bid/ask by default, 15 levels with
depth. (Its full-book channel sends thousands of levels each time, far more than we need.) The
server drops a connection that hasn't pinged it for about a minute. Sizes are in coins.
"""

import time
from collections.abc import AsyncIterator

from plom import feed
from plom.market import Book, Trade

NAME = "bitunix"
WS_URL = "wss://fapi.bitunix.com/public/"


def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
    symbol = f"{coin.upper()}USDT"
    book = "depth_book15" if depth else "depth_book1"
    subscribe = {"op": "subscribe", "args": [{"symbol": symbol, "ch": ch} for ch in (book, "trade")]}
    return feed.merged([feed.Source(
        WS_URL, [subscribe],
        keep=lambda m: "ch" in m,
        heartbeat=lambda: {"op": "ping", "ping": int(time.time())},
    )])


class Parser:
    def events(self, message: dict) -> list[Book | Trade]:
        data = message["data"]
        match message["ch"]:
            case "depth_book1" | "depth_book5" | "depth_book15":
                bids = [(float(p), float(s)) for p, s in data["b"]]
                asks = [(float(p), float(s)) for p, s in data["a"]]
                return [Book(message["ts"], bids, asks)] if bids and asks else []
            case "trade":
                # Each trade's own time is only to the second, so use the message's.
                return [Trade(message["ts"], t["s"], float(t["p"]), float(t["v"])) for t in data]
            case _:
                return []
