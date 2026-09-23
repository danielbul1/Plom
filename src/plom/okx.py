"""OKX's public WebSocket for the USDT perpetual: tick-by-tick best bid/ask, and trades (no auth).

Sizes are in contracts (0.01 BTC each for BTC-USDT-SWAP), not coins. The documented port 8443 is
refused from some networks, so we use 443.
"""

from collections.abc import AsyncIterator

from plom import feed
from plom.market import Book, Trade

NAME = "okx"
WS_URL = "wss://ws.okx.com/ws/v5/public"
BBO_EVERY_MS = 50


def messages(coin: str) -> AsyncIterator[dict]:
    inst = f"{coin.upper()}-USDT-SWAP"
    subscribe = {"op": "subscribe", "args": [{"channel": c, "instId": inst} for c in ("bbo-tbt", "trades")]}
    return feed.merged([feed.Source(
        WS_URL, [subscribe], throttle_ms=BBO_EVERY_MS,
        is_book=lambda m: m.get("arg", {}).get("channel") == "bbo-tbt",
        keep=lambda m: "data" in m,
    )])


class Parser:
    def events(self, message: dict) -> list[Book | Trade]:
        match message.get("arg", {}).get("channel"):
            case "bbo-tbt":
                return [
                    Book(int(d["ts"]), [_level(d["bids"][0])], [_level(d["asks"][0])])
                    for d in message["data"] if d["bids"] and d["asks"]
                ]
            case "trades":
                return [Trade(int(d["ts"]), d["side"], float(d["px"]), float(d["sz"])) for d in message["data"]]
            case _:
                return []


def _level(row: list[str]) -> tuple[float, float]:
    return float(row[0]), float(row[1])
