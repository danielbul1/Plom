"""HTX (formerly Huobi) public WebSockets for spot and the USDT perpetual: best bid/ask and trades.

Both send gzip-compressed frames and a {"ping": ts} every few seconds that must be answered with
{"pong": ts}. Perp book sizes are in contracts (0.001 BTC each); trades carry the coin quantity.
"""

import gzip
import json
from collections.abc import AsyncIterator

from plom import feed
from plom.market import Book, Trade

BBO_EVERY_MS = 50


def _source(url: str, prefix: str) -> feed.Source:
    return feed.Source(
        url,
        [{"sub": f"{prefix}.bbo", "id": "bbo"}, {"sub": f"{prefix}.trade.detail", "id": "trade"}],
        throttle_ms=BBO_EVERY_MS,
        is_book=lambda m: m.get("ch", "").endswith(".bbo"),
        decode=lambda raw: json.loads(gzip.decompress(raw)),
        pong=lambda m: {"pong": m["ping"]} if "ping" in m else None,
        keep=lambda m: "ch" in m,
    )


def _trades(message: dict, size_key: str) -> list[Trade]:
    return [
        Trade(d["ts"], d["direction"], float(d["price"]), float(d[size_key]))
        for d in message["tick"]["data"]
    ]


class Spot:
    NAME = "htx_spot"
    WS_URL = "wss://api.huobi.pro/ws"

    @staticmethod
    def messages(coin: str) -> AsyncIterator[dict]:
        return feed.merged([_source(Spot.WS_URL, f"market.{coin.lower()}usdt")])

    class Parser:
        def events(self, message: dict) -> list[Book | Trade]:
            tick = message["tick"]
            if message["ch"].endswith(".bbo"):
                return [Book(tick["quoteTime"], [(tick["bid"], tick["bidSize"])], [(tick["ask"], tick["askSize"])])]
            if message["ch"].endswith(".trade.detail"):
                return _trades(message, "amount")
            return []


class Perps:
    NAME = "htx_perps"
    WS_URL = "wss://api.hbdm.com/linear-swap-ws"

    @staticmethod
    def messages(coin: str) -> AsyncIterator[dict]:
        return feed.merged([_source(Perps.WS_URL, f"market.{coin.upper()}-USDT")])

    class Parser:
        def events(self, message: dict) -> list[Book | Trade]:
            tick = message["tick"]
            if message["ch"].endswith(".bbo"):
                if not tick.get("bid") or not tick.get("ask"):
                    return []
                return [Book(tick["ts"], [tuple(tick["bid"])], [tuple(tick["ask"])])]
            if message["ch"].endswith(".trade.detail"):
                return _trades(message, "quantity")
            return []
