"""HTX (formerly Huobi) public WebSockets for spot and the USDT perpetual: the book and trades.

Both send gzip-compressed frames and a {"ping": ts} every few seconds that must be answered with
{"pong": ts}. Perp book sizes are in contracts (0.001 BTC each); trades carry the coin quantity.

By default we stream best bid/ask. With depth we also take 150-level snapshots (every 100ms on
perps, only every second on spot), and lay the fresher best bid/ask over the last snapshot.
"""

import gzip
import json
from collections.abc import AsyncIterator

from plom import feed
from plom.market import Book, Level, Trade

BBO_EVERY_MS = 50


def _source(url: str, prefix: str, depth: bool = False) -> feed.Source:
    channels = ["bbo", "trade.detail", *(["depth.step0"] if depth else [])]
    return feed.Source(
        url,
        [{"sub": f"{prefix}.{channel}", "id": channel} for channel in channels],
        throttle_ms=BBO_EVERY_MS,
        is_book=lambda m: m.get("ch", "").endswith(".bbo"),
        decode=lambda raw: json.loads(gzip.decompress(raw)),
        pong=lambda m: {"pong": m["ping"]} if "ping" in m else None,
        keep=lambda m: "ch" in m,
    )


class _Parser:
    """Turns bbo, depth and trade messages into books and trades; subclasses read each venue's fields."""

    size_key = "amount"

    def __init__(self) -> None:
        self._depth: Book | None = None
        self._bbo: Book | None = None

    def events(self, message: dict) -> list[Book | Trade]:
        tick, channel = message["tick"], message["ch"]
        if channel.endswith(".depth.step0"):
            bids, asks = [tuple(level) for level in tick["bids"]], [tuple(level) for level in tick["asks"]]
            if not bids or not asks:
                return []
            self._depth = Book(tick["ts"], bids, asks)
            return [self._combined()]
        if channel.endswith(".bbo"):
            bbo = self._bbo_book(tick)
            if bbo is None:
                return []
            self._bbo = bbo
            return [self._combined()]
        if channel.endswith(".trade.detail"):
            return [
                Trade(d["ts"], d["direction"], float(d["price"]), float(d[self.size_key]))
                for d in tick["data"]
            ]
        return []

    def _bbo_book(self, tick: dict) -> Book | None:
        raise NotImplementedError

    def _combined(self) -> Book:
        bbo, depth = self._bbo, self._depth
        if depth is None:
            return bbo
        if bbo is None or depth.time_ms >= bbo.time_ms:
            return depth
        return _overlay(bbo, depth)


def _overlay(bbo: Book, depth: Book) -> Book:
    """The fresher best bid/ask on top, then the snapshot's levels strictly behind it."""
    bid, ask = bbo.bids[0], bbo.asks[0]
    bids: list[Level] = [bid, *(level for level in depth.bids if level[0] < bid[0])]
    asks: list[Level] = [ask, *(level for level in depth.asks if level[0] > ask[0])]
    return Book(bbo.time_ms, bids, asks)


class Spot:
    NAME = "htx_spot"
    WS_URL = "wss://api.huobi.pro/ws"

    @staticmethod
    def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
        return feed.merged([_source(Spot.WS_URL, f"market.{coin.lower()}usdt", depth)])

    class Parser(_Parser):
        def _bbo_book(self, tick: dict) -> Book | None:
            return Book(tick["quoteTime"], [(tick["bid"], tick["bidSize"])], [(tick["ask"], tick["askSize"])])


class Perps:
    NAME = "htx_perps"
    WS_URL = "wss://api.hbdm.com/linear-swap-ws"

    @staticmethod
    def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
        return feed.merged([_source(Perps.WS_URL, f"market.{coin.upper()}-USDT", depth)])

    class Parser(_Parser):
        size_key = "quantity"

        def _bbo_book(self, tick: dict) -> Book | None:
            if not tick.get("bid") or not tick.get("ask"):
                return None
            return Book(tick["ts"], [tuple(tick["bid"])], [tuple(tick["ask"])])
