"""Lighter's public WebSocket: order book deltas and trades (no auth needed).

The book arrives as one snapshot and then deltas every ~50ms. Each delta's begin_nonce must equal
the previous message's nonce; on a gap we reconnect for a fresh snapshot.
"""

import asyncio
import json
import urllib.request
from collections import deque
from collections.abc import AsyncIterator

import websockets

from plom.market import Book, Level, LocalBook, Trade

NAME = "lighter"
WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"
REST_URL = "https://mainnet.zklighter.elliot.ai/api/v1"
SEEN_TRADES = 10_000


def market_id(coin: str) -> int:
    with urllib.request.urlopen(f"{REST_URL}/orderBooks", timeout=15) as response:
        markets = json.load(response)["order_books"]
    for market in markets:
        if market["symbol"] == coin.upper() and market["market_type"] == "perp":
            return market["market_id"]
    raise ValueError(f"no Lighter perp market for {coin}")


async def messages(coin: str) -> AsyncIterator[dict]:
    """Yield raw book and trade messages, reconnecting on a dropped socket or a nonce gap."""
    market = await asyncio.to_thread(market_id, coin)
    async for ws in websockets.connect(WS_URL):
        try:
            for channel in (f"order_book/{market}", f"trade/{market}"):
                await ws.send(json.dumps({"type": "subscribe", "channel": channel}))
            nonce = None
            async for raw in ws:
                message = json.loads(raw)
                kind = message.get("type", "")
                if kind == "subscribed/order_book":
                    nonce = message["order_book"]["nonce"]
                elif kind == "update/order_book":
                    book = message["order_book"]
                    if book["begin_nonce"] != nonce:
                        await ws.close()
                        break
                    nonce = book["nonce"]
                if kind.endswith(("/order_book", "/trade")):
                    yield message
        except websockets.ConnectionClosed:
            continue


class Parser:
    def __init__(self) -> None:
        self._book = LocalBook()
        self._nonce: int | None = None
        self._seen: set[int] = set()
        self._seen_order: deque[int] = deque()

    def events(self, message: dict) -> list[Book | Trade]:
        match message.get("type"):
            case "subscribed/order_book":
                book = message["order_book"]
                self._book.reset(_levels(book["bids"]), _levels(book["asks"]))
                self._nonce = book["nonce"]
                return [self._book.book(book["last_updated_at"] // 1000)]
            case "update/order_book":
                book = message["order_book"]
                if self._nonce is None or book["begin_nonce"] != self._nonce:
                    self._nonce = None  # Out of sync until the next snapshot.
                    return []
                self._book.apply(_levels(book["bids"]), _levels(book["asks"]))
                self._nonce = book["nonce"]
                return [self._book.book(book["last_updated_at"] // 1000)]
            case "subscribed/trade" | "update/trade":
                trades = message.get("trades", []) + message.get("liquidation_trades", [])
                return [self._trade(t) for t in sorted(trades, key=lambda t: t["trade_id"]) if self._is_new(t)]
            case _:
                return []

    def _is_new(self, trade: dict) -> bool:
        """Every (re)subscribe replays recent trades, so drop ids we've already emitted."""
        trade_id = trade["trade_id"]
        if trade_id in self._seen:
            return False
        self._seen.add(trade_id)
        self._seen_order.append(trade_id)
        if len(self._seen_order) > SEEN_TRADES:
            self._seen.discard(self._seen_order.popleft())
        return True

    @staticmethod
    def _trade(trade: dict) -> Trade:
        # is_maker_ask: the resting order was the ask, so the aggressor bought.
        side = "buy" if trade["is_maker_ask"] else "sell"
        return Trade(trade["timestamp"], side, float(trade["price"]), float(trade["size"]))


def _levels(levels: list[dict]) -> list[Level]:
    return [(float(level["price"]), float(level["size"])) for level in levels]
