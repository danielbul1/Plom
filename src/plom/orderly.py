"""Orderly Network's public WebSocket: order book deltas and trades (no auth needed).

Deltas arrive every ~200ms; each one's data.prevTs must equal the previous delta's ts. Full
snapshots arrive every second, but we only pass one on when we need to (re)sync, to keep
recordings small.
"""

import json
import time
from collections.abc import AsyncIterator

import websockets

from plom.market import Book, Level, LocalBook, Trade

NAME = "orderly"
PUBLIC_WS_KEY = "OqdphuyCtYWxwzhxyLLjOWNdFP7sQt8RPWzmb5xY"
"""Orderly's published key for public streams; the path needs a valid id even without auth."""
WS_URL = f"wss://ws-evm.orderly.org/ws/stream/{PUBLIC_WS_KEY}"


def symbol(coin: str) -> str:
    return f"PERP_{coin.upper()}_USDC"


async def messages(coin: str) -> AsyncIterator[dict]:
    """Yield raw book and trade messages, reconnecting if the socket drops."""
    sym = symbol(coin)
    async for ws in websockets.connect(WS_URL):
        try:
            for topic in ("orderbook", "orderbookupdate", "trade"):
                await ws.send(json.dumps({"id": topic, "topic": f"{sym}@{topic}", "event": "subscribe"}))
            synced_ts = None
            async for raw in ws:
                message = json.loads(raw)
                if message.get("event") == "ping":
                    await ws.send(json.dumps({"event": "pong", "ts": int(time.time() * 1000)}))
                    continue
                match message.get("topic", "").removeprefix(f"{sym}@"):
                    case "orderbook":
                        if synced_ts is not None:
                            continue
                        synced_ts = message["ts"]
                    case "orderbookupdate":
                        if synced_ts is None:
                            continue
                        if message["ts"] <= synced_ts:
                            continue  # Already in the snapshot.
                        if message["data"]["prevTs"] != synced_ts:
                            synced_ts = None  # Gap: wait for the next snapshot.
                        else:
                            synced_ts = message["ts"]
                    case "trade":
                        pass
                    case _:
                        continue
                yield message
        except websockets.ConnectionClosed:
            continue


class Parser:
    def __init__(self) -> None:
        self._book = LocalBook()
        self._ts: int | None = None

    def events(self, message: dict) -> list[Book | Trade]:
        data = message["data"]
        match message["topic"].rsplit("@", 1)[1]:
            case "orderbook":
                self._book.reset(_levels(data["bids"]), _levels(data["asks"]))
                self._ts = message["ts"]
                return [self._book.book(message["ts"])]
            case "orderbookupdate":
                if self._ts is None or message["ts"] <= self._ts:
                    return []
                if data["prevTs"] != self._ts:
                    self._ts = None  # Out of sync until the next snapshot.
                    return []
                self._book.apply(_levels(data["bids"]), _levels(data["asks"]))
                self._ts = message["ts"]
                return [self._book.book(message["ts"])]
            case "trade":
                return [Trade(message["ts"], data["side"].lower(), float(data["price"]), float(data["size"]))]
            case _:
                return []


def _levels(levels: list[list[float]]) -> list[Level]:
    return [(float(price), float(size)) for price, size in levels]
