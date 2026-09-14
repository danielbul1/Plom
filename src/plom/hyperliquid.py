"""Hyperliquid's public WebSocket: live L2 books and trades (no auth needed)."""

import json
from collections.abc import AsyncIterator, Sequence

import websockets

from plom.market import Book, Level, Trade

NAME = "hyperliquid"
WS_URL = "wss://api.hyperliquid.xyz/ws"
CHANNELS = ("l2Book", "trades")


async def messages(coin: str, channels: Sequence[str] = CHANNELS) -> AsyncIterator[dict]:
    """Yield raw channel messages, reconnecting if the socket drops."""
    async for ws in websockets.connect(WS_URL):
        try:
            for channel in channels:
                subscription = {"type": channel, "coin": coin}
                if channel == "l2Book":
                    # Undocumented: without it books arrive every ~5s instead of every ~0.5s.
                    subscription["fast"] = True
                await ws.send(json.dumps({"method": "subscribe", "subscription": subscription}))
            async for raw in ws:
                message = json.loads(raw)
                if message.get("channel") in channels:
                    yield message
        except websockets.ConnectionClosed:
            continue


class Parser:
    """Every Hyperliquid book message is a full snapshot, so nothing carries over between messages."""

    def events(self, message: dict) -> list[Book | Trade]:
        return events(message)


def events(message: dict) -> list[Book | Trade]:
    """Parse a raw channel message. Trade messages can carry several trades."""
    data = message["data"]
    match message["channel"]:
        case "l2Book":
            bids, asks = data["levels"]
            return [Book(data["time"], _levels(bids), _levels(asks))]
        case "trades":
            return [
                Trade(t["time"], "buy" if t["side"] == "B" else "sell", float(t["px"]), float(t["sz"]))
                for t in data
            ]
        case _:
            return []


def _levels(levels: list[dict]) -> list[Level]:
    return [(float(level["px"]), float(level["sz"])) for level in levels]
