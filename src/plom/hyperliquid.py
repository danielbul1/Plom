"""Hyperliquid's public WebSocket: live L2 books and trades (no auth needed)."""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import websockets

from plom.pressure import Level

WS_URL = "wss://api.hyperliquid.xyz/ws"
CHANNELS = ("l2Book", "trades")


@dataclass(frozen=True)
class Book:
    time_ms: int
    bids: list[Level]
    """Best (highest) first."""
    asks: list[Level]
    """Best (lowest) first."""


@dataclass(frozen=True)
class Trade:
    time_ms: int
    side: Literal["buy", "sell"]
    """The aggressor's side: a "sell" trade hit resting bids."""
    price: float
    size: float


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
