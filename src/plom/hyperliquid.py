"""Live L2 order books from Hyperliquid's public WebSocket (no auth needed)."""

import json
from collections.abc import AsyncIterator

import websockets

from plom.pressure import Level

WS_URL = "wss://api.hyperliquid.xyz/ws"


async def l2_books(coin: str) -> AsyncIterator[tuple[list[Level], list[Level]]]:
    """Yield (bids, asks) for every book update, reconnecting if the socket drops."""
    subscribe = {"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}
    async for ws in websockets.connect(WS_URL):
        try:
            await ws.send(json.dumps(subscribe))
            async for raw in ws:
                message = json.loads(raw)
                if message.get("channel") != "l2Book":
                    continue
                bids, asks = message["data"]["levels"]
                yield _parse(bids), _parse(asks)
        except websockets.ConnectionClosed:
            continue


def _parse(levels: list[dict]) -> list[Level]:
    return [(float(level["px"]), float(level["sz"])) for level in levels]
