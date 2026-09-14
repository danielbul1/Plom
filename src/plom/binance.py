"""Binance USD-M futures public WebSockets: best bid/ask and aggregated trades, as a reference price.

Binance serves these on separate endpoints (/public for book tickers, /market for trades), so we
hold one connection to each and merge them. Only the top of book is streamed, so books carry one
level per side.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import websockets

from plom.market import Book, Trade

NAME = "binance"
TICKER_URL = "wss://fstream.binance.com/public/stream?streams={symbol}@bookTicker"
TRADES_URL = "wss://fstream.binance.com/market/stream?streams={symbol}@aggTrade"
TICKER_EVERY_MS = 50
"""BTC's book ticker changes ~1,000 times a second; keep only the first one per interval."""


async def messages(coin: str) -> AsyncIterator[dict]:
    """Yield raw combined-stream messages from both endpoints, reconnecting each if it drops."""
    symbol = f"{coin.lower()}usdt"
    queue: asyncio.Queue[dict] = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump(TICKER_URL.format(symbol=symbol), queue, TICKER_EVERY_MS)),
        asyncio.create_task(_pump(TRADES_URL.format(symbol=symbol), queue, 0)),
    ]
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait([getter, *tasks], return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()
                for task in done:
                    task.result()  # A pump only finishes by raising.
            yield getter.result()
    finally:
        for task in tasks:
            task.cancel()


async def _pump(url: str, queue: asyncio.Queue[dict], every_ms: float) -> None:
    async for ws in websockets.connect(url):
        try:
            last_ms = 0.0
            async for raw in ws:
                if every_ms:
                    now_ms = time.monotonic() * 1000
                    if now_ms - last_ms < every_ms:
                        continue
                    last_ms = now_ms
                queue.put_nowait(json.loads(raw))
        except websockets.ConnectionClosed:
            continue


class Parser:
    def events(self, message: dict) -> list[Book | Trade]:
        data = message["data"]
        match data.get("e"):
            case "bookTicker":
                bids = [(float(data["b"]), float(data["B"]))]
                asks = [(float(data["a"]), float(data["A"]))]
                return [Book(data["T"], bids, asks)]
            case "aggTrade":
                # m: the buyer was the maker, so the aggressor sold.
                side = "sell" if data["m"] else "buy"
                return [Trade(data["T"], side, float(data["p"]), float(data["q"]))]
            case _:
                return []
