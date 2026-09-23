"""Aster's USDT perpetual futures WebSocket, which copies Binance's: book ticker and aggregated trades."""

from collections.abc import AsyncIterator

from plom import binance, feed

NAME = "aster"
WS_URL = "wss://fstream.asterdex.com/stream?streams={symbol}@bookTicker/{symbol}@aggTrade"
TICKER_EVERY_MS = 50


def messages(coin: str) -> AsyncIterator[dict]:
    return feed.merged([feed.Source(
        WS_URL.format(symbol=f"{coin.lower()}usdt"), throttle_ms=TICKER_EVERY_MS,
        is_book=lambda m: m.get("data", {}).get("e") == "bookTicker",
    )])


Parser = binance.Parser
