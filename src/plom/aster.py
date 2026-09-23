"""Aster's USDT perpetual futures WebSocket, which copies Binance's: aggregated trades, and the book
ticker or, with depth, 20-level book snapshots every 100ms."""

from collections.abc import AsyncIterator

from plom import binance, feed

NAME = "aster"
WS_URL = "wss://fstream.asterdex.com/stream?streams={symbol}@{book}/{symbol}@aggTrade"
TICKER_EVERY_MS = 50


def messages(coin: str, depth: bool = False) -> AsyncIterator[dict]:
    book = "depth20@100ms" if depth else "bookTicker"
    return feed.merged([feed.Source(
        WS_URL.format(symbol=f"{coin.lower()}usdt", book=book), throttle_ms=0 if depth else TICKER_EVERY_MS,
        is_book=lambda m: m.get("data", {}).get("e") == "bookTicker",
    )])


Parser = binance.Parser
