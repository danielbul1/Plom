"""Binance USD-M futures public WebSockets: best bid/ask and aggregated trades, as a reference price.

Binance serves these on separate endpoints (/public for book tickers, /market for trades), so we
hold one connection to each and merge them. Only the top of book is streamed, so books carry one
level per side.
"""

from collections.abc import AsyncIterator

from plom import feed
from plom.market import Book, Trade

NAME = "binance"
TICKER_URL = "wss://fstream.binance.com/public/stream?streams={symbol}@bookTicker"
TRADES_URL = "wss://fstream.binance.com/market/stream?streams={symbol}@aggTrade"
TICKER_EVERY_MS = 50
"""BTC's book ticker changes ~1,000 times a second; keep only the first one per interval."""


def messages(coin: str) -> AsyncIterator[dict]:
    """Yield raw combined-stream messages from both endpoints, reconnecting each if it drops."""
    symbol = f"{coin.lower()}usdt"
    return feed.merged([
        feed.Source(TICKER_URL.format(symbol=symbol), throttle_ms=TICKER_EVERY_MS),
        feed.Source(TRADES_URL.format(symbol=symbol)),
    ])


class Parser:
    def events(self, message: dict) -> list[Book | Trade]:
        data = message["data"]
        match data.get("e"):
            case "bookTicker":
                bids = [(float(data["b"]), float(data["B"]))]
                asks = [(float(data["a"]), float(data["A"]))]
                return [Book(data["T"], bids, asks)]
            case "depthUpdate":  # A partial-depth snapshot, as Aster sends.
                bids = [(float(p), float(q)) for p, q in data["b"]]
                asks = [(float(p), float(q)) for p, q in data["a"]]
                return [Book(data["T"], bids, asks)] if bids and asks else []
            case "aggTrade":
                # m: the buyer was the maker, so the aggressor sold.
                side = "sell" if data["m"] else "buy"
                return [Trade(data["T"], side, float(data["p"]), float(data["q"]))]
            case _:
                return []
