"""Live open interest and forced liquidations from the venues that publish them.

Open interest (in coins): OKX's `open-interest` channel (every few seconds), Bybit's tickers
(on every change), Hyperliquid's asset context, and Binance's REST endpoint polled every few
seconds (Binance streams no open interest; some regions are refused, in which case it just fails).

Liquidations: Binance's `forceOrder` stream (at most one per symbol a second), Bybit's
`allLiquidation` (all of them) and OKX's `liquidation-orders` (every swap, filtered here). Each is
reported with the side of the position that was liquidated.
"""

import asyncio
import json
import time
import urllib.request
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal

from plom import feed, hyperliquid

OI_POLL_S = 5.0


@dataclass(frozen=True)
class OpenInterest:
    venue: str
    time_ms: int
    coins: float


@dataclass(frozen=True)
class Liquidation:
    venue: str
    time_ms: int
    position: Literal["long", "short"]
    """The side of the position that was liquidated: a long is closed by a forced sell."""
    price: float
    size: float
    """In coins."""

    def to_json(self) -> dict:
        return {
            "venue": self.venue, "ts_ms": self.time_ms, "position": self.position, "price": self.price,
            "size": self.size, "notional": self.price * self.size,
        }


# Open interest ---------------------------------------------------------------------------------

def _okx_oi(coin: str) -> AsyncIterator[dict]:
    inst = f"{coin.upper()}-USDT-SWAP"
    return feed.merged([feed.Source(
        "wss://ws.okx.com/ws/v5/public", [{"op": "subscribe", "args": [{"channel": "open-interest", "instId": inst}]}],
        keep=lambda m: "data" in m,
    )])


def _bybit_tickers(coin: str, topic: str = "tickers") -> AsyncIterator[dict]:
    return feed.merged([feed.Source(
        "wss://stream.bybit.com/v5/public/linear", [{"op": "subscribe", "args": [f"{topic}.{coin.upper()}USDT"]}],
        keep=lambda m: "topic" in m, heartbeat=lambda: {"op": "ping"},
    )])


async def _hyperliquid_ctx(coin: str) -> AsyncIterator[dict]:
    async for message in hyperliquid.messages(coin, channels=("activeAssetCtx",)):
        yield message


async def _binance_oi(coin: str) -> AsyncIterator[dict]:
    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={coin.upper()}USDT"
    request = urllib.request.Request(url, headers={"User-Agent": "plom/0.1"})
    while True:
        body = await asyncio.to_thread(lambda: json.load(urllib.request.urlopen(request, timeout=10)))
        yield body
        await asyncio.sleep(OI_POLL_S)


def _parse_okx_oi(message: dict) -> list[OpenInterest]:
    return [OpenInterest("okx", int(d["ts"]), float(d["oiCcy"])) for d in message["data"]]


def _parse_bybit_oi(message: dict) -> list[OpenInterest]:
    data = message.get("data", {})
    if "openInterest" not in data:
        return []  # A delta without an open interest change.
    return [OpenInterest("bybit", int(message["ts"]), float(data["openInterest"]))]


def _parse_hyperliquid_oi(message: dict) -> list[OpenInterest]:
    ctx = message.get("data", {}).get("ctx", {})
    return [OpenInterest("hyperliquid", int(time.time() * 1000), float(ctx["openInterest"]))] if "openInterest" in ctx else []


def _parse_binance_oi(message: dict) -> list[OpenInterest]:
    return [OpenInterest("binance", int(message["time"]), float(message["openInterest"]))] if "openInterest" in message else []


OI_SOURCES: dict[str, tuple[Callable[[str], AsyncIterator[dict]], Callable[[dict], list[OpenInterest]]]] = {
    "okx": (_okx_oi, _parse_okx_oi),
    "bybit": (_bybit_tickers, _parse_bybit_oi),
    "hyperliquid": (_hyperliquid_ctx, _parse_hyperliquid_oi),
    "binance": (_binance_oi, _parse_binance_oi),
}


# Liquidations ----------------------------------------------------------------------------------

def _binance_liquidations(coin: str) -> AsyncIterator[dict]:
    return feed.merged([feed.Source(f"wss://fstream.binance.com/market/ws/{coin.lower()}usdt@forceOrder")])


def _bybit_liquidations(coin: str) -> AsyncIterator[dict]:
    return _bybit_tickers(coin, topic="allLiquidation")


def _okx_liquidations(coin: str) -> AsyncIterator[dict]:
    return feed.merged([feed.Source(
        "wss://ws.okx.com/ws/v5/public",
        [{"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}],
        keep=lambda m: "data" in m,
    )])


def _parse_binance_liquidation(message: dict, coin: str, contract: float) -> list[Liquidation]:
    order = message.get("o")
    if not order:
        return []
    # A forced SELL closes a long.
    position = "long" if order["S"] == "SELL" else "short"
    price = float(order.get("ap") or order["p"])
    return [Liquidation("binance", int(order["T"]), position, price, float(order.get("z") or order["q"]))]


def _parse_bybit_liquidation(message: dict, coin: str, contract: float) -> list[Liquidation]:
    # S is the liquidated position's side: "Buy" means a long was liquidated.
    return [
        Liquidation("bybit", int(d["T"]), "long" if d["S"] == "Buy" else "short", float(d["p"]), float(d["v"]))
        for d in message.get("data", [])
    ]


def _parse_okx_liquidation(message: dict, coin: str, contract: float) -> list[Liquidation]:
    found = []
    for item in message.get("data", []):
        if item.get("instId") != f"{coin.upper()}-USDT-SWAP":
            continue
        for d in item.get("details", []):
            found.append(Liquidation("okx", int(d["ts"]), d["posSide"], float(d["bkPx"]), float(d["sz"]) * contract))
    return found


LIQUIDATION_SOURCES: dict[str, tuple[Callable[[str], AsyncIterator[dict]], Callable[[dict, str, float], list[Liquidation]]]] = {
    "binance": (_binance_liquidations, _parse_binance_liquidation),
    "bybit": (_bybit_liquidations, _parse_bybit_liquidation),
    "okx": (_okx_liquidations, _parse_okx_liquidation),
}


# History ---------------------------------------------------------------------------------------

OKX_REST = "https://www.okx.com/api/v5"
OKX_PAGE = 100
SEED_PERIOD_MS = 300_000


def _okx_pages(path: str, since_ms: int, cursor: str = "end") -> list[list[str]]:
    """Rows of an OKX endpoint that pages backwards, newest first, back to since_ms. The trading
    statistics endpoints take the cursor as `end`, the candle endpoints as `after`."""
    rows: list[list[str]] = []
    oldest = ""
    while True:
        url = f"{OKX_REST}{path}&limit={OKX_PAGE}{f'&{cursor}={oldest}' if oldest else ''}"
        request = urllib.request.Request(url, headers={"User-Agent": "plom/0.1"})
        page = json.load(urllib.request.urlopen(request, timeout=20)).get("data", [])
        if not page or (oldest and int(page[-1][0]) >= int(oldest)):
            break  # Nothing older: the end of the history, or a cursor the endpoint ignores.
        rows += page
        oldest = page[-1][0]
        if len(page) < OKX_PAGE or int(oldest) <= since_ms:
            break
        time.sleep(0.2)  # OKX allows a few requests a second on these endpoints.
    return [r for r in rows if int(r[0]) >= since_ms]


def okx_history(coin: str, hours: float) -> list[tuple[int, float, float, float, float, float, float]]:
    """(time, open interest in coins, taker buy, taker sell, low, high, typical price) per 5 minutes, oldest first."""
    inst = f"{coin.upper()}-USDT-SWAP"
    since = int(time.time() * 1000 - hours * 3_600_000)
    oi = {int(r[0]): float(r[2]) for r in _okx_pages(f"/rubik/stat/contracts/open-interest-history?instId={inst}&period=5m", since)}
    # Taker volume rows are [ts, sell, buy] in contracts; only their ratio is used.
    taker = {int(r[0]): (float(r[2]), float(r[1])) for r in _okx_pages(f"/rubik/stat/taker-volume-contract?instId={inst}&period=5m", since)}
    candles = {int(r[0]): (float(r[3]), float(r[2]), (float(r[2]) + float(r[3]) + float(r[4])) / 3)
               for r in _okx_pages(f"/market/history-candles?instId={inst}&bar=5m", since, cursor="after")}
    return [
        (t, oi[t], *taker.get(t, (0.0, 0.0)), *candles[t])
        for t in sorted(oi) if t in candles
    ]
