"""OHLCV candles: built live from every venue's trades, stored in SQLite, backfilled from venue history.

Live candles take every venue's trades, with prices moved onto the composite's level by each venue's
learned basis, so one venue trading a few basis points away doesn't stretch the highs and lows.
They carry taker buy and sell volume. Candles for every interval are built directly from trades.

Where the store has gaps, candles are backfilled from Coinbase (deep history, but no 4h) and
Hyperliquid (only its most recent 5,000 candles per interval): high and low across sources, open
and close their median, volume and trade counts summed. Backfilled candles cover fewer venues and
have no buy/sell split, and a live candle is never overwritten by a backfilled one.
"""

import json
import math
import sqlite3
import statistics
import threading
import time
import urllib.request
from dataclasses import astuple, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

INTERVALS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
COINBASE_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
COINBASE_PAGE = 300


@dataclass
class Candle:
    open_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float | None
    sell_volume: float | None
    trade_count: int
    source: str
    """"live" or "backfill:<venues>"."""
    partial: bool = False
    """A live candle whose interval started before we were watching."""

    def to_json(self) -> dict:
        row = {f.name: getattr(self, f.name) for f in fields(self)}
        row["partial"] = bool(row["partial"])
        return row


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS candles (symbol TEXT, interval TEXT, open_ms INTEGER, open REAL, high REAL,"
                " low REAL, close REAL, volume REAL, buy_volume REAL, sell_volume REAL, trade_count INTEGER,"
                " source TEXT, partial INTEGER, PRIMARY KEY (symbol, interval, open_ms))"
            )

    def write_live(self, symbol: str, interval: str, candles: list[Candle]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(symbol, interval, *astuple(c)) for c in candles],
            )

    def write_backfill(self, symbol: str, interval: str, candles: list[Candle]) -> None:
        """Fill gaps and replace partial live candles, never complete live ones."""
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (symbol, interval, open_ms) DO UPDATE SET open=excluded.open, high=excluded.high,"
                " low=excluded.low, close=excluded.close, volume=excluded.volume, buy_volume=excluded.buy_volume,"
                " sell_volume=excluded.sell_volume, trade_count=excluded.trade_count, source=excluded.source,"
                " partial=0 WHERE candles.partial = 1 OR candles.source != 'live'",
                [(symbol, interval, *astuple(c)) for c in candles],
            )

    def closed_partials(self, symbol: str, interval: str, now_ms: int) -> list[int]:
        """Open times of partial live candles whose interval has ended, so backfill can complete them."""
        with self._lock:
            rows = self._db.execute(
                "SELECT open_ms FROM candles WHERE symbol = ? AND interval = ? AND partial = 1 AND open_ms + ? <= ?",
                (symbol, interval, INTERVALS[interval], now_ms),
            ).fetchall()
        return [row[0] for row in rows]

    def read(self, symbol: str, interval: str, from_ms: int, to_ms: int) -> list[Candle]:
        with self._lock:
            rows = self._db.execute(
                "SELECT open_ms, open, high, low, close, volume, buy_volume, sell_volume, trade_count, source, partial"
                " FROM candles WHERE symbol = ? AND interval = ? AND open_ms >= ? AND open_ms <= ? ORDER BY open_ms",
                (symbol, interval, from_ms, to_ms),
            ).fetchall()
        return [Candle(*row) for row in rows]


class LiveCandles:
    """The candles currently forming for one symbol, one per interval."""

    def __init__(self, started_ms: float) -> None:
        self.started_ms = started_ms
        self.current: dict[str, Candle] = {}
        self.finished: dict[str, list[Candle]] = {interval: [] for interval in INTERVALS}

    def on_trade(self, time_ms: int, price: float, size: float, side: str) -> None:
        for interval, length in INTERVALS.items():
            open_ms = time_ms - time_ms % length
            candle = self.current.get(interval)
            if candle is not None and open_ms < candle.open_ms:
                continue  # A late trade for a candle already closed; rare, and too late to matter.
            if candle is None or open_ms > candle.open_ms:
                if candle is not None:
                    self.finished[interval].append(candle)
                candle = Candle(open_ms, price, price, price, price, 0.0, 0.0, 0.0, 0, "live", open_ms < self.started_ms)
                self.current[interval] = candle
            candle.high = max(candle.high, price)
            candle.low = min(candle.low, price)
            candle.close = price
            candle.volume += size
            if side == "buy":
                candle.buy_volume += size
            else:
                candle.sell_volume += size
            candle.trade_count += 1

    def flush(self, symbol: str, store: Store) -> None:
        """Write finished candles and the ones still forming."""
        for interval in INTERVALS:
            pending = self.finished[interval]
            if interval in self.current:
                pending = [*pending, self.current[interval]]
            if pending:
                store.write_live(symbol, interval, pending)
            self.finished[interval] = []


def _get(url: str, body: dict | None = None) -> object:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": "plom/0.1", **({"Content-Type": "application/json"} if data else {})}
    with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=20) as response:
        return json.load(response)


def fetch_coinbase(coin: str, interval: str, from_ms: int, to_ms: int) -> list[Candle]:
    granularity = COINBASE_GRANULARITY.get(interval)
    if granularity is None:
        return []
    candles, step_ms = [], granularity * 1000 * COINBASE_PAGE
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for start in range(from_ms, to_ms, step_ms):
        end = min(start + step_ms - granularity * 1000, to_ms)
        rows = _get(f"https://api.exchange.coinbase.com/products/{coin}-USD/candles?granularity={granularity}&start={iso(start)}&end={iso(end)}")
        for t, low, high, open_, close, volume in rows:
            candles.append(Candle(t * 1000, open_, high, low, close, volume, None, None, 0, "backfill:coinbase"))
        time.sleep(0.15)  # Stay well under Coinbase's public rate limit.
    return candles


def fetch_hyperliquid(coin: str, interval: str, from_ms: int, to_ms: int) -> list[Candle]:
    rows = _get("https://api.hyperliquid.xyz/info", {
        "type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": from_ms, "endTime": to_ms},
    })
    return [
        Candle(r["t"], float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"]), None, None, r["n"], "backfill:hyperliquid")
        for r in rows
    ]


def merge(by_source: dict[str, list[Candle]], basis: dict[str, float] | None = None) -> list[Candle]:
    """One candle per open time across sources, each source first moved by its basis (in log units)."""
    basis = basis or {}
    grouped: dict[int, list[tuple[str, Candle]]] = {}
    for source, candles in by_source.items():
        for candle in candles:
            grouped.setdefault(candle.open_ms, []).append((source, candle))
    merged = []
    for open_ms in sorted(grouped):
        group = grouped[open_ms]
        scale = {source: math.exp(-basis.get(source, 0.0)) for source, _ in group}
        merged.append(Candle(
            open_ms,
            statistics.median(c.open * scale[s] for s, c in group),
            max(c.high * scale[s] for s, c in group),
            min(c.low * scale[s] for s, c in group),
            statistics.median(c.close * scale[s] for s, c in group),
            sum(c.volume for _, c in group),
            None, None,
            sum(c.trade_count for _, c in group),
            "backfill:" + ",".join(sorted(s for s, _ in group)),
        ))
    return merged


FETCHERS = {"coinbase": fetch_coinbase, "hyperliquid": fetch_hyperliquid}


def backfill(store: Store, symbol: str, interval: str, from_ms: int, to_ms: int, basis: dict[str, float] | None = None) -> int:
    """Fetch every source for the window and store the merged candles; returns how many were written."""
    coin = symbol.split("-")[0]
    by_source = {}
    for name, fetch in FETCHERS.items():
        try:
            by_source[name] = fetch(coin, interval, from_ms, to_ms)
        except Exception:  # One source failing leaves the other's candles.
            continue
    candles = merge(by_source, basis)
    store.write_backfill(symbol, interval, candles)
    return len(candles)
