"""Years of 5-minute open interest, taker flow and price for a coin, from Binance's public archive.

data.binance.vision publishes, for USD-margined perpetuals:

- daily `metrics` files (since December 2021): open interest every 5 minutes, among others;
- monthly and daily 5-minute `klines`: open, high, low, close, volume and taker buy volume.

Files are cached under a directory, so each is downloaded once. The archive has no liquidations, so
real liquidations only exist from when the hub started storing them.
"""

import csv
import io
import logging
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

log = logging.getLogger("plom.history")
ARCHIVE = "https://data.binance.vision/data/futures/um"
FIRST_METRICS = date(2021, 12, 1)
PERIOD_MS = 300_000


@dataclass(frozen=True)
class Bar:
    """One 5-minute interval: its candle, the taker flow within it, and the open interest at its end."""

    time_ms: int
    """The interval's start."""
    open: float
    high: float
    low: float
    close: float
    buy: float
    """Coins bought by takers."""
    sell: float
    oi: float
    """Open interest in coins at the interval's end."""

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3


def load(coin: str, start: date, end: date, cache: Path) -> list[Bar]:
    """Bars from start to end (inclusive days), oldest first, downloading what isn't cached.

    Each bar's open interest is the reading at its end, so it is known when the bar closes."""
    symbol = f"{coin.upper()}USDT"
    start = max(start, FIRST_METRICS)
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    klines: dict[int, tuple[float, ...]] = {}
    oi: dict[int, float] = {}
    with ThreadPoolExecutor(16) as pool:
        for rows in pool.map(lambda p: _rows(p, cache), _kline_paths(symbol, days)):
            for r in rows:
                if r[0].isdigit():  # Newer files start with a header.
                    klines[int(r[0])] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9]))
        for rows in pool.map(lambda p: _rows(p, cache), [f"daily/metrics/{symbol}/{symbol}-metrics-{d}.zip" for d in days]):
            for r in rows:
                if r[0][:1].isdigit() and r[2]:
                    oi[_ms(r[0])] = float(r[2])
    bars = []
    for t in sorted(klines):
        end_oi = oi.get(t + PERIOD_MS)
        if end_oi is None:
            continue
        o, h, l, c, volume, taker_buy = klines[t]
        bars.append(Bar(t, o, h, l, c, taker_buy, volume - taker_buy, end_oi))
    return bars


def _kline_paths(symbol: str, days: list[date]) -> list[str]:
    """Monthly files for whole months that have ended, daily files for the rest."""
    this_month = datetime.now(UTC).date().replace(day=1)
    months = sorted({d.replace(day=1) for d in days})
    paths = []
    for month in months:
        in_month = [d for d in days if d.replace(day=1) == month]
        if month < this_month:
            paths.append(f"monthly/klines/{symbol}/5m/{symbol}-5m-{month:%Y-%m}.zip")
        else:
            paths += [f"daily/klines/{symbol}/5m/{symbol}-5m-{d}.zip" for d in in_month]
    return paths


def _rows(path: str, cache: Path) -> list[list[str]]:
    """The CSV rows of one archive file; a file the archive lacks (yet) gives none."""
    local = cache / path
    if not local.exists():
        try:
            request = urllib.request.Request(f"{ARCHIVE}/{path}", headers={"User-Agent": "plom/0.1"})
            body = urllib.request.urlopen(request, timeout=60).read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return []
            raise
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(body)
    with zipfile.ZipFile(local) as archive:
        text = archive.read(archive.namelist()[0]).decode()
    return list(csv.reader(io.StringIO(text)))


def _ms(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).replace(tzinfo=UTC).timestamp() * 1000)
