"""A composite reference price across venues, like the aggregated price charting services show.

Venues sit at different levels (spot against perps, USD against USDT, funding), so each venue's mid
is first moved onto a common level by a slowly learned basis. The composite is the median of the
adjusted mids of the venues heard from within `stale_ms`, so a venue joining, dropping out or
glitching barely moves it. Timing is by local receive time, the only clock the venues share.
"""

import math
from statistics import median

from plom.market import Book

NAME = "composite"
VENUES = ("coinbase", "okx", "htx_spot", "htx_perps", "blofin", "aster", "hyperliquid")
"""The venues MattCharts aggregates that we can reach, without Binance or Bybit."""


class Composite:
    def __init__(
        self, venues: tuple[str, ...], stale_ms: float = 1000.0, basis_half_life_s: float = 300.0, min_venues: int = 3,
    ) -> None:
        self.venues = venues
        self.stale_ms = stale_ms
        self.basis_half_life_s = basis_half_life_s
        self.min_venues = min_venues
        self._last: dict[str, tuple[float, float]] = {}
        """Venue -> (receive time, log mid)."""
        self.basis: dict[str, float] = {}
        """Venue -> learned log(venue mid / composite)."""
        self.level: float | None = None
        """The composite's log price."""

    def on_book(self, venue: str, recv_ms: float, book: Book) -> Book | None:
        """Take a venue's book; return a one-level composite book when enough venues are fresh."""
        if venue not in self.venues or not book.bids or not book.asks:
            return None
        log_mid = math.log((book.bids[0][0] + book.asks[0][0]) / 2)
        previous = self._last.get(venue)
        self._last[venue] = (recv_ms, log_mid)
        fresh = {v: m for v, (t, m) in self._last.items() if recv_ms - t <= self.stale_ms}
        if len(fresh) < self.min_venues:
            return None
        if self.level is None:
            self.level = median(fresh.values())
        for v, m in fresh.items():
            self.basis.setdefault(v, m - self.level)  # A new venue joins at the current level.
        self.level = median(m - self.basis[v] for v, m in fresh.items())
        if previous is not None:
            dt_s = max(recv_ms - previous[0], 0.0) / 1000
            weight = 1 - 0.5 ** (dt_s / self.basis_half_life_s)
            self.basis[venue] += weight * (log_mid - self.level - self.basis[venue])
        price = math.exp(self.level)
        return Book(round(recv_ms), [(price, 0.0)], [(price, 0.0)])
