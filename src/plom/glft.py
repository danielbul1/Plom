"""GLFT quoting: half spread and inventory skew from the closed-form approximation of Guéant, Lehalle
and Fernandez-Tapia (2013), with the fill intensity A exp(-k δ) calibrated online.

Units are bps and seconds: σ is the mid's one-second volatility in bps, δ is a quote's distance
from fair value in bps, A is fills per second at δ = 0 and k is per bp. Inventory is counted in
lots of order_size, so γ is per bp per lot and the skew is bps per lot.

A maker fee f turns a fill at depth δ into δ - f of edge. Writing δ = f + δ' gives intensity
A exp(-k f) exp(-k δ'), the same problem with a smaller A, so the fee is added to the half spread
and A is scaled down before solving.

Calibration follows hftbacktest's GLFT tutorial. Time is cut into sample_ms intervals; in each we
note how far below fair value a sell trade reached (where it would hit our bid) and how far above
it a buy trade reached. λ(δ) is how often per second, per side, an interval reached at least δ; a
line through log λ(δ) gives log A and -k. Counts and exposure decay with half_life_s.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from plom.market import Trade

FIT_EVERY = 50
"""Refit after this many closed intervals."""


@dataclass(frozen=True)
class Glft:
    half_spread_bps: float
    skew_bps: float
    """How far the reservation price moves against each lot of inventory."""


def glft(sigma_bps: float, a: float, k: float, gamma: float, fee_bps: float = 0.0) -> Glft:
    """GLFT half spread and per-lot skew for one lot, with ξ = γ, net of a maker fee."""
    a *= math.exp(-k * fee_bps)
    x = gamma / k
    c1 = math.log1p(x) / gamma
    c2 = math.sqrt(gamma / (2 * a * k) * math.exp((k / gamma + 1) * math.log1p(x)))
    return Glft(fee_bps + c1 + sigma_bps * c2 / 2, sigma_bps * c2)


@dataclass
class FillIntensity:
    sample_ms: int = 100
    half_life_s: float = 600.0
    step_bps: float = 0.25
    """Depths are fitted at step_bps, 2 x step_bps, ... buckets x step_bps."""
    buckets: int = 40
    min_hits: float = 10.0
    """Depths reached fewer (decayed) times than this are left out of the fit."""
    max_gap_ms: int = 5000
    """A longer gap between books, such as a dropped connection, is not counted as exposure."""
    a: float | None = None
    k: float | None = None
    depths_bps: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.depths_bps = self.step_bps * np.arange(1, self.buckets + 1)
        self._hits = np.zeros(self.buckets)
        self._exposure_s = 0.0
        self._intervals = 0
        self._start_ms: int | None = None
        self._center: float | None = None
        self._deepest = {"buy": -math.inf, "sell": -math.inf}

    def rate(self) -> np.ndarray:
        """λ at each depth, per second per side."""
        return self._hits / (2 * self._exposure_s) if self._exposure_s > 0 else np.zeros(self.buckets)

    def on_book(self, now_ms: int, center: float) -> None:
        """Close the interval once sample_ms has passed and start the next around `center`."""
        if self._start_ms is not None:
            elapsed = now_ms - self._start_ms
            if elapsed < self.sample_ms:
                return
            if elapsed <= self.max_gap_ms:
                self._close(elapsed / 1000)
        self._start_ms, self._center = now_ms, center
        self._deepest = {"buy": -math.inf, "sell": -math.inf}

    def on_trade(self, trade: Trade) -> None:
        if self._center is None:
            return
        if trade.side == "sell":
            depth, resting = (self._center - trade.price) / self._center * 10_000, "buy"
        else:
            depth, resting = (trade.price - self._center) / self._center * 10_000, "sell"
        self._deepest[resting] = max(self._deepest[resting], depth)

    def _close(self, dt_s: float) -> None:
        decay = 0.5 ** (dt_s / self.half_life_s)
        self._hits *= decay
        self._exposure_s = self._exposure_s * decay + dt_s
        for depth in self._deepest.values():
            self._hits += depth >= self.depths_bps - 1e-9
        self._intervals += 1
        if self._intervals % FIT_EVERY == 0:
            self._fit()

    def _fit(self) -> None:
        used = self._hits >= self.min_hits
        if used.sum() < 3:
            return
        slope, intercept = np.polyfit(self.depths_bps[used], np.log(self.rate()[used]), 1)
        if slope < 0:
            self.k, self.a = float(-slope), float(math.exp(intercept))
