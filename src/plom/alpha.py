"""Online short-horizon forecast of the venue's mid move, from order-flow features.

Features, sampled at most every sample_ms:
- gap: the basis-adjusted reference minus the local fair price, in bps (0 without a fresh reference);
- ofi: order-flow imbalance at the top of book (Cont, Kukanov and Stoikov 2014), decayed over
  flow_half_life_ms and divided by typical top-of-book depth;
- tfi: signed trade volume over total trade volume, decayed the same way, in [-1, 1];
- micro: microprice minus mid, in bps.

The target is the mid's move over horizon_ms, in bps. Weights come from exponentially-forgetting
ridge regression, updated only when a sample's target has happened, so every forecast uses the past
alone and its score on later samples is out of sample.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from plom.market import Book, Trade

FEATURES = ("gap", "ofi", "tfi", "micro")
SOLVE_EVERY = 20


@dataclass(frozen=True)
class AlphaConfig:
    horizon_ms: int = 1000
    sample_ms: int = 100
    half_life_samples: float = 3000.0
    """How fast old samples are forgotten: 3000 samples at 100ms is five minutes."""
    ridge: float = 1.0
    warmup_samples: int = 600
    flow_half_life_ms: int = 1000


@dataclass
class OnlineAlpha:
    config: AlphaConfig = field(default_factory=AlphaConfig)
    prediction_bps: float = 0.0
    """Forecast of the mid move over the horizon; 0 until warmed up."""
    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))
    samples: int = 0
    scored: int = 0
    squared_error: float = 0.0
    squared_target: float = 0.0

    def __post_init__(self) -> None:
        n = len(FEATURES)
        self._xtx = np.zeros((n, n))
        self._xty = np.zeros(n)
        self._forget = 0.5 ** (1 / self.config.half_life_samples)
        self._pending: deque[tuple[int, float, np.ndarray, float | None]] = deque()
        self._last_sample_ms: int | None = None
        self._previous_top: tuple[float, float, float, float] | None = None
        self._flow_ms: int | None = None
        self._ofi = 0.0
        self._depth = 0.0
        self._signed_volume = 0.0
        self._volume = 0.0

    @property
    def r2(self) -> float | None:
        """Out-of-sample R squared of forecasts made after warmup."""
        if self.scored < 2 or self.squared_target == 0:
            return None
        return 1 - self.squared_error / self.squared_target

    def on_book(self, now_ms: int, book: Book, mid: float, gap_bps: float) -> None:
        (bid, bid_size), (ask, ask_size) = book.bids[0], book.asks[0]
        self._decay_flow(now_ms)
        if self._previous_top is not None:
            self._ofi += order_flow_imbalance(self._previous_top, (bid, bid_size, ask, ask_size))
        self._previous_top = (bid, bid_size, ask, ask_size)
        self._depth = (bid_size + ask_size) / 2 if self._depth == 0 else 0.99 * self._depth + 0.01 * (bid_size + ask_size) / 2
        self._settle(now_ms, mid)
        if self._last_sample_ms is not None and now_ms - self._last_sample_ms < self.config.sample_ms:
            return
        self._last_sample_ms = now_ms
        total = bid_size + ask_size
        microprice = (bid * ask_size + ask * bid_size) / total if total > 0 else mid
        x = np.array([
            gap_bps,
            self._ofi / self._depth if self._depth > 0 else 0.0,
            self._signed_volume / self._volume if self._volume > 0 else 0.0,
            (microprice / mid - 1) * 10_000,
        ])
        warmed = self.samples >= self.config.warmup_samples
        self.prediction_bps = float(self.weights @ x) if warmed else 0.0
        self._pending.append((now_ms, mid, x, self.prediction_bps if warmed else None))

    def on_trade(self, now_ms: int, trade: Trade) -> None:
        self._decay_flow(now_ms)
        sign = 1.0 if trade.side == "buy" else -1.0
        self._signed_volume += sign * trade.size
        self._volume += trade.size

    def _decay_flow(self, now_ms: int) -> None:
        if self._flow_ms is not None and now_ms > self._flow_ms:
            decay = 0.5 ** ((now_ms - self._flow_ms) / self.config.flow_half_life_ms)
            self._ofi *= decay
            self._signed_volume *= decay
            self._volume *= decay
        self._flow_ms = now_ms if self._flow_ms is None else max(self._flow_ms, now_ms)

    def _settle(self, now_ms: int, mid: float) -> None:
        """Learn from samples whose horizon has passed."""
        while self._pending and self._pending[0][0] + self.config.horizon_ms <= now_ms:
            _, start_mid, x, forecast = self._pending.popleft()
            y = math.log(mid / start_mid) * 10_000
            if forecast is not None:
                self.scored += 1
                self.squared_error += (y - forecast) ** 2
                self.squared_target += y**2
            self._xtx = self._forget * self._xtx + np.outer(x, x)
            self._xty = self._forget * self._xty + x * y
            self.samples += 1
            if self.samples % SOLVE_EVERY == 0:
                penalty = self.config.ridge * np.eye(len(FEATURES))
                self.weights = np.linalg.solve(self._xtx + penalty, self._xty)


def order_flow_imbalance(
    previous: tuple[float, float, float, float], current: tuple[float, float, float, float]
) -> float:
    """Net size added to the bid minus net size added to the ask between two top-of-book states.

    Each state is (bid, bid_size, ask, ask_size). A higher bid, or more size at the same bid, is
    buying pressure; a lower ask, or more size at the same ask, is selling pressure.
    """
    bid0, bid_size0, ask0, ask_size0 = previous
    bid1, bid_size1, ask1, ask_size1 = current
    flow = (bid_size1 if bid1 >= bid0 else 0.0) - (bid_size0 if bid1 <= bid0 else 0.0)
    flow -= (ask_size1 if ask1 <= ask0 else 0.0) - (ask_size0 if ask1 >= ask0 else 0.0)
    return flow
