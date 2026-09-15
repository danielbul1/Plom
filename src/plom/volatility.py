"""Jump-robust volatility and jump detection from the mid sampled on a grid.

The mid is sampled at the first book at least sample_ms after the previous sample, and each return
is scaled by sqrt(sample_ms / elapsed) so uneven book arrival doesn't distort it.

- Realized variance averages squared returns, so a single jump inflates it.
- Bipower variation (Barndorff-Nielsen and Shephard 2004) averages π/2 |r_i| |r_i-1|: a lone jump is
  multiplied by an ordinary neighbouring return, so it barely counts. Realized minus bipower is the
  part of the variance that came from jumps.
- Lee and Mykland (2008) divide a return by the local bipower volatility of the returns before it.
  Without jumps, the largest of n such ratios follows a Gumbel law; with n the samples in a day, a
  ratio beyond the resulting threshold is a jump, with about alpha false jumps a day.

To catch a jump when it happens rather than when its sample closes, the move since the last sample
is tested on every mid, at most one jump per sample. Watching the path within a sample, not just its
end, at most doubles the false jumps (the reflection principle).

On one second of BTC books, most mids haven't moved and bipower variation reads far below realized
variance on every venue we record; the gap shrinks to noise by a 10-30 second grid. Short grids see
tick discreteness, not jumps, hence the 10 second default.

Averages are exponentially weighted. Volatilities are in bps over one second.
"""

import math
from dataclasses import dataclass

C = math.sqrt(2 / math.pi)
"""E|Z| for a standard normal Z."""
DAY_MS = 86_400_000


@dataclass
class TimeEwma:
    """Exponentially weighted mean over time, corrected for starting from nothing."""

    half_life_s: float
    _sum: float = 0.0
    _weight: float = 0.0

    def add(self, value: float, dt_s: float) -> None:
        alpha = 1 - 0.5 ** (dt_s / self.half_life_s)
        self._sum = (1 - alpha) * self._sum + alpha * value
        self._weight = (1 - alpha) * self._weight + alpha

    @property
    def mean(self) -> float:
        return self._sum / self._weight if self._weight > 0 else 0.0


def lee_mykland_threshold(n: float, alpha: float) -> float:
    """|L| beyond which a return is a jump, when n returns are tested at family-wise significance alpha."""
    root = math.sqrt(2 * math.log(n))
    c_n = root / C - (math.log(math.pi) + math.log(math.log(n))) / (2 * C * root)
    s_n = 1 / (C * root)
    return c_n - s_n * math.log(-math.log(1 - alpha))


@dataclass
class GridVolatility:
    sample_ms: int = 10_000
    half_life_s: float = 30.0
    baseline_half_life_s: float = 600.0
    alpha: float = 0.0
    """Lee-Mykland significance, about alpha false jumps a day; 0 never reports a jump."""
    local_half_life_samples: float = 30.0
    """How many samples the Lee-Mykland local volatility remembers; jumps are tested once it has seen this many."""
    floor_bps: float = 0.05
    """Smallest local volatility per sample, so a stretch of unchanged mids doesn't make any move a jump."""
    max_gap_ms: int = 5000
    """No mid for longer than this, such as a dropped connection, restarts the return chain."""

    def __post_init__(self) -> None:
        self.realized = TimeEwma(self.half_life_s)
        self.bipower = TimeEwma(self.half_life_s)
        self.bipower_baseline = TimeEwma(self.baseline_half_life_s)
        self.threshold = lee_mykland_threshold(DAY_MS / self.sample_ms, self.alpha) if self.alpha > 0 else math.inf
        self.samples = 0
        self.jumps = 0
        self.ratio: float | None = None
        """Lee-Mykland ratio of the move since the last sample, as of the last mid."""
        self._local = TimeEwma(self.local_half_life_samples)
        self._local_samples = 0
        self._last: tuple[int, float] | None = None
        """The last sample: time and mid."""
        self._seen_ms: int | None = None
        self._previous_abs: float | None = None
        self._jumped_in: int | None = None
        """Start time of the sample in which a jump was last reported."""

    @property
    def realized_bps(self) -> float:
        return math.sqrt(self.realized.mean)

    @property
    def bipower_bps(self) -> float:
        return math.sqrt(self.bipower.mean)

    @property
    def jump_share(self) -> float:
        """Share of recent realized variance that bipower variation attributes to jumps."""
        return max(0.0, 1 - self.bipower.mean / self.realized.mean) if self.realized.mean > 0 else 0.0

    def on_mid(self, now_ms: int, mid: float) -> bool:
        """Feed a mid. True when it reveals a jump in the current sample, the first time it does."""
        gap = self._seen_ms is None or now_ms - self._seen_ms > self.max_gap_ms
        self._seen_ms = now_ms
        if gap or self._last is None:
            self._last, self._previous_abs, self.ratio = (now_ms, mid), None, None
            return False
        start_ms, start_mid = self._last
        elapsed = now_ms - start_ms
        r = math.log(mid / start_mid) * 10_000 * math.sqrt(self.sample_ms / max(elapsed, self.sample_ms))
        local = math.sqrt(self._local.mean)
        warm = self._local_samples >= self.local_half_life_samples
        self.ratio = r / max(local, self.floor_bps) if warm else None
        jumped = self.ratio is not None and abs(self.ratio) >= self.threshold and self._jumped_in != start_ms
        if jumped:
            self.jumps += 1
            self._jumped_in = start_ms
        if elapsed >= self.sample_ms:
            self._close(r, now_ms, mid, elapsed / 1000)
        return jumped

    def _close(self, r: float, now_ms: int, mid: float, dt_s: float) -> None:
        self._last = (now_ms, mid)
        self.samples += 1
        sample_s = self.sample_ms / 1000
        self.realized.add(r * r / sample_s, dt_s)
        if self._previous_abs is not None:
            product = abs(r) * self._previous_abs
            self._local.add(product, 1.0)
            self._local_samples += 1
            self.bipower.add(math.pi / 2 * product / sample_s, dt_s)
            self.bipower_baseline.add(math.pi / 2 * product / sample_s, dt_s)
        self._previous_abs = abs(r)
