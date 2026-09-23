"""An estimate of where leveraged positions would be liquidated, from open interest and taker flow.

No venue publishes positions' entries or leverage, so, like the liquidation heatmaps sold to
traders, this is a model:

- When a venue's open interest rises, positions were opened at about the price its trades printed
  since the last reading. They are split into longs and shorts by that venue's taker flow over the
  same interval (aggressive buying opens longs), then spread over leverage tiers.
- A long at leverage L opened at p is liquidated near p(1 - 1/L + mmr), a short near p(1 + 1/L - mmr).
- When a venue's open interest falls by some share, its estimated positions shrink by that share
  (any open position is as likely as another to be the one closed).
- When the price trades through a level, its positions are gone (liquidated), so it is removed.
- Levels fade with age, faster at high leverage, since those positions rarely last.

Real liquidations from the venues that publish them are kept alongside, to check the model against.
"""

import math
from bisect import bisect_left, bisect_right, insort
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from plom.hub.positioning import Liquidation, OpenInterest

LEVERAGE_TIERS: dict[int, tuple[float, float]] = {
    # leverage: (share of new positions, half-life in hours)
    5: (0.10, 7 * 24),
    10: (0.25, 5 * 24),
    25: (0.30, 3 * 24),
    50: (0.20, 36),
    100: (0.15, 18),
}
MAINTENANCE_MARGIN = 0.005
ENTRY_BUCKET = 1e-4
"""Positions are merged by entry price in buckets of this fraction of the price (1bp)."""
RECENT_LIQUIDATIONS = 2000
_AFTER_ANY_KEY = ("\uffff",)
"""Sorts after every cluster key, so a bisect can include levels equal to a price."""

Key = tuple[str, str, int, int]


@dataclass
class Cluster:
    """Estimated open positions of one venue, side and leverage entered around one price."""

    venue: str
    side: Literal["long", "short"]
    leverage: int
    entry: float
    size: float
    """Coins as of `as_of_ms`, before the venue's later closes (see LiquidationModel.coins)."""
    as_of_ms: int
    log_scale_at: float = 0.0
    """The venue's cumulative log shrink when `size` was set."""
    liquidation_price: float = field(init=False)

    def __post_init__(self) -> None:
        if self.side == "long":
            self.liquidation_price = self.entry * (1 - 1 / self.leverage + MAINTENANCE_MARGIN)
        else:
            self.liquidation_price = self.entry * (1 + 1 / self.leverage - MAINTENANCE_MARGIN)

    def decayed(self, now_ms: int) -> float:
        half_life_ms = LEVERAGE_TIERS[self.leverage][1] * 3_600_000
        return self.size * 0.5 ** (max(now_ms - self.as_of_ms, 0) / half_life_ms)


@dataclass
class _Flow:
    """A venue's trades since its last open interest reading."""

    buy: float = 0.0
    sell: float = 0.0
    notional: float = 0.0

    def add(self, side: str, price: float, size: float) -> None:
        if side == "buy":
            self.buy += size
        else:
            self.sell += size
        self.notional += price * size


@dataclass(frozen=True)
class Crossed:
    """Estimated positions whose level a price move reached: they were (or are being) liquidated."""

    side: Literal["long", "short"]
    leverage: int
    price: float
    coins: float


@dataclass
class LiquidationModel:
    clusters: dict[Key, Cluster] = field(default_factory=dict)
    last_oi: dict[str, float] = field(default_factory=dict)
    flows: dict[str, _Flow] = field(default_factory=dict)
    last_price: float | None = None
    last_ms: int = 0
    liquidations: deque[Liquidation] = field(default_factory=lambda: deque(maxlen=RECENT_LIQUIDATIONS))
    opened: deque[tuple[int, str, float, float, float]] = field(default_factory=lambda: deque(maxlen=5000))
    """(time, venue, price, long coins, short coins) of every estimated opening, for position zones."""
    removed_by_price: float = 0.0
    """Coins of estimated positions removed because the price traded through their level."""
    _log_scale: dict[str, float] = field(default_factory=dict)
    """Per venue, the log of the share of positions still open after every close so far: closes
    shrink all of a venue's positions at once, so they are applied lazily through this."""
    _longs: list[tuple[float, Key]] = field(default_factory=list)
    """Long clusters by liquidation price, ascending; shorts likewise."""
    _shorts: list[tuple[float, Key]] = field(default_factory=list)

    def coins(self, cluster: Cluster, now_ms: int) -> float:
        """A cluster's estimated coins still open at now_ms, after decay and its venue's closes."""
        return cluster.decayed(now_ms) * math.exp(self._log_scale.get(cluster.venue, 0.0) - cluster.log_scale_at)

    def on_trade(self, venue: str, side: str, price: float, size: float) -> None:
        if venue in self.last_oi:  # Flow only matters between a venue's open interest readings.
            self.flows.setdefault(venue, _Flow()).add(side, price, size)
        self.last_price = price
        self._remove_crossed_range(price, price)

    def on_open_interest(self, oi: OpenInterest) -> None:
        previous = self.last_oi.get(oi.venue)
        self.last_oi[oi.venue] = oi.coins
        self.last_ms = max(self.last_ms, oi.time_ms)
        flow = self.flows.pop(oi.venue, _Flow())
        if previous is None:
            return  # The first reading only sets the baseline.
        change = oi.coins - previous
        traded = flow.buy + flow.sell
        price = flow.notional / traded if traded else self.last_price
        if not price or change == 0:
            return
        if change > 0:
            buy_share = flow.buy / traded if traded else 0.5
            self._open(oi.venue, oi.time_ms, price, change * buy_share, change * (1 - buy_share))
        elif previous > 0 and oi.coins > 0:
            # Any open position is as likely to be the one closed, so all shrink by the share of
            # the venue's open interest that closed.
            self._log_scale[oi.venue] = self._log_scale.get(oi.venue, 0.0) + math.log(oi.coins / previous)

    def on_interval(self, venue: str, time_ms: int, oi: float, buy: float, sell: float, low: float, high: float, price: float) -> list[Crossed]:
        """Replay one historical interval ending at time_ms: its range clears the levels it crossed
        (which are returned), then its change in open interest opens or closes positions at its
        typical price, split by its taker flow."""
        crossed = self._remove_crossed_range(low, high, time_ms)
        self.last_price = price
        if venue in self.last_oi:
            self.flows[venue] = _Flow(buy, sell, (buy + sell) * price)
        self.on_open_interest(OpenInterest(venue, time_ms, oi))
        return crossed

    def on_liquidation(self, liquidation: Liquidation) -> None:
        self.liquidations.append(liquidation)

    def within(self, side: str, low: float, high: float) -> list[Cluster]:
        """The clusters of one side whose liquidation price is in [low, high]."""
        levels = self._longs if side == "long" else self._shorts
        return [self.clusters[k] for _, k in levels[bisect_left(levels, (low,)):bisect_right(levels, (high, _AFTER_ANY_KEY))]]

    def _open(self, venue: str, now_ms: int, price: float, longs: float, shorts: float) -> None:
        self.opened.append((now_ms, venue, price, longs, shorts))
        bucket = round(math.log(price) / ENTRY_BUCKET)
        log_scale = self._log_scale.get(venue, 0.0)
        for side, coins in (("long", longs), ("short", shorts)):
            for leverage, (share, _) in LEVERAGE_TIERS.items():
                key = (venue, side, leverage, bucket)
                cluster = self.clusters.get(key)
                if cluster is None:
                    cluster = self.clusters[key] = Cluster(venue, side, leverage, price, coins * share, now_ms, log_scale)
                    insort(self._longs if side == "long" else self._shorts, (cluster.liquidation_price, key))
                else:
                    cluster.size = self.coins(cluster, now_ms) + coins * share
                    cluster.as_of_ms = now_ms
                    cluster.log_scale_at = log_scale

    def _remove_crossed_range(self, low: float, high: float, now_ms: int | None = None) -> list[Crossed]:
        now_ms = self.last_ms if now_ms is None else now_ms
        cut_longs = bisect_left(self._longs, (low,))  # Longs at or above the low were reached.
        cut_shorts = bisect_right(self._shorts, (high, _AFTER_ANY_KEY))  # Shorts at or below the high.
        if cut_longs == len(self._longs) and cut_shorts == 0:
            return []
        keys = [k for _, k in self._longs[cut_longs:]] + [k for _, k in self._shorts[:cut_shorts]]
        del self._longs[cut_longs:]
        del self._shorts[:cut_shorts]
        crossed = []
        for key in keys:
            cluster = self.clusters.pop(key)
            coins = self.coins(cluster, now_ms)
            self.removed_by_price += coins
            crossed.append(Crossed(cluster.side, cluster.leverage, cluster.liquidation_price, coins))
        return crossed

    def levels(self, now_ms: int, bucket_pct: float = 0.1, min_notional: float = 0.0) -> dict:
        """Estimated liquidation notional per price bucket, for longs (below the price) and shorts (above)."""
        width = (self.last_price or 1.0) * bucket_pct / 100
        sides: dict[str, dict[float, float]] = {"long": {}, "short": {}}
        by_leverage: dict[str, dict[int, float]] = {"long": {}, "short": {}}
        for c in self.clusters.values():
            coins = self.coins(c, now_ms)
            price = c.liquidation_price
            notional = coins * price
            key = round(math.floor(price / width) * width, 10)
            sides[c.side][key] = sides[c.side].get(key, 0.0) + notional
            by_leverage[c.side][c.leverage] = by_leverage[c.side].get(c.leverage, 0.0) + notional
        return {
            "price": self.last_price, "bucket": width,
            "longs": [[p, round(n, 2)] for p, n in sorted(sides["long"].items(), reverse=True) if n >= min_notional],
            "shorts": [[p, round(n, 2)] for p, n in sorted(sides["short"].items()) if n >= min_notional],
            "by_leverage": {s: {str(k): round(v, 2) for k, v in sorted(d.items())} for s, d in by_leverage.items()},
            "venues": sorted(self.last_oi),
        }

    def position_zones(self, now_ms: int, window_s: float, bucket_pct: float = 0.1, top: int = 20) -> list[dict]:
        """Where the most positions were opened recently: estimated opened notional per entry-price bucket."""
        width = (self.last_price or 1.0) * bucket_pct / 100
        zones: dict[float, list[float]] = {}
        for t, _, price, longs, shorts in self.opened:
            if now_ms - t > window_s * 1000:
                continue
            key = round(math.floor(price / width) * width, 10)
            zone = zones.setdefault(key, [0.0, 0.0])
            zone[0] += longs * price
            zone[1] += shorts * price
        ranked = sorted(zones.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:top]
        return [{"price": p, "longs": round(l, 2), "shorts": round(s, 2)} for p, (l, s) in ranked]
