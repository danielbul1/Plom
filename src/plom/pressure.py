"""Order-book pressure: distance-weighted bid depth versus ask depth."""

from collections.abc import Sequence
from dataclasses import dataclass

Level = tuple[float, float]
"""A book level as (price, size)."""

DEFAULT_DEPTH = 12
DEFAULT_HALF_LIFE_BPS = 5.0


@dataclass(frozen=True)
class Pressure:
    value: float
    """-1 when all weighted depth is on the ask, +1 when all of it is on the bid."""
    bid_weight: float
    ask_weight: float
    mid: float


def pressure(
    bids: Sequence[Level],
    asks: Sequence[Level],
    depth: int = DEFAULT_DEPTH,
    half_life_bps: float = DEFAULT_HALF_LIFE_BPS,
) -> Pressure | None:
    """Compare weighted depth on the nearest `depth` levels of each side.

    Each level's size is weighted by 0.5 ** (distance_from_mid_bps / half_life_bps),
    so a level `half_life_bps` away from the mid counts half as much as one at the mid.
    Returns None when either side of the book is empty.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if half_life_bps <= 0:
        raise ValueError("half_life_bps must be positive")
    if not bids or not asks:
        return None

    bids = sorted(bids, key=lambda level: level[0], reverse=True)[:depth]
    asks = sorted(asks, key=lambda level: level[0])[:depth]
    mid = (bids[0][0] + asks[0][0]) / 2

    bid_weight = _weighted_depth(bids, mid, half_life_bps)
    ask_weight = _weighted_depth(asks, mid, half_life_bps)
    total = bid_weight + ask_weight
    value = (bid_weight - ask_weight) / total if total > 0 else 0.0
    return Pressure(value, bid_weight, ask_weight, mid)


def _weighted_depth(levels: Sequence[Level], mid: float, half_life_bps: float) -> float:
    return sum(
        size * 0.5 ** (abs(price - mid) / mid * 10_000 / half_life_bps)
        for price, size in levels
    )
