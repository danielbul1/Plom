"""Does the estimated liquidation map carry information? An event study over years of 5-minute bars.

The liquidation model is replayed bar by bar, so at every bar it knows only the bars before it. Three
questions, one per way of trading the zones:

- Cascade or sweep: when a bar's range crosses a large share of the estimated positions, does the
  price keep going the same way (a cascade: trade with it) or come back (a sweep: fade it)? Each
  event's forward return is signed by the breakout's direction, so positive means continuation.
  Big bars tend to continue or revert on their own, so each is compared with bars of the same size
  and direction that crossed nothing: the difference is what the map adds.
- Magnet: when far more estimated positions sit on one side near the price, does the price go there?

Means come with block-bootstrap intervals (a block is a day, since nearby events are not independent).
"""

import math
from bisect import bisect_left, insort
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from plom.breakout.history import Bar
from plom.evaluate import Interval, _block_bootstrap, _weighted_mean
from plom.hub.liquidations import LiquidationModel

HORIZONS = {"5m": 1, "15m": 3, "1h": 12, "4h": 48}
"""Forward horizons, in bars."""
WINDOW_BARS = 30 * 288
"""Event thresholds are percentiles of the trailing 30 days, so they use no future information."""
EVENT_PERCENTILE = 0.99
MIN_HISTORY_BARS = 7 * 288
"""Bars replayed before any event counts, so the model has built up."""
MOVE_BINS_BPS = (0, 10, 20, 40, 80, 160, 320)
MAGNET_EVERY_BARS = 12
MAGNET_NEAR, MAGNET_FAR = 0.005, 0.015
"""The band on each side of the price, as a fraction of it, whose estimated positions are compared."""
MAGNET_TARGET = 0.01
MAGNET_WITHIN_BARS = 288
SHOCK_SIGMAS = 6.0
"""A bar moving more than this many trailing standard deviations is a shock (often news)."""
DAY_MS = 86_400_000


@dataclass
class Event:
    time_ms: int
    direction: int
    """-1 when longs were crossed (the price broke down), +1 for shorts."""
    share: float
    """Estimated coins crossed as a share of open interest."""
    move_bps: float
    """The bar's close against its open, in the breakout's direction."""
    oi_change: float
    """The bar's change in open interest, as a share: liquidations close positions."""
    shock: bool
    forward_bps: dict[str, float]
    """Close to close, signed by the direction: positive is continuation."""
    excess_bps: dict[str, float] = field(default_factory=dict)
    """forward_bps less the mean of same-size, same-direction bars that crossed nothing."""


@dataclass
class MagnetSample:
    time_ms: int
    imbalance: float
    """log(estimated shorts in the band above / longs in the band below): positive means more above."""
    forward_bps: dict[str, float]
    up_first: bool | None
    """Whether the price went MAGNET_TARGET up before as far down, within a day (None: neither)."""


@dataclass
class Study:
    coin: str
    bars: int
    events: list[Event]
    magnet: list[MagnetSample]


def run(coin: str, bars: list[Bar]) -> Study:
    model = LiquidationModel()
    closes = [b.close for b in bars]
    events: list[Event] = []
    magnet: list[MagnetSample] = []
    controls: dict[tuple[int, int], list[list[float]]] = defaultdict(lambda: [[] for _ in HORIZONS])
    window: deque[float] = deque()
    ranked: list[float] = []
    returns: deque[float] = deque(maxlen=288)
    previous_oi = None
    for i, bar in enumerate(bars):
        # The model as of the previous bar's end meets this bar's range: what it crosses was estimated earlier.
        crossed = model.on_interval("binance", bar.time_ms + 300_000, bar.oi, bar.buy, bar.sell, bar.low, bar.high, bar.typical)
        longs = sum(c.coins for c in crossed if c.side == "long")
        shorts = sum(c.coins for c in crossed if c.side == "short")
        share = max(longs, shorts) / bar.oi if bar.oi else 0.0
        direction = -1 if longs >= shorts else 1
        bar_bps = (bar.close / bar.open - 1) * 1e4
        sigma = _std(returns)
        shock = sigma > 0 and abs(bar_bps) > SHOCK_SIGMAS * sigma
        returns.append(bar_bps)
        oi_change = bar.oi / previous_oi - 1 if previous_oi else 0.0
        previous_oi = bar.oi
        if i >= MIN_HISTORY_BARS and i + max(HORIZONS.values()) < len(bars):
            if share > 0 and ranked and share >= ranked[int(EVENT_PERCENTILE * (len(ranked) - 1))]:
                events.append(Event(bar.time_ms, direction, share, bar_bps * direction, oi_change, shock, _forward(closes, i, direction)))
            elif share == 0:
                d = 1 if bar_bps >= 0 else -1
                for h, value in zip(controls[(d, _bin(abs(bar_bps)))], _forward(closes, i, d).values()):
                    h.append(value)
            if i % MAGNET_EVERY_BARS == 0:
                sample = _magnet(model, bars, closes, i)
                if sample:
                    magnet.append(sample)
        if share > 0:
            window.append(share)
            insort(ranked, share)
            if len(window) > WINDOW_BARS // 10:  # Nonzero bars are a minority; keep about a month of them.
                ranked.pop(bisect_left(ranked, window.popleft()))
    means = {key: [sum(v) / len(v) if v else 0.0 for v in values] for key, values in controls.items()}
    for e in events:
        # A bar that closed against its breakout is compared with bars that moved the other way.
        key = (e.direction if e.move_bps >= 0 else -e.direction, _bin(abs(e.move_bps)))
        control = means.get(key, [0.0] * len(HORIZONS))
        sign = 1 if key[0] == e.direction else -1  # Control returns are signed by their own move.
        e.excess_bps = {h: e.forward_bps[h] - sign * c for h, c in zip(HORIZONS, control)}
    return Study(coin, len(bars), events, magnet)


def _forward(closes: list[float], i: int, direction: int) -> dict[str, float]:
    return {name: (closes[i + n] / closes[i] - 1) * 1e4 * direction for name, n in HORIZONS.items()}


def _magnet(model: LiquidationModel, bars: list[Bar], closes: list[float], i: int) -> MagnetSample | None:
    price, now = closes[i], bars[i].time_ms + 300_000
    above = sum(model.coins(c, now) for c in model.within("short", price * (1 + MAGNET_NEAR), price * (1 + MAGNET_FAR)))
    below = sum(model.coins(c, now) for c in model.within("long", price * (1 - MAGNET_FAR), price * (1 - MAGNET_NEAR)))
    if above <= 0 or below <= 0:
        return None
    up_first = None
    for bar in bars[i + 1:i + 1 + MAGNET_WITHIN_BARS]:
        up, down = bar.high >= price * (1 + MAGNET_TARGET), bar.low <= price * (1 - MAGNET_TARGET)
        if up or down:
            up_first = None if up and down else up
            break
    return MagnetSample(now, math.log(above / below), _forward(closes, i, 1), up_first)


def _bin(bps: float) -> int:
    return max(k for k, edge in enumerate(MOVE_BINS_BPS) if bps >= edge)


def _std(values) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def by_day(values: list[tuple[int, float]]) -> Interval | None:
    blocks: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for t, v in values:
        blocks[t // DAY_MS].append((v, 1.0))
    return _block_bootstrap(list(blocks.values()), _weighted_mean)


def format_report(s: Study, fee_bps: float = 10.0) -> str:
    lines = [f"{s.coin}: {s.bars} bars of 5 minutes, {len(s.events)} events (top {100 - EVENT_PERCENTILE * 100:g}% of crossings, trailing month)"]
    lines.append("")
    lines.append("Cascade or sweep: forward return in the breakout's direction, bps (positive = continuation, negative = reversal)")
    groups = {
        "all events": s.events,
        "  not shocks": [e for e in s.events if not e.shock],
        "  shocks": [e for e in s.events if e.shock],
        "  longs crossed (down)": [e for e in s.events if e.direction < 0],
        "  shorts crossed (up)": [e for e in s.events if e.direction > 0],
        "  OI fell in the bar": [e for e in s.events if e.oi_change < 0],
        "  OI rose in the bar": [e for e in s.events if e.oi_change >= 0],
    }
    lines.append(f"{'':26}{'n':>6}" + "".join(f"{h:>30}" for h in HORIZONS))
    for name, events in groups.items():
        for kind in ("forward_bps", "excess_bps"):
            label = name if kind == "forward_bps" else "    vs same-size bars"
            cells = [by_day([(e.time_ms, getattr(e, kind)[h]) for e in events]) for h in HORIZONS]
            lines.append(f"{label:26}{len(events):>6}" + "".join(f"{str(c) if c else '-':>30}" for c in cells))
    lines.append("")
    lines.append("By year, 1h forward (bps):")
    years = sorted({_year(e.time_ms) for e in s.events})
    for y in years:
        ev = [e for e in s.events if _year(e.time_ms) == y]
        c = by_day([(e.time_ms, e.forward_bps["1h"]) for e in ev])
        x = by_day([(e.time_ms, e.excess_bps["1h"]) for e in ev])
        lines.append(f"  {y}  n={len(ev):>4}  forward {c or '-'}   vs same-size {x or '-'}")
    lines.append("")
    lines.append("Magnet: with more estimated positions on one side (0.5-1.5% away), where does the price go?")
    ranked = sorted(s.magnet, key=lambda m: m.imbalance)
    q = len(ranked) // 5
    for k, name in enumerate(("far more below", "more below", "balanced", "more above", "far more above")):
        part = ranked[k * q:(k + 1) * q] if k < 4 else ranked[4 * q:]
        if not part:
            continue
        decided = [m for m in part if m.up_first is not None]
        up = sum(m.up_first for m in decided) / len(decided) if decided else float("nan")
        c1 = by_day([(m.time_ms, m.forward_bps["1h"]) for m in part])
        c4 = by_day([(m.time_ms, m.forward_bps["4h"]) for m in part])
        lines.append(f"  {name:16} n={len(part):>6}  up {MAGNET_TARGET:.0%} before down: {up:5.1%}   1h {c1 or '-'}   4h {c4 or '-'}")
    lines.append("")
    lines.append(f"A taker round trip costs about {fee_bps:g}bps: an edge must clear that to trade.")
    return "\n".join(lines)


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, UTC).year
