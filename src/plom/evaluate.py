"""Judge a paper market maker: PnL attribution, markouts by horizon, and block-bootstrap confidence.

PnL splits exactly into spread capture (each fill's edge against the mid at fill time), inventory
PnL (the held position marked through later mid moves) and fees. Fills cluster in time, so
uncertainty comes from resampling whole blocks of time rather than individual fills.
"""

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from plom import runner
from plom.mm import Config, MarketMaker

BLOCK_S = 300
BOOTSTRAP_SAMPLES = 2000


@dataclass
class Tracker:
    """Samples PnL at fixed block boundaries while a market maker runs."""

    block_s: float = BLOCK_S
    start_ms: int | None = None
    end_ms: int | None = None
    block_pnls: list[float] = field(default_factory=list)
    _marked_pnl: float = 0.0

    def observe(self, mm: MarketMaker) -> None:
        if mm.mid is None:
            return
        if self.start_ms is None:
            self.start_ms = mm.now_ms
        self.end_ms = mm.now_ms
        block_ms = self.block_s * 1000
        while mm.now_ms >= self.start_ms + (len(self.block_pnls) + 1) * block_ms:
            self.block_pnls.append(mm.pnl - self._marked_pnl)
            self._marked_pnl = mm.pnl

    def block_of(self, time_ms: int) -> int:
        return int((time_ms - (self.start_ms or 0)) // (self.block_s * 1000))

    @property
    def hours(self) -> float:
        return ((self.end_ms or 0) - (self.start_ms or 0)) / 3_600_000


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"


@dataclass(frozen=True)
class HorizonStats:
    horizon_ms: int
    fills: int
    mean_bps: Interval | None
    """Size-weighted markout, with a block-bootstrap interval."""
    realized_usd: float
    """Spread capture that survived to this horizon: sum of markout x notional."""


@dataclass(frozen=True)
class Evaluation:
    label: str
    hours: float
    fills: int
    volume: float
    fees: float
    pnl: float
    spread_capture: float
    inventory_pnl: float
    position: float
    edge_bps: float
    pnl_per_hour: Interval | None
    horizons: list[HorizonStats]
    block_pnls: list[float]
    tx_sent: int
    tx_skipped: int


def evaluate(mm: MarketMaker, tracker: Tracker, label: str = "") -> Evaluation:
    blocks_per_hour = 3600 / tracker.block_s
    pnl_per_hour = _bootstrap_mean(tracker.block_pnls)
    if pnl_per_hour is not None:
        pnl_per_hour = Interval(*(value * blocks_per_hour for value in (pnl_per_hour.mean, pnl_per_hour.low, pnl_per_hour.high)))
    notional = sum(f.price * f.size for f in mm.fills)
    edge_usd = sum(_edge_usd(f.side, f.mid, f.price, f.size) for f in mm.fills)
    horizons = []
    for horizon in mm.config.markout_horizons_ms:
        markouts = [m for m in mm.markouts if m.horizon_ms == horizon]
        by_block: dict[int, list[tuple[float, float]]] = {}
        for m in markouts:
            weight = m.fill.price * m.fill.size
            by_block.setdefault(tracker.block_of(m.fill.time_ms), []).append((m.bps, weight))
        horizons.append(HorizonStats(
            horizon,
            len(markouts),
            _block_bootstrap(list(by_block.values()), _weighted_mean),
            sum(bps * weight for block in by_block.values() for bps, weight in block) / 10_000,
        ))
    return Evaluation(
        label=label,
        hours=tracker.hours,
        fills=len(mm.fills),
        volume=notional,
        fees=mm.fees,
        pnl=mm.pnl,
        spread_capture=mm.spread_capture,
        inventory_pnl=mm.inventory_pnl,
        position=mm.position,
        edge_bps=edge_usd / notional * 10_000 if notional else 0.0,
        pnl_per_hour=pnl_per_hour,
        horizons=horizons,
        block_pnls=tracker.block_pnls,
        tx_sent=mm.tx_sent,
        tx_skipped=mm.tx_skipped,
    )


def replay(
    path: Path,
    venue: str,
    config: Config,
    label: str = "",
    block_s: float = BLOCK_S,
    reference: str | None = None,
) -> Evaluation:
    """Run a market maker over one venue's events in a recording, optionally with a reference venue, and evaluate it."""
    mm = MarketMaker(config)
    tracker = Tracker(block_s)
    dispatcher = runner.Dispatcher(mm)
    for is_reference, recv_ms, event in runner.replay(path, venue, reference):
        dispatcher.feed(is_reference, recv_ms, event)
        tracker.observe(mm)
    return evaluate(mm, tracker, label)


def paired_difference(variant: Evaluation, base: Evaluation) -> Interval | None:
    """PnL per hour of variant minus base, pairing the same blocks of market time."""
    pairs = list(zip(variant.block_pnls, base.block_pnls))
    blocks_per_hour = len(pairs) / variant.hours if variant.hours else 0.0
    difference = _bootstrap_mean([v - b for v, b in pairs])
    if difference is None:
        return None
    return Interval(*(x * blocks_per_hour for x in (difference.mean, difference.low, difference.high)))


def format_report(e: Evaluation) -> str:
    lines = [
        "",
        f"--- {e.label or 'summary'} ---",
        f"hours          {e.hours:.2f}",
        f"fills          {e.fills:,}   volume ${e.volume:,.2f}   position {e.position:+.5f}",
        f"pnl            ${e.pnl:+,.4f}  = spread capture ${e.spread_capture:+,.4f}"
        f" + inventory ${e.inventory_pnl:+,.4f} - fees ${e.fees:,.4f}",
        f"pnl per hour   {e.pnl_per_hour or 'n/a (needs 2+ blocks)'}  (95% block bootstrap)",
        f"edge at fill   {e.edge_bps:+.2f} bps",
        f"transactions   {e.tx_sent:,} sent, {e.tx_skipped:,} skipped by the rate limit",
        "",
        "horizon   fills   markout bps [95%]              realized $",
    ]
    for h in e.horizons:
        interval = str(h.mean_bps) if h.mean_bps else "n/a"
        lines.append(f"{_horizon(h.horizon_ms):>7}  {h.fills:>6}   {interval:<30} {h.realized_usd:+.4f}")
    lines.append("(markout: how far the mid moved in our favour after the fill; edge minus markout is adverse selection)")
    return "\n".join(lines)


def _edge_usd(side: str, mid: float, price: float, size: float) -> float:
    return (mid - price) * size if side == "buy" else (price - mid) * size


def _weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    total = weight = 0.0
    for value, w in values:
        total += value * w
        weight += w
    return total / weight if weight else 0.0


def _bootstrap_mean(values: Sequence[float]) -> Interval | None:
    return _block_bootstrap([[(v, 1.0)] for v in values], _weighted_mean)


def _block_bootstrap(
    blocks: Sequence[Sequence[tuple[float, float]]],
    statistic: Callable[[Iterable[tuple[float, float]]], float],
) -> Interval | None:
    """Resample whole blocks with replacement and take the 2.5th and 97.5th percentiles of the statistic."""
    if len(blocks) < 2:
        return None
    rng = random.Random(0)
    samples = sorted(
        statistic(value for block in rng.choices(blocks, k=len(blocks)) for value in block)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    observed = statistic(value for block in blocks for value in block)
    return Interval(observed, samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1])


def _horizon(ms: int) -> str:
    return f"{ms / 1000:g}s" if ms < 60_000 else f"{ms / 60_000:g}m"
