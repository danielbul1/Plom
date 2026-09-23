"""Trade the study's events and see what survives fees, stops and choosing the rule without hindsight.

The rule fades a breakout through the estimated liquidation levels (the study found continuation
rarely follows): at the close of the bar that crossed them, take the other side with a taker order;
stop out beyond that bar's extreme, take profit back at its open, or leave after a holding time.
One position at a time; events while in a position are skipped.

Walk-forward: every parameter combination is scored on the years before `split_year` only, the best
is picked there, and the report shows how that same rule did on the years after, which it never saw.
"""

import itertools
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from plom.breakout.history import Bar
from plom.breakout.study import Event, by_day
from plom.evaluate import Interval

FEE_BPS = 11.0
"""A taker round trip on Hyperliquid (4.5bps each way) plus a basis point of slippage each way."""


@dataclass(frozen=True)
class Rule:
    sides: str = "longs"
    """Which breakouts to fade: "longs" (longs crossed: buy the flush), "shorts", or "both"."""
    oi_fell: bool = False
    """Only when open interest fell in the breakout bar, a sign positions really were closed."""
    skip_shocks: bool = False
    stop_bps: float = 30.0
    """Beyond the breakout bar's extreme."""
    hold_bars: int = 48
    take_profit: bool = True
    """Exit when the price gets back to the breakout bar's open."""


@dataclass(frozen=True)
class Trade:
    time_ms: int
    side: int
    """+1 bought, -1 sold."""
    entry: float
    exit: float
    reason: str
    pnl_bps: float
    """After fees."""


def simulate(bars: list[Bar], events: list[Event], rule: Rule, fee_bps: float = FEE_BPS) -> list[Trade]:
    times = [b.time_ms for b in bars]
    trades: list[Trade] = []
    free_from = 0
    for e in events:
        if not _wanted(e, rule):
            continue
        i = bisect_left(times, e.time_ms)
        if i < free_from or i + 1 >= len(bars):
            continue
        breakout = bars[i]
        side = -e.direction  # Fade: longs crossed on the way down, so buy.
        entry = breakout.close
        if side > 0:
            stop = breakout.low * (1 - rule.stop_bps / 1e4)
            target = breakout.open if rule.take_profit and breakout.open > entry else None
        else:
            stop = breakout.high * (1 + rule.stop_bps / 1e4)
            target = breakout.open if rule.take_profit and breakout.open < entry else None
        exit_price, reason, j = bars[min(i + rule.hold_bars, len(bars) - 1)].close, "time", min(i + rule.hold_bars, len(bars) - 1)
        for j in range(i + 1, min(i + rule.hold_bars, len(bars) - 1) + 1):
            b = bars[j]
            # Within a bar the order of high and low is unknown: assume the stop came first.
            if (side > 0 and b.low <= stop) or (side < 0 and b.high >= stop):
                exit_price, reason = stop, "stop"
                break
            if target is not None and ((side > 0 and b.high >= target) or (side < 0 and b.low <= target)):
                exit_price, reason = target, "target"
                break
        free_from = j + 1
        pnl = side * (exit_price / entry - 1) * 1e4 - fee_bps
        trades.append(Trade(breakout.time_ms + 300_000, side, entry, exit_price, reason, pnl))
    return trades


def _wanted(e: Event, rule: Rule) -> bool:
    if rule.sides == "longs" and e.direction > 0 or rule.sides == "shorts" and e.direction < 0:
        return False
    if rule.oi_fell and e.oi_change >= 0:
        return False
    return not (rule.skip_shocks and e.shock)


GRID = {
    "sides": ("longs", "shorts", "both"),
    "oi_fell": (False, True),
    "skip_shocks": (False, True),
    "stop_bps": (20.0, 50.0, 100.0),
    "hold_bars": (12, 48, 144),
    "take_profit": (True, False),
}
MIN_TRADES = 30


@dataclass
class Summary:
    trades: int
    hit_rate: float
    mean_bps: Interval | None
    total_bps: float
    max_drawdown_bps: float


def summarize(trades: list[Trade]) -> Summary:
    pnls = [t.pnl_bps for t in trades]
    peak = equity = drawdown = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    hits = sum(p > 0 for p in pnls)
    return Summary(len(pnls), hits / len(pnls) if pnls else 0.0, by_day([(t.time_ms, t.pnl_bps) for t in trades]), equity, drawdown)


@dataclass
class WalkForward:
    split_year: int
    rule: Rule
    train: Summary
    test: Summary
    test_by_year: dict[int, Summary]
    default: Summary
    """The plan's rule as first written, before any fitting, over all years."""


def walk_forward(bars: list[Bar], events: list[Event], split_year: int, fee_bps: float = FEE_BPS) -> WalkForward:
    split_ms = int(datetime(split_year, 1, 1, tzinfo=UTC).timestamp() * 1000)
    train_events = [e for e in events if e.time_ms < split_ms]
    test_events = [e for e in events if e.time_ms >= split_ms]
    best: tuple[float, Rule, Summary] | None = None
    for values in itertools.product(*GRID.values()):
        rule = Rule(**dict(zip(GRID, values)))
        s = summarize(simulate(bars, train_events, rule, fee_bps))
        if s.trades < MIN_TRADES or s.mean_bps is None:
            continue
        # Rank by the interval's lower end: a rule that is only lucky scores low.
        if best is None or s.mean_bps.low > best[0]:
            best = (s.mean_bps.low, rule, s)
    assert best is not None, "no rule traded enough in the training years"
    _, rule, train = best
    test_trades = simulate(bars, test_events, rule, fee_bps)
    per_year: dict[int, list[Trade]] = defaultdict(list)
    for t in test_trades:
        per_year[datetime.fromtimestamp(t.time_ms / 1000, UTC).year].append(t)
    return WalkForward(
        split_year, rule, train, summarize(test_trades), {y: summarize(ts) for y, ts in sorted(per_year.items())},
        summarize(simulate(bars, events, Rule(), fee_bps)),
    )


def format_report(coin: str, w: WalkForward, fee_bps: float = FEE_BPS) -> str:
    def line(name: str, s: Summary) -> str:
        return (f"  {name:24} trades {s.trades:>4}  hit {s.hit_rate:5.1%}  per trade {s.mean_bps or '-'} bps"
                f"  total {s.total_bps:+8.0f}  max drawdown {s.max_drawdown_bps:6.0f}")
    r = w.rule
    return "\n".join([
        f"{coin}: fading breakouts through the estimated liquidation levels, {fee_bps:g}bps fees per round trip",
        line("plan's rule, all years", w.default),
        f"  chosen on years before {w.split_year}: sides={r.sides} oi_fell={r.oi_fell} skip_shocks={r.skip_shocks} "
        f"stop={r.stop_bps:g}bps hold={r.hold_bars * 5}m take_profit={r.take_profit}",
        line(f"training (< {w.split_year})", w.train),
        line(f"test (>= {w.split_year}), unseen", w.test),
        *(line(f"  {y}", s) for y, s in w.test_by_year.items()),
    ])
