import pytest

from plom.evaluate import Interval, Tracker, _block_bootstrap, _bootstrap_mean, _weighted_mean, evaluate, paired_difference
from plom.market import Book, Trade
from plom.mm import MarketMaker
from test_mm import CONFIG, book


def run(events, config=CONFIG, block_s=1):
    mm = MarketMaker(config)
    tracker = Tracker(block_s)
    for event in events:
        (mm.on_book if isinstance(event, Book) else mm.on_trade)(event)
        tracker.observe(mm)
    return mm, tracker


def test_pnl_splits_into_spread_capture_inventory_and_fees():
    from dataclasses import replace

    mm, _ = run(
        [
            book(0),
            book(100),
            Trade(150, "sell", 99.50, 5.0),  # Buy 1 at 99.90 with the mid at 100: +0.10 edge.
            book(1150, bids=((99.49, 1.0),), asks=((99.51, 1.0),)),  # Mid drops 0.50 holding 1.
        ],
        config=replace(CONFIG, maker_fee_bps=10),
    )
    assert mm.spread_capture == pytest.approx(0.10)
    assert mm.inventory_pnl == pytest.approx(-0.50)
    assert mm.pnl == pytest.approx(mm.spread_capture + mm.inventory_pnl - mm.fees)


def test_tracker_samples_pnl_once_per_block():
    tracker = Tracker(block_s=1)
    mm = MarketMaker(CONFIG)
    for t, pnl in [(0, 0.0), (400, 1.0), (1000, 2.0), (1500, 2.5), (3200, 5.0)]:
        mm.now_ms, mm.mid, mm.cash = t, 100.0, pnl
        tracker.observe(mm)
    # Block 1 ends at 1000 (pnl 2), blocks 2 and 3 both close at the 3200 update (pnl 5).
    assert tracker.block_pnls == [2.0, 3.0, 0.0]
    assert tracker.hours == pytest.approx(3.2 / 3600)


def test_bootstrap_interval_brackets_the_mean_and_is_deterministic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    interval = _bootstrap_mean(values)
    assert interval.mean == pytest.approx(3.5)
    assert interval.low < 3.5 < interval.high
    assert _bootstrap_mean(values) == interval


def test_bootstrap_needs_two_blocks():
    assert _bootstrap_mean([1.0]) is None


def test_block_bootstrap_keeps_blocks_together():
    # Two identical blocks: every resample gives the same weighted mean.
    blocks = [[(1.0, 1.0), (3.0, 3.0)], [(1.0, 1.0), (3.0, 3.0)]]
    assert _block_bootstrap(blocks, _weighted_mean) == Interval(2.5, 2.5, 2.5)


def test_evaluation_reports_markouts_per_horizon():
    mm, tracker = run([
        book(0), book(100), Trade(150, "sell", 99.50, 5.0),
        book(1150, bids=((99.94, 1.0),), asks=((99.96, 1.0),)),
        book(2150, bids=((99.94, 1.0),), asks=((99.96, 1.0),)),
    ])
    evaluation = evaluate(mm, tracker)
    [horizon] = evaluation.horizons
    assert horizon.horizon_ms == 1000 and horizon.fills == 1
    assert horizon.realized_usd == pytest.approx(0.05)
    assert evaluation.edge_bps == pytest.approx(0.10 / 99.90 * 10_000)


def test_paired_difference_compares_the_same_blocks():
    mm, tracker = run([book(0), book(3600_000)], block_s=900)
    base = evaluate(mm, tracker)
    better = evaluate(mm, tracker)
    object.__setattr__(better, "block_pnls", [p + 1.0 for p in base.block_pnls])
    difference = paired_difference(better, base)
    assert difference.mean == pytest.approx(4.0)  # +1 per 15-minute block is +4 an hour.
