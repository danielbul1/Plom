import pytest

from plom.pressure import pressure


def test_balanced_book_has_zero_pressure():
    bids = [(99.0, 1.0), (98.0, 2.0)]
    asks = [(101.0, 1.0), (102.0, 2.0)]
    assert pressure(bids, asks).value == pytest.approx(0.0)


def test_only_bid_depth_gives_positive_pressure():
    reading = pressure([(99.0, 5.0)], [(101.0, 0.0)])
    assert reading.value == pytest.approx(1.0)
    assert reading.mid == pytest.approx(100.0)


def test_heavier_ask_gives_negative_pressure():
    assert pressure([(99.0, 1.0)], [(101.0, 3.0)]).value < 0


def test_closer_levels_weigh_more():
    # Same total size per side, but the bid's size sits nearer the mid.
    bids = [(99.9, 10.0), (90.0, 1.0)]
    asks = [(100.1, 1.0), (110.0, 10.0)]
    assert pressure(bids, asks, half_life_bps=10).value > 0


def test_level_at_half_life_counts_half():
    # mid 100, both bests 10 bps away; the extra bid level is 10 bps further out.
    reading = pressure([(99.9, 1.0), (99.8, 2.0)], [(100.1, 1.0)], half_life_bps=10)
    assert reading.bid_weight == pytest.approx(0.5 + 2.0 * 0.25)
    assert reading.ask_weight == pytest.approx(0.5)


def test_only_nearest_depth_levels_count():
    bids = [(99.0, 1.0)] + [(90.0 - i, 1000.0) for i in range(5)]
    asks = [(101.0, 1.0)]
    assert pressure(bids, asks, depth=1).value == pytest.approx(0.0)


def test_unsorted_levels_are_ordered_best_first():
    reading = pressure([(98.0, 1.0), (99.0, 1.0)], [(102.0, 1.0), (101.0, 1.0)], depth=1)
    assert reading.mid == pytest.approx(100.0)


def test_empty_side_returns_none():
    assert pressure([], [(101.0, 1.0)]) is None


@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"half_life_bps": 0}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        pressure([(99.0, 1.0)], [(101.0, 1.0)], **kwargs)
