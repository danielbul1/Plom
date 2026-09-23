import math
from dataclasses import replace

import pytest

from plom.market import Book
from plom.mm import MarketMaker
from plom.runner import Dispatcher
from test_mm import CONFIG, book

REFERENCE = replace(CONFIG, reference_weight=0.5, reference_stale_ms=1000, reference_jump_bps=2.0, reference_jump_window_ms=250)


def ref(time_ms, mid, half=0.005):
    return Book(time_ms, [(mid - half, 1.0)], [(mid + half, 1.0)])


def with_basis(config=REFERENCE):
    """Venue mid 100, reference mid 99: a learned basis of log(100/99)."""
    mm = MarketMaker(config)
    mm.on_reference(ref(0, 99.0))
    mm.on_book(book(10))
    return mm


def test_basis_is_learned_from_venue_minus_reference():
    mm = with_basis()
    assert mm.basis == pytest.approx(math.log(100 / 99))
    assert mm.fair == pytest.approx(100.0)


def test_fair_moves_part_way_to_the_basis_adjusted_reference():
    mm = with_basis()
    mm.on_reference(ref(300, 99.0099))  # +1 bp: the adjusted reference says 100.01.
    assert mm.fair == pytest.approx(100.005, abs=1e-6)


def test_stale_reference_is_ignored():
    mm = with_basis()
    mm.on_reference(ref(300, 99.0099))
    mm.on_book(book(2000))
    assert mm.fair == pytest.approx(100.0)


def test_reference_jump_pulls_the_side_it_runs_towards():
    mm = with_basis()
    mm.on_reference(ref(100, 99.0297))  # +3 bps within the window.
    assert mm.swept["sell"] and not mm.swept["buy"]
    assert mm.is_jumping
    assert mm.reference_jumps == 1


def test_slow_reference_drift_is_not_a_jump():
    mm = with_basis()
    for i, t in enumerate(range(300, 3000, 300), start=1):
        mm.on_reference(ref(t, 99.0 * (1 + i * 0.5 / 10_000)))
    assert mm.reference_jumps == 0


def test_dispatcher_translates_reference_time_through_receive_time():
    mm = MarketMaker(REFERENCE)
    dispatcher = Dispatcher(mm)
    dispatcher.feed(True, 50.0, ref(9_999, 99.0))  # Before any venue book: no clock mapping yet.
    assert mm.reference_mid is None
    dispatcher.feed(False, 1150.0, book(1000))  # The venue's feed reaches us 150ms after its timestamps.
    dispatcher.feed(True, 1300.0, ref(9_999_999, 99.0))
    assert mm.reference_ms == 1150


@pytest.mark.parametrize("hold_ms, pulled_at_300", [(0, False), (500, True)])
def test_a_jump_can_hold_its_pull_past_the_next_venue_book(hold_ms, pulled_at_300):
    mm = with_basis(replace(REFERENCE, reference_jump_hold_ms=hold_ms))
    mm.on_reference(ref(100, 99.0297))  # +3 bps: pull the asks.
    assert mm.swept["sell"]
    mm.on_book(book(300))
    assert mm.swept["sell"] is pulled_at_300 and not mm.swept["buy"]
    mm.on_book(book(700))
    assert not mm.swept["sell"]


def test_bitunix_profiles_carry_its_fees_and_ticks():
    from plom.profiles import DEFAULT_PROFILE, PROFILES

    assert DEFAULT_PROFILE["bitunix"] == "bitunix"
    assert PROFILES["bitunix"].config["tick_size"] == 0.1 and PROFILES["bitunix-eth"].config["tick_size"] == 0.01
    assert PROFILES["bitunix"].config["maker_fee_bps"] == 2.0
