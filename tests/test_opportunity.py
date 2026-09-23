import numpy as np
import pytest

from plom import opportunity


def test_sharp_moves_are_found_once_per_gap():
    t = np.arange(0, 6000, 50, dtype=float)
    px = np.full(t.size, 100.0)
    px[t >= 1000] = 100.03  # +3bps at 1s...
    px[t >= 1100] = 100.06  # ...and more within the gap: still one event.
    px[t >= 4000] = 100.0   # -6bps at 4s: a second event.
    assert opportunity.sharp_moves(t, px, 2.0) == [(1000.0, 1), (4000.0, -1)]


def series(**columns):
    return {k: np.array(v) for k, v in columns.items()}


def test_outcome_counts_a_stale_fill_and_a_favorable_fill():
    # The composite jumps up at t=1000. The venue's ask at 100.0 is lifted at 1100, and it only
    # catches up to 100.05 by 3000. A seller hits the bid (100.0 - 0.01 = 99.99) at 1300 for more
    # than the 1.0 queued there.
    book = series(
        t=[0.0, 2500.0, 5000.0], bid=[99.99, 100.04, 100.04], bid_sz=[1.0, 1.0, 1.0],
        ask=[100.0, 100.05, 100.05], ask_sz=[1.0, 1.0, 1.0],
    )
    trades = series(t=[1100.0, 1300.0], px=[100.0, 99.99], sz=[0.5, 2.0], buy=[True, False])
    fast = opportunity._outcome([(1000.0, 1)], book, trades, latency_ms=50, window_ms=500, markout_ms=2000)
    assert fast.events == 1 and fast.stale_fill_rate == 0.0  # Cancelled by 1050, before the lift.
    assert fast.good_fill_rate == 1.0 and fast.good_edge_bps == pytest.approx((100.045 - 99.99) / 99.99 * 1e4)
    slow = opportunity._outcome([(1000.0, 1)], book, trades, latency_ms=150, window_ms=500, markout_ms=2000)
    assert slow.stale_fill_rate == 1.0 and slow.stale_loss_bps == pytest.approx(4.5, abs=0.01)
    stale, good = slow.per_event_bps(maker_fee_bps=2.0)
    assert stale == pytest.approx(-6.5, abs=0.01) and good == pytest.approx(slow.good_edge_bps - 2.0)
