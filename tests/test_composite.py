import gzip
import json
import math

import pytest

from plom import runner
from plom.composite import Composite
from plom.market import Book


def mid(price):
    return Book(0, [(price - 0.5, 1.0)], [(price + 0.5, 1.0)])


def price(book):
    return book.bids[0][0]


def test_waits_for_enough_fresh_venues():
    composite = Composite(("a", "b", "c"), min_venues=3)
    assert composite.on_book("a", 0, mid(100)) is None
    assert composite.on_book("b", 10, mid(101)) is None
    assert price(composite.on_book("c", 20, mid(102))) == pytest.approx(101)


def test_ignores_venues_outside_the_set_and_stale_ones():
    composite = Composite(("a", "b", "c"), stale_ms=1000, min_venues=2)
    assert composite.on_book("x", 0, mid(100)) is None
    composite.on_book("a", 0, mid(100))
    assert composite.on_book("b", 2000, mid(100)) is None  # a is stale by now.


def test_venue_levels_are_absorbed_by_the_basis():
    composite = Composite(("a", "b", "c"), min_venues=3)
    for venue, level in (("a", 100), ("b", 110), ("c", 90)):
        composite.on_book(venue, 0, mid(level))
    # A 1% move on one venue moves the median of adjusted mids by at most that venue's move...
    moved = price(composite.on_book("b", 10, mid(111.1)))
    assert moved == pytest.approx(100)  # ...and not at all while the others hold still.
    # When all three move 1%, so does the composite, despite their different levels.
    composite.on_book("a", 20, mid(101))
    assert price(composite.on_book("c", 30, mid(90.9))) == pytest.approx(101, rel=1e-4)


def test_a_venue_joining_does_not_jump_the_composite():
    composite = Composite(("a", "b", "c", "d"), min_venues=2)
    composite.on_book("a", 0, mid(100))
    before = price(composite.on_book("b", 10, mid(100.2)))
    after = price(composite.on_book("c", 20, mid(120)))  # Far away, but a new venue joins at the level.
    assert after == pytest.approx(before)


def test_basis_learns_towards_a_persistent_offset():
    composite = Composite(("a", "b", "c"), basis_half_life_s=1.0, min_venues=3)
    for venue in ("a", "b", "c"):
        composite.on_book(venue, 0, mid(100))
    for t in range(1, 11):
        composite.on_book("a", t * 1000 - 20, mid(100))
        composite.on_book("b", t * 1000 - 10, mid(100))
        composite.on_book("c", t * 1000, mid(101))
    assert composite.basis["c"] == pytest.approx(math.log(1.01), rel=0.01)


def test_router_feeds_the_composite_without_the_quoting_venue(tmp_path):
    path = tmp_path / "r.jsonl.gz"
    ticker = lambda p: {"data": {"e": "bookTicker", "b": str(p - 0.5), "B": "1", "a": str(p + 0.5), "A": "1", "T": 1}}
    lines = [
        {"venue": "aster", "recv_ms": 0.0, "msg": ticker(100)},
        {"venue": "okx", "recv_ms": 5.0, "msg": {"arg": {"channel": "bbo-tbt"}, "data": [{"bids": [["100", "1"]], "asks": [["101", "1"]], "ts": "5"}]}},
        {"venue": "blofin", "recv_ms": 10.0, "msg": {"arg": {"channel": "books5"}, "data": {"bids": [["99", "1"]], "asks": [["100", "1"]], "ts": "10"}}},
    ]
    with gzip.open(path, "wt") as out:
        for line in lines:
            out.write(json.dumps(line) + "\n")
    router = runner.Router("okx", "composite")
    assert "okx" not in router.composite.venues
    events = list(runner.replay(path, "okx", "composite"))
    assert [is_reference for is_reference, _, _ in events] == [False]  # Two composite venues: not enough.
    router.composite.min_venues = 2
    routed = [router.route(v["venue"], v["recv_ms"], v["msg"]) for v in lines]
    assert routed[0] == [] and routed[1][0][0] is False
    [(is_reference, recv_ms, book)] = routed[2]
    assert is_reference and recv_ms == 10.0 and price(book) == pytest.approx(math.sqrt(100 * 99.5))


def test_sharp_moves_track_how_a_lagging_venue_catches_up():
    import numpy as np

    from plom import leadlag

    steps = 400
    ref = np.zeros(steps)
    ref[100:] = 2e-4  # The reference jumps 2bps at step 100...
    mid = np.zeros(steps)
    mid[100 + 500 // leadlag.GRID_MS:] = 2e-4  # ...and the venue follows 500ms later.
    moves = leadlag.sharp_moves(ref, mid, np.zeros(steps), threshold_bps=1.5, warmup=0)
    assert moves.count == 1 and moves.reference_bps == pytest.approx(2.0)
    assert moves.gap_bps[0] == pytest.approx(2.0) and moves.venue_bps[0] == 0
    assert moves.gap_bps[500] == pytest.approx(0.0) and moves.venue_bps[1000] == pytest.approx(2.0)
