import math
import random
from dataclasses import replace

import pytest

from plom.mm import MarketMaker
from plom.volatility import DAY_MS, GridVolatility, lee_mykland_threshold
from test_mm import CONFIG, book


def test_lee_mykland_threshold():
    # n = 86,400 one-second returns, 1% a day: sqrt(2 log n) = 4.768, C_n = 5.504, S_n = 0.263.
    assert lee_mykland_threshold(86_400, 0.01) == pytest.approx(6.713, abs=0.01)
    assert lee_mykland_threshold(86_400, 0.001) > lee_mykland_threshold(86_400, 0.01)
    assert lee_mykland_threshold(864_000, 0.01) > lee_mykland_threshold(86_400, 0.01)


def random_walk(volatility, samples, sigma_bps=1.0, jumps=(), seed=5):
    """One mid per second with sigma_bps normal returns, plus a jump of bps at each (second, bps).

    Returns the seconds at which a jump was reported."""
    rng = random.Random(seed)
    jumps = dict(jumps)
    mid, flagged = 100.0, []
    for i in range(samples):
        mid *= math.exp((rng.gauss(0, sigma_bps) + jumps.get(i, 0.0)) / 10_000)
        if volatility.on_mid(i * 1000, mid):
            flagged.append(i)
    return flagged


def test_bipower_variation_ignores_jumps_that_inflate_realized_variance():
    volatility = GridVolatility(sample_ms=1000, half_life_s=1e9)
    # Twenty 20 bp jumps in 20,000 seconds add 20 x 400 / 20,000 = 0.4 to a variance of 1 a second.
    random_walk(volatility, 20_000, jumps=[(i, 20.0) for i in range(500, 20_000, 1000)])
    assert volatility.bipower_bps == pytest.approx(1.0, rel=0.05)
    assert volatility.realized_bps == pytest.approx(math.sqrt(1.4), rel=0.05)
    assert volatility.jump_share == pytest.approx(0.4 / 1.4, abs=0.05)


def test_lee_mykland_flags_the_jump_and_nothing_else():
    volatility = GridVolatility(sample_ms=1000, alpha=0.01, local_half_life_samples=60)
    assert volatility.threshold == pytest.approx(lee_mykland_threshold(DAY_MS / 1000, 0.01))
    assert random_walk(volatility, 5000, jumps=[(3000, 15.0)]) == [3000]
    assert volatility.jumps == 1


def test_a_jump_is_caught_when_it_happens_and_once_per_sample():
    # Ten-second samples of a 1 bp a second walk: a 40 bp jump three seconds into a sample.
    volatility = GridVolatility(sample_ms=10_000, alpha=0.01)
    assert random_walk(volatility, 2000, jumps=[(1503, 40.0)]) == [1503]


def test_no_jumps_without_alpha():
    volatility = GridVolatility(sample_ms=1000, local_half_life_samples=60)
    assert random_walk(volatility, 5000, jumps=[(3000, 15.0)]) == []


def test_returns_over_uneven_intervals_are_scaled_to_the_sample():
    volatility = GridVolatility(sample_ms=1000, half_life_s=1e9)
    volatility.on_mid(0, 100.0)
    volatility.on_mid(4000, 100.0 * math.exp(2 / 10_000))  # 2 bps over four samples is 1 bp a sample.
    assert volatility.realized_bps == pytest.approx(1.0)


def test_a_long_gap_restarts_the_return_chain():
    volatility = GridVolatility(sample_ms=1000, max_gap_ms=5000)
    volatility.on_mid(0, 100.0)
    volatility.on_mid(60_000, 101.0)
    assert volatility.realized.mean == 0.0


def test_samples_longer_than_the_gap_limit_still_chain_while_mids_keep_arriving():
    volatility = GridVolatility(sample_ms=15_000, max_gap_ms=5000, half_life_s=1e9)
    for t in range(0, 30_001, 1000):
        volatility.on_mid(t, 100.0 * math.exp(t / 15_000 / 10_000))  # 1 bp per sample
    assert volatility.samples == 2
    assert volatility.realized_bps == pytest.approx(1 / math.sqrt(15))


RIPPLE = [dict(), dict(bids=((100.00, 1.0),), asks=((100.02, 1.0),))]  # mid 100.00 / 100.01: 1 bp returns


def rippling(config, seconds=300):
    mm = MarketMaker(config)
    for i in range(seconds):
        mm.on_book(book(i * 1000, **RIPPLE[i % 2]))
    return mm, seconds * 1000


def test_lee_mykland_jump_widens_quotes():
    mm, t = rippling(replace(CONFIG, vol_sample_ms=1000, jump_alpha=0.01))
    assert not mm.is_jumping
    mm.on_book(book(t, bids=((100.09, 1.0),), asks=((100.11, 1.0),)))  # About 10 local volatilities.
    assert mm.volatility.jumps == 1
    assert mm.is_jumping


def test_lee_mykland_is_off_by_default():
    mm, t = rippling(replace(CONFIG, vol_sample_ms=1000))
    mm.on_book(book(t, bids=((100.09, 1.0),), asks=((100.11, 1.0),)))
    assert mm.volatility.jumps == 0
    assert not mm.is_jumping


def test_bipower_volatility_can_drive_quoting():
    mm, _ = rippling(replace(CONFIG, vol_bipower=1, vol_sample_ms=1000))
    # Alternating 1 bp returns: bipower variance is pi/2 per second.
    assert mm.vol_bps == pytest.approx(math.sqrt(math.pi / 2), rel=0.01)
    assert rippling(CONFIG)[0].vol_bps != mm.vol_bps
