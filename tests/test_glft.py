import math
import random
from dataclasses import replace

import pytest

from plom.glft import FillIntensity, Glft, glft
from plom.market import Trade
from plom.mm import MarketMaker, half_spread_bps, quotes, skew_bps
from test_mm import CONFIG, LAYERED, book, live_mm


def test_half_spread_tends_to_one_over_k_without_risk_aversion_or_volatility():
    model = glft(0.0, a=1.0, k=0.5, gamma=1e-9)
    assert model.half_spread_bps == pytest.approx(2.0)
    assert model.skew_bps == 0.0


def test_matches_the_closed_form():
    sigma, a, k, gamma = 3.0, 2.0, 0.8, 0.1
    c1 = math.log(1 + gamma / k) / gamma
    c2 = math.sqrt(gamma / (2 * a * k) * (1 + gamma / k) ** (k / gamma + 1))
    model = glft(sigma, a, k, gamma)
    assert model.half_spread_bps == pytest.approx(c1 + sigma * c2 / 2)
    assert model.skew_bps == pytest.approx(sigma * c2)


def test_a_maker_fee_is_solved_as_a_smaller_a_one_fee_further_out():
    sigma, a, k, gamma, fee = 3.0, 2.0, 0.8, 0.1, 1.5
    net = glft(sigma, a * math.exp(-k * fee), k, gamma)
    model = glft(sigma, a, k, gamma, fee)
    assert model.half_spread_bps == pytest.approx(fee + net.half_spread_bps)
    assert model.skew_bps == pytest.approx(net.skew_bps)
    assert model.skew_bps > glft(sigma, a, k, gamma).skew_bps


def test_more_risk_aversion_skews_harder_and_volatility_widens():
    assert glft(3.0, 2.0, 0.8, 0.5).skew_bps > glft(3.0, 2.0, 0.8, 0.1).skew_bps
    assert glft(6.0, 2.0, 0.8, 0.1).half_spread_bps > glft(3.0, 2.0, 0.8, 0.1).half_spread_bps


def test_calibration_recovers_a_and_k():
    # Each 100ms, each side is reached with probability 0.5 at an exponential depth with k = 2/bp,
    # so λ(δ) = 0.5 exp(-2δ) per 100ms: A = 5 per second.
    rng = random.Random(3)
    intensity = FillIntensity(sample_ms=100, half_life_s=1e9, step_bps=0.25, buckets=40, min_hits=30)
    for i in range(40_000):
        t = i * 100
        intensity.on_book(t, 100.0)
        for side in ("buy", "sell"):
            if rng.random() < 0.5:
                depth = rng.expovariate(2.0)
                sign = 1 if side == "buy" else -1
                intensity.on_trade(Trade(t + 50, side, 100.0 * (1 + sign * depth / 10_000), 1.0))
    assert intensity.k == pytest.approx(2.0, rel=0.1)
    assert intensity.a == pytest.approx(5.0, rel=0.15)


def test_long_gaps_are_not_exposure():
    intensity = FillIntensity(sample_ms=100, max_gap_ms=1000)
    intensity.on_book(0, 100.0)
    intensity.on_book(100, 100.0)
    intensity.on_book(60_100, 100.0)
    assert intensity._exposure_s == pytest.approx(0.1)


def test_glft_skews_by_lots_of_order_size():
    config = replace(CONFIG, order_size=0.5, pressure_skew_bps=1.0)
    model = Glft(half_spread_bps=4.0, skew_bps=2.0)
    assert skew_bps(1.0, 1, config, model) == pytest.approx(-4.0 + 1.0)
    assert half_spread_bps(100.0, replace(config, base_half_spread_bps=1.0), model) == 4.0
    assert half_spread_bps(100.0, config, model) == 10.0  # base_half_spread_bps is a floor


def test_older_positions_skew_harder():
    config = replace(CONFIG, inventory_skew_bps=5, pressure_skew_bps=1.0)
    assert skew_bps(5.0, 1, config, age_mult=3.0) == pytest.approx(-15.0 + 1.0)


def test_layers_can_step_out_by_a_fixed_distance():
    assert quotes(100.0, 99.99, 100.01, 0, 10, 10, LAYERED, layer_step_bps=10) == (
        [99.90, 99.80, 99.70],
        [100.10, 100.20, 100.30],
    )


def test_calibrated_market_maker_quotes_the_glft_spread_with_glft_layers():
    config = replace(LAYERED, base_half_spread_bps=0, glft_gamma=1e-6, glft_layer_spacing=1.0)
    mm = MarketMaker(config)
    mm.intensity.a, mm.intensity.k = 1.0, 0.1  # Half spread ~1/k = 10 bps in a flat market.
    mm.on_book(book(0))
    mm.on_book(book(100))
    assert [mm.orders[("buy", layer)].price for layer in range(3)] == [99.90, 99.80, 99.70]


def test_uncalibrated_or_off_glft_falls_back():
    assert MarketMaker(replace(CONFIG, glft_gamma=0.1)).glft is None
    mm = MarketMaker(CONFIG)
    mm.intensity.a, mm.intensity.k = 1.0, 0.1
    assert mm.glft is None


def test_position_age_runs_from_opening_until_flat():
    mm = live_mm()
    mm.on_trade(Trade(150, "sell", 99.50, 5.0))  # Bid fills: long 1.
    mm.on_book(book(2150))
    assert mm.position_age_s == pytest.approx(2.0)
    mm.on_trade(Trade(2200, "buy", 100.50, 5.0))  # Ask fills: flat.
    assert mm.position == 0
    assert mm.position_age_s == 0.0


FLATTEN = replace(CONFIG, flatten_age_s=1.0, taker_fee_bps=10)


def test_old_position_crosses_the_spread_after_latency():
    mm = live_mm(FLATTEN)
    mm.on_trade(Trade(150, "sell", 99.50, 5.0))
    mm.on_book(book(1150))  # One second old: the flatten lands at 1250.
    assert mm.position == 1.0
    mm.on_book(book(1250, bids=((99.99, 0.4), (99.98, 1.0))))
    assert [(f.price, f.size) for f in mm.flattens] == [(99.99, pytest.approx(0.4)), (99.98, pytest.approx(0.6))]
    assert mm.position == pytest.approx(0.0)
    assert mm.fees == pytest.approx((99.99 * 0.4 + 99.98 * 0.6) * 0.001)
    assert len(mm.fills) == 1
    assert mm.pnl == pytest.approx(mm.spread_capture + mm.inventory_pnl - mm.fees)


def test_flatten_never_overshoots_a_position_that_shrank_meanwhile():
    mm = live_mm(FLATTEN)
    mm.on_trade(Trade(150, "sell", 99.50, 5.0))
    mm.on_book(book(1150))
    mm.position = 0.25  # As if the ask had filled most of it while the flatten was in flight.
    mm.on_book(book(1250, bids=((99.99, 5.0),)))
    assert [f.size for f in mm.flattens] == [pytest.approx(0.25)]
    assert mm.position == pytest.approx(0.0)


def test_flatten_waits_for_its_cooldown():
    mm = live_mm(replace(FLATTEN, flatten_fraction=0.5, flatten_cooldown_ms=5000))
    mm.on_trade(Trade(150, "sell", 99.50, 5.0))
    mm.on_book(book(1150))
    mm.on_book(book(1250))
    mm.on_book(book(3000))
    assert len(mm.flattens) == 1
    assert mm.position == pytest.approx(0.5)
    mm.on_book(book(6150))
    mm.on_book(book(6250))
    assert len(mm.flattens) == 2
    assert mm.position == pytest.approx(0.25)
