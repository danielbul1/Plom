import math
import random
from dataclasses import replace

import pytest

from plom.alpha import AlphaConfig, OnlineAlpha, order_flow_imbalance
from plom.market import Book, Trade
from plom.mm import MarketMaker
from test_mm import BUY, CONFIG, SELL, book


@pytest.mark.parametrize(
    ("previous", "current", "flow"),
    [
        ((100, 5, 101, 5), (100, 7, 101, 5), 2),  # More size at the same bid.
        ((100, 5, 101, 5), (100.5, 3, 101, 5), 3),  # A higher bid adds its whole size.
        ((100, 5, 101, 5), (99.5, 4, 101, 5), -5),  # The old bid level is gone.
        ((100, 5, 101, 5), (100, 5, 100.5, 2), -2),  # A lower ask is selling pressure.
        ((100, 5, 101, 5), (100, 5, 101, 1), 4),  # Size leaving the ask is buying pressure.
    ],
)
def test_order_flow_imbalance(previous, current, flow):
    assert order_flow_imbalance(previous, current) == pytest.approx(flow)


def top(time_ms, mid, size=1.0):
    return Book(time_ms, [(mid - 0.01, size)], [(mid + 0.01, size)])


def test_learns_a_linear_relation_out_of_sample():
    rng = random.Random(1)
    alpha = OnlineAlpha(AlphaConfig(horizon_ms=100, sample_ms=100, half_life_samples=1e9, ridge=1e-6, warmup_samples=200))
    mid = 100.0
    for i in range(3000):
        gap = rng.uniform(-2, 2)
        alpha.on_book(i * 100, top(i * 100, mid), mid, gap)
        mid *= math.exp((0.5 * gap + rng.gauss(0, 0.1)) / 10_000)
    assert alpha.weights[0] == pytest.approx(0.5, abs=0.05)
    assert alpha.r2 > 0.8


def test_no_forecast_before_warmup():
    alpha = OnlineAlpha(AlphaConfig(warmup_samples=10))
    for i in range(5):
        alpha.on_book(i * 100, top(i * 100, 100.0), 100.0, 1.0)
    assert alpha.prediction_bps == 0.0
    assert alpha.r2 is None


def test_trade_flow_imbalance_decays():
    alpha = OnlineAlpha(AlphaConfig(flow_half_life_ms=1000))
    alpha.on_trade(0, Trade(0, "buy", 100.0, 3.0))
    alpha.on_trade(0, Trade(0, "sell", 100.0, 1.0))
    assert alpha._signed_volume / alpha._volume == pytest.approx(0.5)
    alpha.on_trade(1000, Trade(1000, "sell", 100.0, 1.0))
    assert alpha._signed_volume == pytest.approx(0.0)


def forecasting(prediction_bps, **overrides):
    """A live market maker whose forecast is pinned, for testing how quotes use it."""
    mm = MarketMaker(replace(CONFIG, **overrides))
    mm.alpha.on_book = lambda *args: None
    mm.alpha.prediction_bps = prediction_bps
    mm.on_book(book(0))
    mm.on_book(book(100))
    return mm


def test_forecast_against_a_side_pulls_it():
    mm = forecasting(-15.0, alpha_pull_margin_bps=0.0)  # A 15 bp drop beats the bid's 10 bp half spread.
    assert BUY not in mm.orders
    assert SELL in mm.orders
    assert mm.alpha_pulls["buy"] > 0


def test_small_forecast_keeps_both_sides():
    mm = forecasting(-5.0, alpha_pull_margin_bps=0.0)
    assert BUY in mm.orders and SELL in mm.orders


def test_forecast_widens_the_side_it_moves_against():
    mm = forecasting(-5.0, alpha_widen=2.0)  # Bid half spread 10 + 2 x 5 = 20 bps.
    assert mm.orders[BUY].price == 99.80
    assert mm.orders[SELL].price == 100.10


def test_forecast_shifts_fair_value():
    mm = forecasting(10.0, alpha_weight=0.5)
    assert mm.fair == pytest.approx(100.0 * (1 + 5 / 10_000))
