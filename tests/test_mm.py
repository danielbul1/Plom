import math
from dataclasses import replace

import pytest

from plom.hyperliquid import Book, Trade
from plom.mm import Config, MarketMaker, half_spread_bps, quotes, skew_bps, tick_size

# One layer, mid 100, tick 0.01, 10 bps half spread: quotes at 99.90 / 100.10.
CONFIG = Config(
    order_size=1.0,
    max_position=5.0,
    layers=1,
    base_half_spread_bps=10,
    vol_multiplier=0,
    inventory_skew_bps=0,
    pressure_skew_bps=0,
    latency_ms=100,
    requote_interval_ms=0,
    fill_cooldown_ms=0,
    markout_horizons_ms=(1000,),
    pickoff_spread_mult=0,
    pickoff_size_cut=0,
    trend_enter_z=math.inf,
    requote_move_bps=0,
    jump_bps=math.inf,
)
# Three layers at 10 / 20 / 40 bps sized 1 / 2 / 4.
LAYERED = replace(CONFIG, layers=3, layer_spacing=2.0, size_growth=2.0, max_position=10.0)
BUY = ("buy", 0)
SELL = ("sell", 0)


def book(time_ms, bids=((99.99, 1.0),), asks=((100.01, 1.0),)):
    return Book(time_ms, list(bids), list(asks))


def live_mm(config=CONFIG, **book_kwargs):
    """A market maker whose first quotes are already live at t=100."""
    mm = MarketMaker(config)
    mm.on_book(book(0, **book_kwargs))
    mm.on_book(book(100, **book_kwargs))
    return mm


def test_quotes_are_symmetric_without_skew():
    assert quotes(100.0, 99.99, 100.01, 0, 10, 10, CONFIG) == ([99.90], [100.10])


def test_long_inventory_skews_down():
    config = replace(CONFIG, inventory_skew_bps=5)
    assert skew_bps(5.0, 0, config) == -5
    assert quotes(100.0, 99.99, 100.01, -5, 10, 10, config) == ([99.85], [100.05])


def test_bias_skews_towards_pressure():
    config = replace(CONFIG, pressure_skew_bps=5)
    assert skew_bps(0.0, 1, config) == 5
    # Reservation 100.05; +-10 bps of it is 99.94995 / 100.15005, rounded away from the mid.
    assert quotes(100.0, 99.99, 100.01, 5, 10, 10, config) == ([99.94], [100.16])


def test_sides_can_have_different_half_spreads():
    assert quotes(100.0, 99.99, 100.01, 0, 20, 10, CONFIG) == ([99.80], [100.10])


def test_quotes_never_cross_the_book():
    bids, _ = quotes(100.0, 99.99, 100.01, 50, 0, 0, CONFIG)
    assert bids == [100.00]


def test_zero_spread_quotes_never_cross_each_other():
    bids, asks = quotes(100.0, 99.99, 100.01, 0, 0, 0, CONFIG)
    assert bids[0] < asks[0]


def test_volatility_widens_the_spread():
    assert half_spread_bps(10.0, replace(CONFIG, vol_multiplier=2)) == 20
    assert half_spread_bps(1.0, replace(CONFIG, vol_multiplier=2)) == 10


def test_layers_move_outwards_geometrically():
    assert quotes(100.0, 99.99, 100.01, 0, 10, 10, LAYERED) == (
        [99.90, 99.80, 99.60],
        [100.10, 100.20, 100.40],
    )


def test_layers_never_share_a_price():
    bids, asks = quotes(100.0, 99.99, 100.01, 0, 0, 0, LAYERED)
    assert bids == [100.00, 99.99, 99.98]
    assert asks == [100.01, 100.02, 100.03]


def test_tick_size_keeps_five_significant_figures():
    assert tick_size(78_629.0) == 1.0
    assert tick_size(100.0) == pytest.approx(0.01)


def test_orders_go_live_only_after_latency():
    mm = MarketMaker(CONFIG)
    mm.on_book(book(0))
    mm.on_trade(Trade(50, "sell", 99.0, 1.0))
    assert mm.fills == []
    mm.on_book(book(100))
    assert mm.orders[BUY].price == 99.90


def test_trade_through_fills_whole_order():
    mm = live_mm()
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    assert mm.position == 1.0
    assert mm.cash == pytest.approx(-99.90)
    assert BUY not in mm.orders


def test_trade_before_order_went_live_is_ignored():
    mm = live_mm()
    mm.on_trade(Trade(90, "sell", 99.50, 5.0))
    assert mm.fills == []


def test_trade_at_our_price_eats_queue_first():
    mm = live_mm(bids=((99.99, 1.0), (99.90, 2.0)))
    assert mm.orders[BUY].queue_ahead == 2.0
    mm.on_trade(Trade(150, "sell", 99.90, 1.5))
    assert mm.fills == []
    mm.on_trade(Trade(160, "sell", 99.90, 0.7))
    assert mm.position == pytest.approx(0.2)
    assert mm.orders[BUY].size == pytest.approx(0.8)


def test_cancels_ahead_of_us_move_us_up_the_queue():
    mm = live_mm(bids=((99.99, 1.0), (99.90, 2.0)))
    mm.on_book(book(150, bids=((99.99, 1.0), (99.90, 0.5), (99.80, 1.0))))
    assert mm.orders[BUY].queue_ahead == 0.5


def test_book_crossing_our_price_fills():
    mm = live_mm()
    mm.on_book(book(150, bids=((99.80, 1.0),), asks=((99.85, 1.0),)))
    assert mm.position == 1.0


def test_aggressive_buy_fills_our_ask():
    mm = live_mm()
    mm.on_trade(Trade(150, "buy", 100.20, 0.01))
    assert mm.position == -1.0
    assert mm.cash == pytest.approx(100.10)


def test_maker_fee_is_charged():
    mm = live_mm(replace(CONFIG, maker_fee_bps=10))
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    assert mm.fees == pytest.approx(99.90 * 0.001)
    assert mm.cash == pytest.approx(-99.90 - 99.90 * 0.001)


def test_stops_buying_at_max_position():
    mm = live_mm()
    mm.position = CONFIG.max_position
    mm.on_book(book(200))
    assert mm.pending[BUY].size == 0
    mm.on_book(book(300))
    assert BUY not in mm.orders
    assert SELL in mm.orders


def test_markout_measures_mid_move_after_fill():
    mm = live_mm()
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    mm.on_book(book(1150, bids=((99.94, 1.0),), asks=((99.96, 1.0),)))
    [markout] = mm.markouts
    assert markout.horizon_ms == 1000
    assert markout.bps == pytest.approx((99.95 - 99.90) / 99.90 * 10_000)


def test_pnl_marks_position_at_mid():
    mm = live_mm()
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    assert mm.pnl == pytest.approx(100.0 - 99.90)


def test_bias_has_hysteresis():
    mm = MarketMaker(CONFIG)
    mm.on_book(book(0, bids=((99.99, 2.0),)))  # pressure 0.33
    assert mm.bias == 1
    mm.on_book(book(10, bids=((99.99, 1.5),)))  # pressure 0.2
    assert mm.bias == 1
    mm.on_book(book(20, bids=((99.99, 1.1),)))  # pressure 0.05
    assert mm.bias == 0


def test_layer_sizes_grow_outwards():
    mm = live_mm(LAYERED)
    assert [mm.orders[("buy", layer)].size for layer in range(3)] == [1.0, 2.0, 4.0]


def test_layers_stop_at_max_position():
    mm = live_mm(replace(LAYERED, max_position=5.0))
    assert [mm.orders[("buy", layer)].size for layer in range(3)] == [1.0, 2.0, 2.0]


def test_sweep_fills_every_layer_it_passes():
    mm = live_mm(LAYERED)
    mm.on_trade(Trade(150, "sell", 99.60, 0.01))
    # Layers 0 and 1 were traded through; layer 2 is at the trade price with an empty queue.
    assert [(f.layer, f.size) for f in mm.fills] == [(0, 1.0), (1, 2.0), (2, 0.01)]
    assert mm.orders[("buy", 2)].size == pytest.approx(3.99)


def test_fewer_layers_cancels_the_outer_ones():
    mm = live_mm(LAYERED)
    mm.config = replace(LAYERED, layers=1)
    mm.on_book(book(200))
    mm.on_book(book(300))
    assert set(mm.orders) == {BUY, SELL}


PICKOFF = replace(
    CONFIG, pickoff_spread_mult=1.0, pickoff_size_cut=0.5, pickoff_full_bps=2.0, pickoff_half_life_fills=1.0
)


def picked_off_bid(config=PICKOFF):
    """Our bid at 99.90 fills, then the mid drops to 99.50 a second later (-40 bps markout)."""
    mm = live_mm(config)
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    mm.on_book(book(1150, bids=((99.49, 1.0),), asks=((99.51, 1.0),)))
    return mm


def test_picked_off_side_scores_adverse_markouts():
    mm = picked_off_bid()
    assert mm.pickoff_bps("buy") == pytest.approx(0.5 * (99.90 - 99.50) / 99.90 * 10_000)
    assert mm.pickoff_score("buy") == 1.0
    assert mm.pickoff_score("sell") == 0.0


def test_picked_off_side_quotes_wider_and_smaller():
    mm = picked_off_bid()
    # Mid 99.50 (tick 0.001): the bid side doubles its 10 bps half spread, the ask side keeps it.
    assert mm.pending[BUY].price == 99.301
    assert mm.pending[BUY].size == pytest.approx(0.5)
    assert mm.pending[SELL].price == 99.60
    assert mm.pending[SELL].size == pytest.approx(1.0)


def test_pickoff_score_decays_without_new_fills():
    mm = picked_off_bid(replace(PICKOFF, pickoff_decay_s=1.0))
    before = mm.pickoff_bps("buy")
    mm.on_book(book(2150, bids=((99.49, 1.0),), asks=((99.51, 1.0),)))
    assert mm.pickoff_bps("buy") == pytest.approx(before / 2)


def test_favourable_markouts_do_not_widen():
    mm = live_mm(PICKOFF)
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    mm.on_book(book(1150, bids=((100.49, 1.0),), asks=((100.51, 1.0),)))
    assert mm.pickoff_bps("buy") < 0
    assert mm.pickoff_score("buy") == 0.0


TREND = replace(CONFIG, trend_window_s=1.0, trend_enter_z=2.0, trend_exit_z=1.0, trend_floor_bps=1.0)


def test_uptrend_pulls_asks_and_keeps_bids():
    mm = live_mm(TREND)
    mm.on_book(book(1100, bids=((100.09, 1.0),), asks=((100.11, 1.0),)))  # +10 bps in a second
    assert mm.trend == 1
    mm.on_book(book(1250, bids=((100.09, 1.0),), asks=((100.11, 1.0),)))
    assert BUY in mm.orders
    assert SELL not in mm.orders
    assert mm.pulled_ms["sell"] == 150


def test_downtrend_pulls_bids():
    mm = live_mm(TREND)
    mm.on_book(book(1100, bids=((99.89, 1.0),), asks=((99.91, 1.0),)))
    assert mm.trend == -1
    mm.on_book(book(1250, bids=((99.89, 1.0),), asks=((99.91, 1.0),)))
    assert BUY not in mm.orders
    assert SELL in mm.orders


def test_trend_ends_when_drift_fades_and_asks_return():
    mm = live_mm(TREND)
    flat = dict(bids=((100.09, 1.0),), asks=((100.11, 1.0),))
    mm.on_book(book(1100, **flat))
    mm.on_book(book(1250, **flat))
    mm.on_book(book(2300, **flat))
    assert mm.trend == 0
    assert SELL in mm.pending


CADENCE = replace(CONFIG, requote_move_bps=5, requote_ttl_ms=10_000)


def test_small_mid_moves_do_not_requote():
    mm = live_mm(CADENCE)
    mm.on_book(book(200, bids=((100.01, 1.0),), asks=((100.03, 1.0),)))  # +2 bps
    assert mm.pending == {}


def test_mid_move_past_threshold_requotes():
    mm = live_mm(CADENCE)
    mm.on_book(book(200, bids=((100.09, 1.0),), asks=((100.11, 1.0),)))  # +10 bps
    assert mm.pending[BUY].price == 99.99


def test_quotes_refresh_after_ttl_even_without_a_move():
    mm = live_mm(CADENCE)
    moved = dict(bids=((100.01, 1.0),), asks=((100.03, 1.0),))
    mm.on_book(book(9_000, **moved))
    assert mm.pending == {}
    mm.on_book(book(10_100, **moved))
    assert mm.pending[BUY].price == 99.91


def test_requote_interval_still_applies():
    mm = live_mm(replace(CONFIG, requote_interval_ms=1000))
    moved = dict(bids=((100.09, 1.0),), asks=((100.11, 1.0),))
    mm.on_book(book(500, **moved))
    assert mm.pending == {}
    mm.on_book(book(1000, **moved))
    assert BUY in mm.pending


def test_fill_requotes_without_a_move():
    mm = live_mm(CADENCE)
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    mm.on_book(book(200))
    assert mm.pending[BUY].price == 99.90


JUMP = replace(CONFIG, jump_bps=5, jump_hold_ms=3000, jump_spread_mult=2.0, jump_size_mult=0.5)


def test_book_jump_widens_and_shrinks_quotes_for_a_while():
    mm = live_mm(JUMP)
    jumped = dict(bids=((100.09, 1.0),), asks=((100.11, 1.0),))
    mm.on_book(book(200, **jumped))  # +10 bps in one step
    assert mm.is_jumping
    assert mm.pending[BUY].price == 99.89
    assert mm.pending[BUY].size == pytest.approx(0.5)
    mm.on_book(book(3300, **jumped))
    assert not mm.is_jumping
    assert mm.pending[BUY].price == 99.99
    assert mm.pending[BUY].size == pytest.approx(1.0)


def test_sweep_pulls_the_swept_side_until_the_next_book():
    mm = live_mm(replace(LAYERED, jump_bps=5, jump_spread_mult=2.0, jump_size_mult=0.5))
    mm.on_trade(Trade(150, "sell", 99.70, 0.01))  # 30 bps through the mid
    assert [f.layer for f in mm.fills] == [0, 1]
    assert mm.swept["buy"]
    assert mm.pending[("buy", 2)].size == 0
    assert mm.pending[SELL].price == 100.20
    assert mm.pending[SELL].size == pytest.approx(0.5)
    mm.on_book(book(400))
    assert not mm.swept["buy"]
    assert mm.pending[BUY].price == 99.80


def test_old_trades_do_not_count_as_jumps():
    mm = live_mm(JUMP)
    mm.on_book(book(3000))
    mm.on_trade(Trade(1000, "sell", 99.00, 0.01))
    assert not mm.is_jumping
    assert not mm.swept["buy"]
