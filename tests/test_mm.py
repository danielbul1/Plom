import pytest

from plom.hyperliquid import Book, Trade
from plom.mm import Config, MarketMaker, quotes, tick_size

# Mid 100, tick 0.01, 10 bps half spread: quotes at 99.90 / 100.10.
CONFIG = Config(
    order_size=1.0,
    max_position=5.0,
    base_half_spread_bps=10,
    vol_multiplier=0,
    inventory_skew_bps=0,
    pressure_skew_bps=0,
    latency_ms=100,
    requote_interval_ms=0,
    fill_cooldown_ms=0,
    markout_horizons_ms=(1000,),
)


def book(time_ms, bids=((99.99, 1.0),), asks=((100.01, 1.0),)):
    return Book(time_ms, list(bids), list(asks))


def live_mm(config=CONFIG, **book_kwargs):
    """A market maker whose first quotes are already live at t=100."""
    mm = MarketMaker(config)
    mm.on_book(book(0, **book_kwargs))
    mm.on_book(book(100, **book_kwargs))
    return mm


def test_quotes_are_symmetric_without_skew():
    assert quotes(100.0, 99.99, 100.01, 0.0, 0.0, 0, CONFIG) == (99.90, 100.10)


def test_long_inventory_shifts_quotes_down():
    config = Config(**{**CONFIG.__dict__, "inventory_skew_bps": 5})
    bid, ask = quotes(100.0, 99.99, 100.01, 0.0, 5.0, 0, config)
    assert (bid, ask) == (99.85, 100.05)


def test_bias_shifts_quotes_towards_pressure():
    config = Config(**{**CONFIG.__dict__, "pressure_skew_bps": 5})
    # Reservation 100.05; +-10 bps of it is 99.94995 / 100.15005, rounded away from the mid.
    assert quotes(100.0, 99.99, 100.01, 0.0, 0.0, 1, config) == (99.94, 100.16)


def test_quotes_never_cross_the_book():
    config = Config(**{**CONFIG.__dict__, "base_half_spread_bps": 0, "pressure_skew_bps": 50})
    bid, _ = quotes(100.0, 99.99, 100.01, 0.0, 0.0, 1, config)
    assert bid == 100.00


def test_volatility_widens_the_spread():
    config = Config(**{**CONFIG.__dict__, "vol_multiplier": 2})
    assert quotes(100.0, 99.99, 100.01, 10.0, 0.0, 0, config) == (99.80, 100.20)


def test_tick_size_keeps_five_significant_figures():
    assert tick_size(78_629.0) == 1.0
    assert tick_size(100.0) == pytest.approx(0.01)


def test_orders_go_live_only_after_latency():
    mm = MarketMaker(CONFIG)
    mm.on_book(book(0))
    mm.on_trade(Trade(50, "sell", 99.0, 1.0))
    assert mm.fills == []
    mm.on_book(book(100))
    assert mm.orders["buy"].price == 99.90


def test_trade_through_fills_whole_order():
    mm = live_mm()
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    assert mm.position == 1.0
    assert mm.cash == pytest.approx(-99.90)
    assert mm.orders["buy"] is None


def test_trade_before_order_went_live_is_ignored():
    mm = live_mm()
    mm.on_trade(Trade(90, "sell", 99.50, 5.0))
    assert mm.fills == []


def test_trade_at_our_price_eats_queue_first():
    mm = live_mm(bids=((99.99, 1.0), (99.90, 2.0)))
    assert mm.orders["buy"].queue_ahead == 2.0
    mm.on_trade(Trade(150, "sell", 99.90, 1.5))
    assert mm.fills == []
    mm.on_trade(Trade(160, "sell", 99.90, 0.7))
    assert mm.position == pytest.approx(0.2)
    assert mm.orders["buy"].size == pytest.approx(0.8)


def test_cancels_ahead_of_us_move_us_up_the_queue():
    mm = live_mm(bids=((99.99, 1.0), (99.90, 2.0)))
    mm.on_book(book(150, bids=((99.99, 1.0), (99.90, 0.5), (99.80, 1.0))))
    assert mm.orders["buy"].queue_ahead == 0.5


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
    mm = live_mm(Config(**{**CONFIG.__dict__, "maker_fee_bps": 10}))
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    assert mm.fees == pytest.approx(99.90 * 0.001)
    assert mm.cash == pytest.approx(-99.90 - 99.90 * 0.001)


def test_stops_buying_at_max_position():
    mm = live_mm()
    mm.position = CONFIG.max_position
    mm.on_book(book(200))
    assert mm.pending["buy"].size == 0
    mm.on_book(book(300))
    assert mm.orders["buy"] is None
    assert mm.orders["sell"] is not None


def test_markout_measures_mid_move_after_fill():
    mm = live_mm()
    mm.on_trade(Trade(150, "sell", 99.50, 0.01))
    mm.on_book(book(1150, bids=((99.94, 1.0),), asks=((99.96, 1.0),)))
    assert mm.markouts[1000] == [pytest.approx((99.95 - 99.90) / 99.90 * 10_000)]


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
