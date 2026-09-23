import io
import random
import zipfile
from datetime import date

import pytest

from plom.breakout import backtest, history, study
from plom.breakout.history import Bar
from plom.breakout.study import Event
from plom.hub import candles
from plom.hub.liquidations import LiquidationModel
from plom.hub.positioning import Liquidation, OpenInterest


def random_bars(n: int, seed: int = 1) -> list[Bar]:
    rng = random.Random(seed)
    bars, price, oi = [], 100.0, 1000.0
    for i in range(n):
        close = price * (1 + rng.gauss(0, 0.004))
        high, low = max(price, close) * (1 + abs(rng.gauss(0, 0.002))), min(price, close) * (1 - abs(rng.gauss(0, 0.002)))
        oi *= 1 + rng.gauss(0.0002, 0.003)
        bars.append(Bar(i * 300_000, price, high, low, close, rng.uniform(1, 10), rng.uniform(1, 10), oi))
        price = close
    return bars


def test_replayed_intervals_report_the_levels_they_cross():
    model = LiquidationModel()
    model.on_interval("binance", 0, 100.0, 0, 0, 99.0, 101.0, 100.0)
    model.on_interval("binance", 300_000, 110.0, 3.0, 1.0, 99.9, 100.1, 100.0)  # Opens 7.5 longs, 2.5 shorts.
    crossed = model.on_interval("binance", 600_000, 110.0, 0, 0, 98.0, 100.2, 99.0)  # Reaches 99.5 and 98.5.
    assert {(c.side, c.leverage) for c in crossed} == {("long", 100), ("long", 50)}
    assert sum(c.coins for c in crossed) == pytest.approx(7.5 * (0.15 + 0.20), rel=1e-2)  # Less 5 minutes of decay.
    assert model.within("long", 0, 1000) and not any(c.leverage in (50, 100) for c in model.within("long", 0, 1000))


def test_the_study_uses_no_future_bars():
    bars = random_bars(study.MIN_HISTORY_BARS + 1500)
    full = study.run("X", bars)
    cut = study.run("X", bars[:-300])
    last = bars[-300 - max(study.HORIZONS.values()) - 1].time_ms
    key = lambda s: [(e.time_ms, e.direction, e.share, e.forward_bps) for e in s.events if e.time_ms < last]
    assert key(full) == key(cut)
    assert full.events, "the random walk should cross some estimated levels"


def event(time_ms: int, direction: int) -> Event:
    return Event(time_ms, direction, 0.01, 50.0, -0.01, False, {})


def test_fading_a_breakdown_buys_and_takes_profit_at_its_open():
    bars = [
        Bar(0, 100, 100, 98, 98.5, 1, 1, 1),  # Broke down through longs.
        Bar(300_000, 98.5, 99.0, 98.2, 98.9, 1, 1, 1),
        Bar(600_000, 98.9, 100.2, 98.8, 100.1, 1, 1, 1),  # Back to the breakdown's open.
    ]
    [t] = backtest.simulate(bars, [event(0, -1)], backtest.Rule(stop_bps=30), fee_bps=10)
    assert (t.side, t.reason, t.exit) == (1, "target", 100)
    assert t.pnl_bps == pytest.approx((100 / 98.5 - 1) * 1e4 - 10)


def test_the_stop_is_assumed_to_come_first_within_a_bar():
    bars = [Bar(0, 100, 100, 98, 98.5, 1, 1, 1), Bar(300_000, 98.5, 100.5, 97.0, 99, 1, 1, 1)]
    [t] = backtest.simulate(bars, [event(0, -1)], backtest.Rule(stop_bps=30), fee_bps=0)
    assert t.reason == "stop" and t.exit == pytest.approx(98 * 0.997)


def test_the_rule_filters_sides_and_skips_events_while_in_a_position():
    bars = [Bar(i * 300_000, 100, 100.1, 99.9, 100, 1, 1, 1) for i in range(20)]
    events = [event(0, -1), event(300_000, -1), event(600_000, 1)]
    assert len(backtest.simulate(bars, events, backtest.Rule(sides="longs", hold_bars=5))) == 1
    assert [t.side for t in backtest.simulate(bars, events, backtest.Rule(sides="shorts"))] == [-1]


def test_history_joins_klines_with_the_open_interest_at_their_end(tmp_path):
    def archive(path: str, text: str) -> None:
        local = tmp_path / path
        local.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("x.csv", text)
        local.write_bytes(buffer.getvalue())

    archive("monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-08.zip",
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume\n"
            "1722470400000,100,101,99,100.5,10,0,0,0,6\n1722470700000,100.5,102,100,101,8,0,0,0,2\n")
    archive("daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-08-01.zip",
            "create_time,symbol,sum_open_interest\n2024-08-01 00:05:00,BTCUSDT,500\n")
    [bar] = history.load("BTC", date(2024, 8, 1), date(2024, 8, 1), tmp_path)
    assert (bar.time_ms, bar.buy, bar.sell, bar.oi) == (1722470400000, 6.0, 4.0, 500.0)


def test_store_keeps_liquidations_and_open_interest(tmp_path):
    store = candles.Store(tmp_path / "c.sqlite")
    store.write_positioning("BTC-USD", [Liquidation("bybit", 5, "long", 99.0, 2.0)], [OpenInterest("okx", 6, 1000.0)])
    assert store.read_liquidations("BTC-USD", 0, 10) == [("bybit", 5, "long", 99.0, 2.0)]
    assert store.read_open_interest("BTC-USD", 0, 10) == [("okx", 6, 1000.0)]
