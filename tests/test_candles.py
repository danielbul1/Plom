import math

import pytest
from fastapi.testclient import TestClient

from plom.hub import candles
from plom.hub.api import create_app
from plom.hub.candles import Candle, LiveCandles, Store, merge
from plom.hub.runner import Hub

MINUTE = 60_000


def backfilled(open_ms, price, source="backfill:coinbase", volume=1.0):
    return Candle(open_ms, price, price + 1, price - 1, price, volume, None, None, 3, source)


def test_live_candles_roll_over_and_split_taker_volume():
    live = LiveCandles(started_ms=MINUTE)
    live.on_trade(MINUTE + 1, 100.0, 1.0, "buy")
    live.on_trade(MINUTE + 2, 102.0, 2.0, "sell")
    live.on_trade(MINUTE + 3, 99.0, 0.5, "buy")
    live.on_trade(2 * MINUTE, 101.0, 1.0, "sell")
    [first] = live.finished["1m"]
    assert (first.open, first.high, first.low, first.close) == (100.0, 102.0, 99.0, 99.0)
    assert (first.volume, first.buy_volume, first.sell_volume, first.trade_count) == (3.5, 1.5, 2.0, 3)
    assert not first.partial
    assert live.current["1m"].open_ms == 2 * MINUTE
    assert live.current["1h"].volume == 4.5 and live.current["1h"].partial  # Its hour began before we started.


def test_late_trades_for_a_closed_candle_are_dropped():
    live = LiveCandles(started_ms=0)
    live.on_trade(2 * MINUTE, 100.0, 1.0, "buy")
    live.on_trade(MINUTE + 5, 90.0, 1.0, "buy")
    assert live.current["1m"].low == 100.0 and live.finished["1m"] == []


def test_backfill_fills_gaps_and_partials_but_never_complete_live_candles(tmp_path):
    store = Store(tmp_path / "c.sqlite")
    live = LiveCandles(started_ms=MINUTE + 30_000)
    live.on_trade(MINUTE + 40_000, 100.0, 1.0, "buy")  # Partial: began before we started.
    live.on_trade(2 * MINUTE + 1, 101.0, 1.0, "buy")  # Complete.
    live.flush("BTC-USD", store)
    store.write_backfill("BTC-USD", "1m", [backfilled(0, 50.0), backfilled(MINUTE, 60.0), backfilled(2 * MINUTE, 70.0)])
    rows = store.read("BTC-USD", "1m", 0, 3 * MINUTE)
    assert [(c.open_ms, c.open, c.source) for c in rows] == [
        (0, 50.0, "backfill:coinbase"), (MINUTE, 60.0, "backfill:coinbase"), (2 * MINUTE, 101.0, "live"),
    ]


def test_merge_takes_extremes_and_medians_after_the_basis():
    merged = merge(
        {"coinbase": [backfilled(0, 100.0)], "hyperliquid": [backfilled(0, 101.0, volume=2.0)]},
        basis={"hyperliquid": math.log(1.01)},
    )
    [candle] = merged
    assert candle.open == pytest.approx(100.0)  # Hyperliquid's 101 is 100 at the composite's level.
    assert candle.high == pytest.approx(101.0) and candle.low == pytest.approx(99.0, abs=0.01)
    assert candle.volume == 3.0 and candle.trade_count == 6
    assert candle.source == "backfill:coinbase,hyperliquid"


def test_history_backfills_a_sparse_window(tmp_path, monkeypatch):
    calls = []

    def fake_coinbase(coin, interval, from_ms, to_ms):
        calls.append((coin, interval))
        first = -(-from_ms // MINUTE) * MINUTE
        return [backfilled(t, 100.0) for t in range(first, to_ms, MINUTE)]

    monkeypatch.setattr(candles, "FETCHERS", {"coinbase": fake_coinbase})
    hub = Hub(["BTC"], ["coinbase"], Store(tmp_path / "c.sqlite"))
    with TestClient(create_app(hub, None, start_hub=False)) as client:
        to_ms = 100 * MINUTE
        response = client.get(f"/api/history/BTC-USD?interval=1m&from={to_ms - 30 * MINUTE}&to={to_ms}").json()
        assert calls == [("BTC", "1m")]
        assert response["count"] == 30 and not response["backfilling"]
        client.get(f"/api/history/BTC-USD?interval=1m&from={to_ms - 30 * MINUTE}&to={to_ms}")
        assert len(calls) == 1  # Already covered.
        assert client.get("/api/history/BTC-USD?interval=2m").status_code == 400


def test_closed_partials_are_found_for_repair(tmp_path):
    store = Store(tmp_path / "c.sqlite")
    live = LiveCandles(started_ms=MINUTE + 30_000)
    live.on_trade(MINUTE + 40_000, 100.0, 1.0, "buy")
    live.flush("BTC-USD", store)
    assert store.closed_partials("BTC-USD", "1m", 2 * MINUTE - 1) == []  # Still forming.
    assert store.closed_partials("BTC-USD", "1m", 2 * MINUTE) == [MINUTE]
    store.write_backfill("BTC-USD", "1m", [backfilled(MINUTE, 60.0)])
    assert store.closed_partials("BTC-USD", "1m", 2 * MINUTE) == []
