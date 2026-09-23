import pytest

from plom.hub import positioning
from plom.hub.liquidations import LEVERAGE_TIERS, MAINTENANCE_MARGIN, LiquidationModel
from plom.hub.positioning import OpenInterest


def opened_model():
    model = LiquidationModel()
    model.on_open_interest(OpenInterest("okx", 0, 100.0))  # Baseline.
    model.on_trade("okx", "buy", 100.0, 3.0)
    model.on_trade("okx", "sell", 100.0, 1.0)
    model.on_open_interest(OpenInterest("okx", 1000, 110.0))  # +10 coins: 7.5 longs, 2.5 shorts.
    return model


def test_rising_open_interest_opens_positions_split_by_taker_flow():
    model = opened_model()
    longs = sum(model.coins(c, 1000) for c in model.clusters.values() if c.side == "long")
    shorts = sum(model.coins(c, 1000) for c in model.clusters.values() if c.side == "short")
    assert longs == pytest.approx(7.5) and shorts == pytest.approx(2.5)
    ten_x_long = next(c for c in model.clusters.values() if c.side == "long" and c.leverage == 10)
    assert ten_x_long.liquidation_price == pytest.approx(100 * (1 - 0.1 + MAINTENANCE_MARGIN))
    assert ten_x_long.size == pytest.approx(7.5 * LEVERAGE_TIERS[10][0])


def test_falling_open_interest_shrinks_that_venues_positions():
    model = opened_model()
    model.on_open_interest(OpenInterest("okx", 2000, 99.0))  # A tenth of all open interest closed.
    assert sum(model.coins(c, 2000) for c in model.clusters.values()) == pytest.approx(9.0, rel=1e-3)


def test_trading_through_a_level_removes_it():
    model = opened_model()
    hundred_x = 100 * (1 - 0.01 + MAINTENANCE_MARGIN)  # 99.5
    model.on_trade("okx", "sell", hundred_x - 0.01, 0.1)
    assert not any(c.side == "long" and c.leverage == 100 for c in model.clusters.values())
    assert any(c.side == "long" and c.leverage == 50 for c in model.clusters.values())
    assert model.removed_by_price == pytest.approx(7.5 * LEVERAGE_TIERS[100][0])


def test_levels_bucket_longs_below_and_shorts_above_and_decay():
    model = opened_model()
    levels = model.levels(1000, bucket_pct=0.5)
    assert all(price < 100 for price, _ in levels["longs"]) and all(price > 100 for price, _ in levels["shorts"])
    total = sum(n for _, n in levels["longs"]) + sum(n for _, n in levels["shorts"])
    later = model.levels(1000 + 18 * 3_600_000, bucket_pct=0.5)
    assert sum(n for _, n in later["longs"]) + sum(n for _, n in later["shorts"]) < total
    assert levels["by_leverage"]["long"]["100"] == pytest.approx(7.5 * 0.15 * 99.5, rel=1e-3)


def test_position_zones_rank_recent_openings():
    model = opened_model()
    [zone] = model.position_zones(1000, window_s=60, bucket_pct=1)
    assert zone["longs"] == pytest.approx(750.0) and zone["shorts"] == pytest.approx(250.0)
    assert model.position_zones(10 * 60_000, window_s=60) == []


def test_liquidation_parsers_report_the_liquidated_side():
    [b] = positioning._parse_binance_liquidation({"o": {"S": "SELL", "p": "100", "ap": "99.5", "z": "0.5", "q": "0.5", "T": 7}}, "BTC", 1.0)
    assert (b.position, b.price, b.size) == ("long", 99.5, 0.5)
    [y] = positioning._parse_bybit_liquidation({"data": [{"T": 8, "S": "Sell", "v": "2", "p": "101"}]}, "BTC", 1.0)
    assert (y.position, y.size) == ("short", 2.0)
    okx = {"data": [
        {"instId": "ETH-USDT-SWAP", "details": [{"ts": "1", "posSide": "long", "bkPx": "3000", "sz": "5"}]},
        {"instId": "BTC-USDT-SWAP", "details": [{"ts": "9", "posSide": "short", "bkPx": "100", "sz": "20"}]},
    ]}
    [o] = positioning._parse_okx_liquidation(okx, "BTC", 0.01)
    assert (o.position, o.size) == ("short", pytest.approx(0.2))


def test_open_interest_parsers():
    assert positioning._parse_okx_oi({"data": [{"ts": "5", "oiCcy": "30217.2"}]}) == [OpenInterest("okx", 5, 30217.2)]
    assert positioning._parse_bybit_oi({"ts": 6, "data": {"lastPrice": "1"}}) == []
    assert positioning._parse_bybit_oi({"ts": 6, "data": {"openInterest": "59925.9"}}) == [OpenInterest("bybit", 6, 59925.9)]
    [h] = positioning._parse_hyperliquid_oi({"data": {"ctx": {"openInterest": "38985.0"}}})
    assert h.venue == "hyperliquid" and h.coins == 38985.0


def test_replayed_intervals_open_positions_and_clear_crossed_levels():
    model = LiquidationModel()
    model.on_interval("okx", 0, 100.0, 0, 0, 99.0, 101.0, 100.0)  # Baseline.
    model.on_interval("okx", 300_000, 110.0, 3.0, 1.0, 99.0, 101.0, 100.0)  # +10: 7.5 longs, 2.5 shorts.
    assert sum(c.size for c in model.clusters.values()) == pytest.approx(10.0)
    # The next interval's low reaches the 100x longs' level (99.5) but not the 50x (98.5).
    model.on_interval("okx", 600_000, 110.0, 0, 0, 99.4, 100.2, 99.8)
    assert not any(c.side == "long" and c.leverage == 100 for c in model.clusters.values())
    assert any(c.side == "long" and c.leverage == 50 for c in model.clusters.values())


def test_okx_pages_retry_when_rate_limited(monkeypatch):
    import io
    import json
    import urllib.error

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)
        return io.BytesIO(json.dumps({"data": [["5", "1"]]}).encode())

    monkeypatch.setattr(positioning.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(positioning.time, "sleep", lambda s: None)
    assert positioning._get_okx("https://example/x") == [["5", "1"]]
    assert len(calls) == 2
