import time

import pytest
from fastapi.testclient import TestClient

from plom.hub.api import create_app
from plom.hub.runner import Hub, coin_of, scale_book, symbol_of
from plom.hub.state import FRESH_MS, SymbolState, default_bucket
from plom.market import Book, Trade

VENUES = ("coinbase", "okx", "aster", "hyperliquid")


def book(bid, ask, size=1.0):
    return Book(0, [(bid, size)], [(ask, size)])


def filled_state(now=1_000_000.0):
    state = SymbolState("BTC-USD", VENUES)
    state.on_book("coinbase", now, book(100.0, 100.2, 1.0))
    state.on_book("okx", now, book(100.1, 100.3, 2.0))
    state.on_book("aster", now, book(99.9, 100.1, 3.0))
    return state


def test_tick_has_the_composite_and_each_fresh_venue():
    now = 1_000_000.0
    state = filled_state(now)
    state.on_book("hyperliquid", now - FRESH_MS - 1, book(1.0, 2.0))  # Stale.
    tick = state.tick(now)
    assert tick["price"] == pytest.approx(100.1, rel=1e-6)
    assert tick["prices"] == pytest.approx({"aster": 100.0, "coinbase": 100.1, "okx": 100.2})
    assert set(tick["ages_ms"]) == {"aster", "coinbase", "okx"}


def test_volume_1s_counts_only_the_last_second_of_trades():
    state = filled_state()
    state.on_trade("okx", 0.0, Trade(0, "buy", 100.0, 5.0))
    state.on_trade("okx", 999_500.0, Trade(0, "buy", 100.0, 1.5))
    state.on_trade("aster", 999_900.0, Trade(0, "sell", 100.0, 0.5))
    assert state.tick(1_000_000.0)["volume_1s"] == pytest.approx(2.0)


def test_dom_sums_venues_into_buckets_rounding_away_from_the_touch():
    state = filled_state()
    dom = state.dom(1_000_000.0, bucket=0.5)
    assert [(level["price"], level["size"]) for level in dom["bids"]] == [(100.0, 3.0), (99.5, 3.0)]
    assert dom["bids"][0]["venues"] == {"coinbase": 1.0, "okx": 2.0}
    assert [(level["price"], level["size"]) for level in dom["asks"]] == [(100.5, 6.0)]


def test_adjusted_dom_moves_venues_onto_the_composite_level():
    state = SymbolState("BTC-USD", VENUES)
    now = 1_000_000.0
    for venue, mid in (("coinbase", 100.0), ("okx", 100.0), ("hyperliquid", 101.0)):
        state.on_book(venue, now, book(mid - 0.05, mid + 0.05))
    # Hyperliquid joined at the composite's level, so its basis is ~1% and adjusting removes it.
    raw = state.dom(now, bucket=0.1)
    adjusted = state.dom(now, bucket=0.1, adjusted=True)
    assert raw["bids"][0]["venues"] == {"hyperliquid": 1.0}
    assert adjusted["bids"][0]["size"] == pytest.approx(3.0)


def test_prints_after_returns_newer_trades_oldest_first():
    state = SymbolState("BTC-USD", VENUES)
    for i in range(5):
        state.on_trade("okx", float(i), Trade(i, "buy", 100.0, 1.0))
    assert [p.seq for p in state.prints_after(2)] == [3, 4, 5]
    assert [p.seq for p in state.prints_after(0, limit=2)] == [4, 5]
    assert state.prints_after(5) == []


def test_default_bucket_is_about_a_basis_point():
    assert default_bucket(85_000) == 1
    assert default_bucket(3_000) == pytest.approx(0.1)
    assert default_bucket(0.12) == pytest.approx(1e-5)


def test_symbols_and_contract_scaling():
    assert symbol_of("btc") == "BTC-USD" and coin_of("ETH-USD") == "ETH"
    assert scale_book(book(100.0, 101.0, 20.0), 0.01) == book(100.0, 101.0, 0.2)


@pytest.fixture
def client():
    hub = Hub(["BTC"], list(VENUES))
    state = hub.states["BTC-USD"]
    now = time.time() * 1000
    state.on_book("coinbase", now, book(100.0, 100.2))
    state.on_book("okx", now, book(100.1, 100.3))
    state.on_book("aster", now, book(99.9, 100.1))
    state.on_trade("okx", now, Trade(1, "sell", 100.1, 2.0))
    with TestClient(create_app(hub, "secret", start_hub=False)) as test_client:
        yield test_client


def test_api_requires_the_token(client):
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/universe").status_code == 401
    assert client.get("/api/v1/universe", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/v1/universe?token=secret").json() == {"symbols": ["BTC-USD"]}


def test_api_market_endpoints(client):
    auth = {"Authorization": "Bearer secret"}
    assert client.get("/api/v1/market/tick?symbol=btc-usd", headers=auth).json()["price"] == pytest.approx(100.1, rel=1e-6)
    assert client.get("/api/v1/market/tick?symbol=NOPE-USD", headers=auth).status_code == 404
    assert client.get("/api/v1/market/dom?symbol=BTC-USD&bucket=0.5", headers=auth).json()["asks"][0]["price"] == 100.5
    [trade] = client.get("/api/v1/market/tape?symbol=BTC-USD", headers=auth).json()["trades"]
    assert trade["venue"] == "okx" and trade["notional"] == pytest.approx(200.2)
    batch = client.get("/api/v1/market/snapshot/batch?symbols=BTC-USD&include=tick,dom", headers=auth).json()
    assert set(batch["snapshots"]["BTC-USD"]) == {"tick", "dom"}


def test_websocket_streams_subscribed_symbols(client):
    with client.websocket_connect("/api/ws?token=secret") as ws:
        ws.send_json({"event": "subscribe_symbols", "symbols": ["BTC-USD", "NOPE-USD"]})
        assert ws.receive_json() == {"event": "subscribed", "symbols": ["BTC-USD"]}
        events = {ws.receive_json()["event"] for _ in range(3)}
        assert {"tick", "dom"} <= events


def test_heatmap_columns_put_bids_and_asks_on_one_grid_around_the_price():
    state = SymbolState("BTC-USD", VENUES)
    now = 1_000_000.0
    state.on_book("coinbase", now, Book(0, [(100.0, 1.0), (99.8, 2.0)], [(100.2, 3.0)]))
    state.on_book("okx", now, Book(0, [(100.0, 1.0)], [(100.2, 1.0), (100.5, 4.0)]))
    state.on_book("aster", now, Book(0, [(100.0, 1.0)], [(100.2, 1.0)]))
    column = state.sample_heatmap(now)
    assert column.bucket == 0.01 and column.price == pytest.approx(100.1)
    index = lambda price: round((price - column.low) / column.bucket)
    assert column.bids[index(100.0)] == pytest.approx(3.0) and column.bids[index(99.8)] == pytest.approx(2.0)
    assert column.asks[index(100.2)] == pytest.approx(5.0) and column.asks[index(100.5)] == pytest.approx(4.0)
    assert column.bids[index(100.2)] == 0 and column.asks[index(100.0)] == 0
    for t in range(1, 5):
        state.sample_heatmap(now + t * 1000)
    assert [c.ts_ms for c in state.heatmap_since(now + 4000, 2.5)] == [now + 2000, now + 3000, now + 4000]
    assert [c.ts_ms for c in state.heatmap_since(now + 4000, 10, step=2)] == [now, now + 2000, now + 4000]


def test_api_heatmap(client):
    response = client.get("/api/v1/market/heatmap?symbol=BTC-USD&token=secret").json()
    assert response["columns"] == [] and response["levels"] > 0


def test_dashboard_and_its_chart_library_are_served_without_a_token(client):
    page = client.get("/")
    assert page.status_code == 200 and "/static/lightweight-charts.js" in page.text
    assert client.get("/static/lightweight-charts.js").status_code == 200


def test_recordings_are_listed_and_served_by_plain_name_only(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "btc_20260923_1200.jsonl.gz").write_bytes(b"data")
    (tmp_path / "secret.jsonl.gz").write_bytes(b"outside")
    app = create_app(Hub(["BTC"], list(VENUES)), "secret", start_hub=False, recordings=recordings)
    with TestClient(app) as client:
        auth = {"Authorization": "Bearer secret"}
        assert client.get("/api/v1/recordings").status_code == 401
        [listed] = client.get("/api/v1/recordings", headers=auth).json()["recordings"]
        assert listed["name"] == "btc_20260923_1200.jsonl.gz" and listed["bytes"] == 4
        assert client.get("/api/v1/recordings/btc_20260923_1200.jsonl.gz", headers=auth).content == b"data"
        for bad in ("..%2Fsecret.jsonl.gz", "%2E%2E%2Fsecret.jsonl.gz", "missing.jsonl.gz", "candles.sqlite"):
            assert client.get(f"/api/v1/recordings/{bad}", headers=auth).status_code == 404
