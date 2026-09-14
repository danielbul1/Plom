import gzip
import json

import pytest

from plom import binance, hyperliquid, lighter, orderly, recording
from plom.market import Book, LocalBook, Trade


def test_local_book_applies_deltas_and_deletes_zero_sizes():
    book = LocalBook()
    book.reset([(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0)])
    book.apply([(100.0, 0.0), (99.5, 3.0)], [(101.0, 4.0), (102.0, 1.0)])
    assert book.book(7) == Book(7, [(99.5, 3.0), (99.0, 2.0)], [(101.0, 4.0), (102.0, 1.0)])


def test_local_book_truncates_depth():
    book = LocalBook()
    book.reset([(100.0 - i, 1.0) for i in range(10)], [(101.0 + i, 1.0) for i in range(10)])
    snapshot = book.book(0, depth=3)
    assert [p for p, _ in snapshot.bids] == [100.0, 99.0, 98.0]
    assert [p for p, _ in snapshot.asks] == [101.0, 102.0, 103.0]


# Lighter ---------------------------------------------------------------------------------------

def lighter_book(kind, nonce, begin_nonce, bids=(), asks=(), updated_us=1_000_000):
    levels = lambda rows: [{"price": str(p), "size": str(s)} for p, s in rows]
    return {
        "type": kind,
        "channel": "order_book:1",
        "order_book": {
            "nonce": nonce, "begin_nonce": begin_nonce, "last_updated_at": updated_us,
            "bids": levels(bids), "asks": levels(asks),
        },
    }


def lighter_trade(trade_id, is_maker_ask, price=100.0, size=0.5, ts=5):
    return {"trade_id": trade_id, "is_maker_ask": is_maker_ask, "price": str(price), "size": str(size), "timestamp": ts}


def test_lighter_snapshot_then_delta():
    parser = lighter.Parser()
    [snapshot] = parser.events(lighter_book("subscribed/order_book", 10, 0, [(100.0, 1.0)], [(101.0, 1.0)], 2_000_000))
    assert snapshot == Book(2000, [(100.0, 1.0)], [(101.0, 1.0)])
    [updated] = parser.events(lighter_book("update/order_book", 11, 10, [(100.0, 0.0), (99.0, 2.0)], []))
    assert updated.bids == [(99.0, 2.0)]


def test_lighter_nonce_gap_drops_books_until_the_next_snapshot():
    parser = lighter.Parser()
    parser.events(lighter_book("subscribed/order_book", 10, 0, [(100.0, 1.0)], [(101.0, 1.0)]))
    assert parser.events(lighter_book("update/order_book", 13, 12, [(99.0, 1.0)], [])) == []
    assert parser.events(lighter_book("update/order_book", 14, 13, [(98.0, 1.0)], [])) == []
    [resynced] = parser.events(lighter_book("subscribed/order_book", 20, 0, [(97.0, 1.0)], [(98.0, 1.0)]))
    assert resynced.bids == [(97.0, 1.0)]


def test_lighter_trades_map_maker_side_to_aggressor_and_skip_replays():
    parser = lighter.Parser()
    message = {"type": "subscribed/trade", "trades": [lighter_trade(2, True), lighter_trade(1, False)], "liquidation_trades": []}
    assert [(t.side, t.time_ms) for t in parser.events(message)] == [("sell", 5), ("buy", 5)]
    replay = {"type": "update/trade", "trades": [lighter_trade(2, True), lighter_trade(3, True)], "liquidation_trades": [lighter_trade(4, False)]}
    assert [t.side for t in parser.events(replay)] == ["buy", "sell"]


# Orderly ---------------------------------------------------------------------------------------

def orderly_snapshot(ts, bids, asks):
    return {"topic": "PERP_BTC_USDC@orderbook", "ts": ts, "data": {"symbol": "PERP_BTC_USDC", "bids": bids, "asks": asks}}


def orderly_update(ts, prev_ts, bids=(), asks=()):
    data = {"symbol": "PERP_BTC_USDC", "prevTs": prev_ts, "bids": list(bids), "asks": list(asks)}
    return {"topic": "PERP_BTC_USDC@orderbookupdate", "ts": ts, "data": data}


def test_orderly_ignores_updates_until_a_snapshot():
    parser = orderly.Parser()
    assert parser.events(orderly_update(100, 90, [[100.0, 1.0]])) == []


def test_orderly_snapshot_then_chained_updates():
    parser = orderly.Parser()
    [snapshot] = parser.events(orderly_snapshot(100, [[100.0, 1.0]], [[101.0, 1.0]]))
    assert snapshot == Book(100, [(100.0, 1.0)], [(101.0, 1.0)])
    assert parser.events(orderly_update(100, 90)) == []  # Already in the snapshot.
    [updated] = parser.events(orderly_update(300, 100, [[100.0, 0.0], [99.5, 2.0]], [[101.0, 3.0]]))
    assert updated == Book(300, [(99.5, 2.0)], [(101.0, 3.0)])


def test_orderly_gap_waits_for_the_next_snapshot():
    parser = orderly.Parser()
    parser.events(orderly_snapshot(100, [[100.0, 1.0]], [[101.0, 1.0]]))
    assert parser.events(orderly_update(500, 300, [[99.0, 1.0]])) == []
    assert parser.events(orderly_update(700, 500, [[98.0, 1.0]])) == []
    [resynced] = parser.events(orderly_snapshot(800, [[97.0, 1.0]], [[98.0, 1.0]]))
    assert resynced.bids == [(97.0, 1.0)]


def test_orderly_trade_side_is_the_aggressor():
    message = {"topic": "PERP_BTC_USDC@trade", "ts": 42, "data": {"price": 100.5, "size": 0.1, "side": "SELL"}}
    assert orderly.Parser().events(message) == [Trade(42, "sell", 100.5, 0.1)]


# Binance ---------------------------------------------------------------------------------------

def test_binance_book_ticker_is_a_one_level_book():
    message = {"stream": "btcusdt@bookTicker", "data": {"e": "bookTicker", "b": "100.1", "B": "2", "a": "100.2", "A": "3", "T": 9, "E": 10}}
    assert binance.Parser().events(message) == [Book(9, [(100.1, 2.0)], [(100.2, 3.0)])]


@pytest.mark.parametrize(("buyer_is_maker", "side"), [(True, "sell"), (False, "buy")])
def test_binance_agg_trade_side(buyer_is_maker, side):
    message = {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "p": "100.0", "q": "0.5", "T": 7, "m": buyer_is_maker}}
    assert binance.Parser().events(message) == [Trade(7, side, 100.0, 0.5)]


# Recording -------------------------------------------------------------------------------------

HL_BOOK = {"channel": "l2Book", "data": {"time": 1, "levels": [[{"px": "1", "sz": "1"}], [{"px": "2", "sz": "1"}]]}}


def test_read_tags_legacy_lines_as_hyperliquid(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(HL_BOOK) + "\n")
    assert list(recording.read(path)) == [("hyperliquid", None, HL_BOOK)]
    assert hyperliquid.Parser().events(HL_BOOK) == [Book(1, [(1.0, 1.0)], [(2.0, 1.0)])]


def test_read_gzip_and_stop_at_a_truncated_tail(tmp_path):
    path = tmp_path / "multi.jsonl.gz"
    line = {"venue": "binance", "recv_ms": 5.0, "msg": {"x": 1}}
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps(line) + "\n")
        out.write('{"venue": "binance", "recv')
    assert list(recording.read(path)) == [("binance", 5.0, {"x": 1})]
