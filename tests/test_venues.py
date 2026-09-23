import gzip
import json

import pytest

from plom import aster, binance, blofin, coinbase, feed, htx, hyperliquid, lighter, okx, orderly, recording
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


# Coinbase --------------------------------------------------------------------------------------

def test_coinbase_updates_wait_for_a_snapshot_and_apply_deltas():
    parser = coinbase.Parser()
    update = {"type": "l2update", "time": "1970-01-01T00:00:01.5Z", "changes": [["buy", "100.5", "2"], ["sell", "101", "0"]]}
    assert parser.events(update) == []
    snapshot = {"type": "snapshot", "bids": [["100", "1"]], "asks": [["101", "1"], ["102", "3"]]}
    assert parser.events(snapshot) == []
    assert parser.events(update) == [Book(1500, [(100.5, 2.0), (100.0, 1.0)], [(102.0, 3.0)])]


@pytest.mark.parametrize("maker_side, side", [("buy", "sell"), ("sell", "buy")])
def test_coinbase_match_side_is_the_takers(maker_side, side):
    message = {"type": "match", "time": "1970-01-01T00:00:00.007Z", "side": maker_side, "price": "100", "size": "0.5"}
    assert coinbase.Parser().events(message) == [Trade(7, side, 100.0, 0.5)]


# OKX -------------------------------------------------------------------------------------------

def test_okx_bbo_and_trades():
    bbo = {"arg": {"channel": "bbo-tbt"}, "data": [{"asks": [["101", "3", "0", "1"]], "bids": [["100", "2", "0", "1"]], "ts": "9"}]}
    assert okx.Parser().events(bbo) == [Book(9, [(100.0, 2.0)], [(101.0, 3.0)])]
    trades = {"arg": {"channel": "trades"}, "data": [{"px": "100.5", "sz": "1.2", "side": "sell", "ts": "11"}]}
    assert okx.Parser().events(trades) == [Trade(11, "sell", 100.5, 1.2)]


# HTX -------------------------------------------------------------------------------------------

def test_htx_spot_bbo_and_trades():
    bbo = {"ch": "market.btcusdt.bbo", "tick": {"ask": 101.0, "askSize": 3.0, "bid": 100.0, "bidSize": 2.0, "quoteTime": 9}}
    assert htx.Spot.Parser().events(bbo) == [Book(9, [(100.0, 2.0)], [(101.0, 3.0)])]
    trade = {"ch": "market.btcusdt.trade.detail", "tick": {"data": [{"ts": 11, "amount": 0.4, "price": 100.5, "direction": "buy"}]}}
    assert htx.Spot.Parser().events(trade) == [Trade(11, "buy", 100.5, 0.4)]


def test_htx_perps_bbo_and_trades_in_coin_quantity():
    bbo = {"ch": "market.BTC-USDT.bbo", "tick": {"bid": [100.0, 2], "ask": [101.0, 3], "ts": 9}}
    assert htx.Perps.Parser().events(bbo) == [Book(9, [(100.0, 2)], [(101.0, 3)])]
    trade = {"ch": "market.BTC-USDT.trade.detail", "tick": {"data": [{"ts": 11, "amount": 4, "quantity": 0.004, "price": 100.5, "direction": "sell"}]}}
    assert htx.Perps.Parser().events(trade) == [Trade(11, "sell", 100.5, 0.004)]


def test_htx_answers_pings_and_decodes_gzip():
    source = htx._source(htx.Spot.WS_URL, "market.btcusdt")
    assert source.pong(source.decode(gzip.compress(b'{"ping": 42}'))) == {"pong": 42}
    assert source.pong({"ch": "market.btcusdt.bbo"}) is None


# BloFin and Aster ------------------------------------------------------------------------------

def test_blofin_books5_and_trades():
    books = {"arg": {"channel": "books5"}, "data": {"asks": [["101", "3"], ["102", "1"]], "bids": [["100", "2"]], "ts": "9"}}
    assert blofin.Parser().events(books) == [Book(9, [(100.0, 2.0)], [(101.0, 3.0), (102.0, 1.0)])]
    trades = {"arg": {"channel": "trades"}, "data": [{"price": "100.5", "size": "8", "side": "buy", "ts": "11"}]}
    assert blofin.Parser().events(trades) == [Trade(11, "buy", 100.5, 8.0)]


def test_aster_parses_like_binance():
    message = {"stream": "btcusdt@bookTicker", "data": {"e": "bookTicker", "b": "100.1", "B": "2", "a": "100.2", "A": "3", "T": 9}}
    assert aster.Parser().events(message) == [Book(9, [(100.1, 2.0)], [(100.2, 3.0)])]


# Depth -----------------------------------------------------------------------------------------

def okx_books(action, seq, prev, bids=(), asks=(), ts="5"):
    rows = lambda levels: [[str(p), str(s), "0", "1"] for p, s in levels]
    return {"arg": {"channel": "books"}, "action": action, "data": [{"bids": rows(bids), "asks": rows(asks), "ts": ts, "seqId": seq, "prevSeqId": prev}]}


def test_okx_books_apply_chained_deltas_and_drop_after_a_gap():
    parser = okx.Parser(depth=2)
    [book] = parser.events(okx_books("snapshot", 10, -1, [(100, 1), (99, 1), (98, 1)], [(101, 1)]))
    assert book.bids == [(100.0, 1.0), (99.0, 1.0)]  # Truncated to the parser's depth.
    [book] = parser.events(okx_books("update", 11, 10, [(100, 0)], [(101, 5)]))
    assert book.bids[0] == (99.0, 1.0) and book.asks == [(101.0, 5.0)]
    assert parser.events(okx_books("update", 13, 12, [(97, 1)])) == []
    assert parser.events(okx_books("update", 14, 13, [(96, 1)])) == []  # Still out of sync.
    [book] = parser.events(okx_books("snapshot", 20, -1, [(90, 1)], [(91, 1)]))
    assert book.bids == [(90.0, 1.0)]


def test_chained_sync_check_flags_a_gap():
    in_sync = feed.chained(okx._sequence)()
    assert in_sync({"arg": {"channel": "trades"}, "data": []})
    assert in_sync(okx_books("snapshot", 10, -1))
    assert in_sync(okx_books("update", 11, 10))
    assert not in_sync(okx_books("update", 13, 12))


def test_blofin_books_snapshot_then_delta():
    parser = blofin.Parser()
    snapshot = {"arg": {"channel": "books"}, "action": "snapshot", "data": {"bids": [["100", "1"]], "asks": [["101", "2"]], "ts": "5", "prevSeqId": "0", "seqId": "7"}}
    update = {"arg": {"channel": "books"}, "action": "update", "data": {"bids": [["100.5", "3"]], "asks": [], "ts": "6", "prevSeqId": "7", "seqId": "8"}}
    parser.events(snapshot)
    [book] = parser.events(update)
    assert book == Book(6, [(100.5, 3.0), (100.0, 1.0)], [(101.0, 2.0)])


def test_htx_lays_a_fresher_bbo_over_the_depth_snapshot():
    parser = htx.Perps.Parser()
    depth = {"ch": "market.BTC-USDT.depth.step0", "tick": {"bids": [[100.0, 5], [99.0, 5]], "asks": [[101.0, 5], [102.0, 5]], "ts": 10}}
    [book] = parser.events(depth)
    assert len(book.bids) == 2
    bbo = {"ch": "market.BTC-USDT.bbo", "tick": {"bid": [100.5, 1], "ask": [101.5, 2], "ts": 20}}
    [book] = parser.events(bbo)
    assert book == Book(20, [(100.5, 1), (100.0, 5), (99.0, 5)], [(101.5, 2), (102.0, 5)])
    [book] = parser.events({**depth, "tick": {**depth["tick"], "ts": 30}})
    assert book.bids[0] == (100.0, 5)  # A newer snapshot wins.


def test_aster_depth_snapshot():
    message = {"data": {"e": "depthUpdate", "T": 9, "b": [["100", "1"], ["99", "2"]], "a": [["101", "3"]]}}
    assert aster.Parser().events(message) == [Book(9, [(100.0, 1.0), (99.0, 2.0)], [(101.0, 3.0)])]
