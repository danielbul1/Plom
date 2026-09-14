import argparse
import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import fields, replace
from pathlib import Path
from statistics import fmean

from plom import hyperliquid
from plom.hyperliquid import Book
from plom.mm import Config, MarketMaker, edge_bps
from plom.pressure import DEFAULT_DEPTH, DEFAULT_HALF_LIFE_BPS, pressure

BAR_WIDTH = 20


def main() -> None:
    parser = argparse.ArgumentParser(prog="plom")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pressure", help="live order-book pressure")
    p.add_argument("coin", nargs="?", default="BTC")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--half-life-bps", type=float, default=DEFAULT_HALF_LIFE_BPS)

    m = commands.add_parser("mm", help="paper market maker on the live book or a recording")
    m.add_argument("coin", nargs="?", default="BTC")
    source = m.add_mutually_exclusive_group()
    source.add_argument("--replay", type=Path, help="JSONL file written by `plom record`")
    source.add_argument("--record", type=Path, help="also append the live stream to this JSONL file")
    m.add_argument("--status-every-s", type=float, default=5.0)
    for f in fields(Config):
        if isinstance(f.default, int | float):
            m.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)

    r = commands.add_parser("record", help="save the live book and trades to a JSONL file")
    r.add_argument("coin")
    r.add_argument("path", type=Path)

    args = parser.parse_args()
    command = {"pressure": _pressure, "mm": _mm, "record": _record}[args.command]
    try:
        asyncio.run(command(args))
    except KeyboardInterrupt:
        pass


async def _pressure(args: argparse.Namespace) -> None:
    async for message in hyperliquid.messages(args.coin, channels=("l2Book",)):
        for book in hyperliquid.events(message):
            reading = pressure(book.bids, book.asks, args.depth, args.half_life_bps)
            if reading is None:
                continue
            print(
                f"{_clock(book.time_ms)}  {args.coin}  mid {reading.mid:>12,.4f}  "
                f"pressure {reading.value:+.3f}  {_bar(reading.value)}",
                flush=True,
            )


async def _record(args: argparse.Namespace) -> None:
    print(f"Recording {args.coin} to {args.path} (Ctrl+C to stop)", flush=True)
    with args.path.open("a", buffering=1) as out:
        async for message in hyperliquid.messages(args.coin):
            out.write(json.dumps(message) + "\n")


async def _mm(args: argparse.Namespace) -> None:
    config = replace(Config(), **{
        f.name: getattr(args, f.name) for f in fields(Config) if hasattr(args, f.name)
    })
    mm = MarketMaker(config)
    last_status_ms = None
    try:
        async for message in _mm_messages(args):
            for event in hyperliquid.events(message):
                if isinstance(event, Book):
                    mm.on_book(event)
                else:
                    mm.on_trade(event)
            if mm.mid is None:
                continue
            if last_status_ms is None or mm.now_ms - last_status_ms >= args.status_every_s * 1000:
                last_status_ms = mm.now_ms
                print(_status(args.coin, mm), flush=True)
    finally:
        print(_summary(mm), flush=True)


async def _mm_messages(args: argparse.Namespace) -> AsyncIterator[dict]:
    if args.replay:
        with args.replay.open() as recording:
            for line in recording:
                yield json.loads(line)
        return
    out = args.record.open("a", buffering=1) if args.record else None
    try:
        async for message in hyperliquid.messages(args.coin):
            if out:
                out.write(json.dumps(message) + "\n")
            yield message
    finally:
        if out:
            out.close()


def _status(coin: str, mm: MarketMaker) -> str:
    quotes = "  ".join(_side_quotes(mm, side) for side in ("buy", "sell"))
    markouts = " ".join(
        f"mo{h / 1000:g}s {_mean([m.bps for m in mm.markouts if m.horizon_ms == h]):+.2f}"
        for h in mm.config.markout_horizons_ms
    )
    return (
        f"{_clock(mm.now_ms)}  {coin}  mid {mm.mid:,.6g}  [{quotes}]  "
        f"pos {mm.position:+.5f}  pnl {mm.pnl:+.2f}  fills {len(mm.fills)}  "
        f"edge {_mean([edge_bps(f, f.mid) for f in mm.fills]):+.2f}bps  {markouts}  "
        f"vol {mm.vol_bps:.2f}bps  bias {mm.bias:+d}  trend {mm.trend:+d}  "
        f"pickoff b{mm.pickoff_score('buy'):.2f}/s{mm.pickoff_score('sell'):.2f}"
    )


def _side_quotes(mm: MarketMaker, side: str) -> str:
    """The inner price and how many layers are live, e.g. `buy 78,611 x3`."""
    orders = sorted((o for o in mm.orders.values() if o.side == side), key=lambda o: o.layer)
    return f"{side} {orders[0].price:,.6g} x{len(orders)}" if orders else f"{side} -"


def _summary(mm: MarketMaker) -> str:
    horizons = mm.config.markout_horizons_ms
    lines = [
        "",
        "--- summary ---",
        f"fills          {len(mm.fills)}",
        f"volume         ${mm.volume:,.2f}",
        f"position       {mm.position:+.5f}",
        f"fees           ${mm.fees:,.4f}",
        f"pnl (at mid)   ${mm.pnl:+,.4f}",
        f"pulled         buy {mm.pulled_ms['buy'] / 1000:,.0f}s  sell {mm.pulled_ms['sell'] / 1000:,.0f}s",
        f"pickoff        buy {mm.pickoff_bps('buy'):+.2f}bps  sell {mm.pickoff_bps('sell'):+.2f}bps",
        "",
        "layer  fills     volume    edge  " + "  ".join(f"{f'mo{h / 1000:g}s':>6}" for h in horizons),
    ]
    for layer in sorted({f.layer for f in mm.fills}):
        fills = [f for f in mm.fills if f.layer == layer]
        markouts = [
            _mean([m.bps for m in mm.markouts if m.fill.layer == layer and m.horizon_ms == h])
            for h in horizons
        ]
        lines.append(
            f"{layer:>5}  {len(fills):>5}  {sum(f.price * f.size for f in fills):>9,.0f}  "
            f"{_mean([edge_bps(f, f.mid) for f in fills]):>+6.2f}  "
            + "  ".join(f"{m:>+6.2f}" for m in markouts)
        )
    lines.append("(edge and markouts in bps; volume in $)")
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _clock(time_ms: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(time_ms / 1000))


def _bar(value: float) -> str:
    filled = round(abs(value) * BAR_WIDTH)
    ask_side = ("-" * filled if value < 0 else "").rjust(BAR_WIDTH)
    bid_side = ("+" * filled if value > 0 else "").ljust(BAR_WIDTH)
    return f"[{ask_side}|{bid_side}]"
