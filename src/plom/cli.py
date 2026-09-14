import argparse
import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import fields, replace
from pathlib import Path
from statistics import fmean

from plom import recording
from plom.market import Book, Trade
from plom.mm import Config, MarketMaker, edge_bps
from plom.pressure import DEFAULT_DEPTH, DEFAULT_HALF_LIFE_BPS, pressure
from plom.profiles import DEFAULT_PROFILE, PROFILES
from plom.venues import VENUES

BAR_WIDTH = 20


def main() -> None:
    parser = argparse.ArgumentParser(prog="plom")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pressure", help="live order-book pressure")
    p.add_argument("coin", nargs="?", default="BTC")
    p.add_argument("--venue", choices=VENUES, default="hyperliquid")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--half-life-bps", type=float, default=DEFAULT_HALF_LIFE_BPS)

    m = commands.add_parser("mm", help="paper market maker on the live book or a recording")
    m.add_argument("coin", nargs="?", default="BTC")
    m.add_argument("--venue", choices=[v for v in VENUES if v in DEFAULT_PROFILE], help="the venue to quote on")
    m.add_argument(
        "--profile", choices=PROFILES,
        help="fees, latencies, tick and rate limits (default: the venue's; `ideal` is free and unlimited)",
    )
    m.add_argument("--replay", type=Path, help="recording written by `plom record`")
    m.add_argument("--status-every-s", type=float, default=5.0)
    for f in fields(Config):
        if isinstance(f.default, int | float):
            m.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), help=f"default {f.default}")

    r = commands.add_parser("record", help="save live books and trades from several venues to one file")
    r.add_argument("coin")
    r.add_argument("path", type=Path, help="JSONL file to append to; .gz compresses it")
    r.add_argument("--venues", default=",".join(VENUES), help="comma-separated, default: all")
    r.add_argument("--status-every-s", type=float, default=60.0)

    args = parser.parse_args()
    command = {"pressure": _pressure, "mm": _mm, "record": _record}[args.command]
    try:
        asyncio.run(command(args))
    except KeyboardInterrupt:
        pass


async def _pressure(args: argparse.Namespace) -> None:
    async for event in _live_events(args.venue, args.coin):
        if not isinstance(event, Book):
            continue
        reading = pressure(event.bids, event.asks, args.depth, args.half_life_bps)
        if reading is None:
            continue
        print(
            f"{_clock(event.time_ms)}  {args.coin}  mid {reading.mid:>12,.4f}  "
            f"pressure {reading.value:+.3f}  {_bar(reading.value)}",
            flush=True,
        )


async def _record(args: argparse.Namespace) -> None:
    venues = args.venues.split(",")
    unknown = set(venues) - set(VENUES)
    if unknown:
        raise SystemExit(f"unknown venues: {', '.join(sorted(unknown))}")
    print(f"Recording {args.coin} from {', '.join(venues)} to {args.path} (Ctrl+C to stop)", flush=True)
    await recording.record(args.coin, venues, args.path, args.status_every_s)


async def _mm(args: argparse.Namespace) -> None:
    profile = PROFILES[args.profile or DEFAULT_PROFILE[args.venue or "hyperliquid"]]
    venue = args.venue or profile.venue
    explicit = {f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name, None) is not None}
    config = replace(Config(), **{**profile.config, **explicit})
    print(
        f"{venue} ({args.profile or DEFAULT_PROFILE[venue]}): maker {config.maker_fee_bps}bps, "
        f"latency {config.order_latency_ms}/{config.cancel_latency_ms}ms, {config.tx_per_minute} tx/min",
        flush=True,
    )
    mm = MarketMaker(config)
    events = _replay_events(args.replay, venue) if args.replay else _live_events(venue, args.coin)
    last_status_ms = None
    try:
        async for event in events:
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


async def _live_events(venue: str, coin: str) -> AsyncIterator[Book | Trade]:
    parser = VENUES[venue].Parser()
    async for message in VENUES[venue].messages(coin):
        for event in parser.events(message):
            yield event


async def _replay_events(path: Path, venue: str) -> AsyncIterator[Book | Trade]:
    parser = VENUES[venue].Parser()
    for recorded_venue, _, message in recording.read(path):
        if recorded_venue == venue:
            for event in parser.events(message):
                yield event


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
        f"vol {mm.vol_bps:.2f}bps x{mm.vol_ratio:.1f} {mm.regime}  bias {mm.bias:+d}  trend {mm.trend:+d}  "
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
        f"jumps          {mm.jumps}",
        f"transactions   {mm.tx_sent:,} sent, {mm.tx_skipped:,} skipped by the rate limit",
        "regime time    " + "  ".join(f"{r} {ms / 1000:,.0f}s" for r, ms in mm.regime_ms.items()),
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
    lines += ["", "regime   fills    edge  " + "  ".join(f"{f'mo{h / 1000:g}s':>6}" for h in horizons)]
    for regime in mm.regime_ms:
        fills = [f for f in mm.fills if f.regime == regime]
        if not fills:
            continue
        markouts = [
            _mean([m.bps for m in mm.markouts if m.fill.regime == regime and m.horizon_ms == h])
            for h in horizons
        ]
        lines.append(
            f"{regime:<7}  {len(fills):>5}  {_mean([edge_bps(f, f.mid) for f in fills]):>+6.2f}  "
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
