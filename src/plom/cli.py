import argparse
import asyncio
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
import time
from collections.abc import AsyncIterator
from dataclasses import fields, replace
from pathlib import Path
from statistics import fmean

from plom import composite, evaluate, recording, runner
from plom.alpha import FEATURES
from plom.market import Book, Trade
from plom.mm import Config, MarketMaker, edge_bps
from plom.pressure import DEFAULT_DEPTH, DEFAULT_HALF_LIFE_BPS, pressure
from plom.profiles import DEFAULT_PROFILE, PROFILES
from plom.venues import VENUES

BAR_WIDTH = 20
SERVE_COINS = ("BTC", "ETH")
SERVE_VENUES = tuple(dict.fromkeys(("hyperliquid", "orderly", *composite.VENUES)))
RECORD_VENUES = dict.fromkeys(("hyperliquid", "lighter", "orderly", *composite.VENUES))


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
    m.add_argument(
        "--reference", choices=[*VENUES, composite.NAME, "none"], default=composite.NAME,
        help="venue whose books pull fair value (default: the composite; none to quote on the venue alone)",
    )
    m.add_argument("--status-every-s", type=float, default=5.0)
    m.add_argument("--block-s", type=float, default=evaluate.BLOCK_S, help="bootstrap block length")
    for f in fields(Config):
        if isinstance(f.default, int | float):
            m.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), help=f"default {f.default}")

    r = commands.add_parser("record", help="save live books and trades from several venues to one file")
    r.add_argument("coin")
    r.add_argument("path", type=Path, help="JSONL file to append to; .gz compresses it")
    r.add_argument(
        "--venues", default=",".join(RECORD_VENUES),
        help=f"comma-separated from {', '.join(VENUES)}; default: the quoting venues and the composite's",
    )
    r.add_argument("--status-every-s", type=float, default=60.0)

    k = commands.add_parser("compare", help="replay a recording under several configs, in parallel, and compare")
    k.add_argument("replay", type=Path)
    k.add_argument("--venue", choices=[v for v in VENUES if v in DEFAULT_PROFILE])
    k.add_argument("--profile", choices=PROFILES)
    k.add_argument(
        "--reference", choices=[*VENUES, composite.NAME, "none"], default=composite.NAME,
        help="venue whose books pull fair value (default: the composite; none to quote on the venue alone)",
    )
    k.add_argument(
        "--grid", action="append", default=[], metavar="FLAG=V1,V2,...",
        help="config values to try, e.g. --grid base-half-spread-bps=0.5,1,2; repeat to cross several",
    )
    k.add_argument("--block-s", type=float, default=evaluate.BLOCK_S)
    k.add_argument("--workers", type=int, default=os.cpu_count())

    ll = commands.add_parser("leadlag", help="how far each venue's mid lags a reference, and how well the gap predicts catch-up")
    ll.add_argument("replay", type=Path)
    ll.add_argument("--reference", choices=[*VENUES, composite.NAME], default=composite.NAME)
    ll.add_argument("--venues", default="hyperliquid,lighter,orderly")

    sv = commands.add_parser("serve", help="aggregate live data from every venue and serve it over REST and WebSocket")
    sv.add_argument("--coins", default=os.environ.get("PLOM_COINS", ",".join(SERVE_COINS)))
    sv.add_argument("--venues", default=os.environ.get("PLOM_VENUES", ",".join(SERVE_VENUES)))
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))

    args = parser.parse_args()
    command = {
        "pressure": _pressure, "mm": _mm, "record": _record, "compare": _compare, "leadlag": _leadlag, "serve": _serve,
    }[args.command]
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
    reference = None if args.reference in ("none", venue) else args.reference
    mm = MarketMaker(config)
    tracker = evaluate.Tracker(args.block_s)
    dispatcher = runner.Dispatcher(mm)
    events = _replay(args.replay, venue, reference) if args.replay else runner.live(venue, reference, args.coin)
    last_status_ms = None
    try:
        async for is_reference, recv_ms, event in events:
            dispatcher.feed(is_reference, recv_ms, event)
            tracker.observe(mm)
            if mm.mid is None:
                continue
            if last_status_ms is None or mm.now_ms - last_status_ms >= args.status_every_s * 1000:
                last_status_ms = mm.now_ms
                print(_status(args.coin, mm), flush=True)
    finally:
        print(evaluate.format_report(evaluate.evaluate(mm, tracker)), flush=True)
        print(breakdown(mm), flush=True)


async def _compare(args: argparse.Namespace) -> None:
    profile_name = args.profile or DEFAULT_PROFILE[args.venue or "hyperliquid"]
    profile = PROFILES[profile_name]
    venue = args.venue or profile.venue
    base = replace(Config(), **profile.config)
    defaults = {f.name: f.default for f in fields(Config)}
    axes = []
    for spec in args.grid:
        flag, _, values = spec.partition("=")
        name = flag.removeprefix("--").replace("-", "_")
        if name not in defaults or not values:
            raise SystemExit(f"bad --grid {spec!r}: expected FLAG=V1,V2 with a config flag")
        axes.append([(name, type(defaults[name])(value)) for value in values.split(",")])
    variants = [("base", base)] + [
        (" ".join(f"{name}={value}" for name, value in combo), replace(base, **dict(combo)))
        for combo in itertools.product(*axes)
    ]
    reference = None if args.reference in ("none", venue) else args.reference
    print(f"{venue} ({profile_name}), reference {reference}: {len(variants)} runs on {args.replay}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(evaluate.replay, args.replay, venue, config, label, args.block_s, reference)
            for label, config in variants
        ]
        results = [future.result() for future in futures]
    base_result = results[0]
    print(f"\n{base_result.hours:.2f}h of market time, {len(base_result.block_pnls)} blocks of {args.block_s:g}s\n")
    print(f"{'variant':<40} {'fills':>6} {'fees $':>8} {'pnl $':>9} {'pnl $/h [95%]':>30} {'vs base $/h [95%]':>30} {'mo1s':>6} {'mo60s':>6}")
    for e in sorted(results, key=lambda e: e.pnl, reverse=True):
        horizon = {h.horizon_ms: h for h in e.horizons}
        mo = lambda ms: f"{horizon[ms].mean_bps.mean:+6.2f}" if ms in horizon and horizon[ms].mean_bps else "   n/a"
        difference = "" if e is base_result else str(evaluate.paired_difference(e, base_result) or "n/a")
        print(
            f"{e.label:<40} {e.fills:>6} {e.fees:>8.3f} {e.pnl:>+9.3f} {str(e.pnl_per_hour or 'n/a'):>30} "
            f"{difference:>30} {mo(1000)} {mo(60_000)}"
        )


async def _leadlag(args: argparse.Namespace) -> None:
    from plom import leadlag

    venues = [v for v in args.venues.split(",") if v != args.reference]
    print(leadlag.format_report(args.reference, leadlag.measure(args.replay, args.reference, venues)))


async def _serve(args: argparse.Namespace) -> None:
    import logging

    import uvicorn

    from plom.hub.api import create_app
    from plom.hub.runner import Hub

    venues = args.venues.split(",")
    unknown = set(venues) - set(VENUES)
    if unknown:
        raise SystemExit(f"unknown venues: {', '.join(sorted(unknown))}")
    token = os.environ.get("PLOM_TOKEN")
    if token is None:
        print("PLOM_TOKEN is not set: the API is open to anyone who can reach it", flush=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from plom.hub.candles import Store

    data_dir = Path(os.environ.get("PLOM_DATA", "data"))
    hub = Hub(args.coins.split(","), venues, Store(data_dir / "candles.sqlite"))
    server = uvicorn.Server(uvicorn.Config(create_app(hub, token), host=args.host, port=args.port, log_level="info"))
    await server.serve()


async def _replay(path: Path, venue: str, reference: str | None) -> AsyncIterator[runner.Event]:
    for item in runner.replay(path, venue, reference):
        yield item


async def _live_events(venue: str, coin: str) -> AsyncIterator[Book | Trade]:
    parser = VENUES[venue].Parser()
    async for message in VENUES[venue].messages(coin):
        for event in parser.events(message):
            yield event


def _status(coin: str, mm: MarketMaker) -> str:
    quotes = "  ".join(_side_quotes(mm, side) for side in ("buy", "sell"))
    markouts = " ".join(
        f"mo{h / 1000:g}s {_mean([m.bps for m in mm.markouts if m.horizon_ms == h]):+.2f}"
        for h in mm.config.markout_horizons_ms
        if h in (1000, 5000)
    )
    return (
        f"{_clock(mm.now_ms)}  {coin}  mid {mm.mid:,.6g}  [{quotes}]  "
        f"pos {mm.position:+.5f} {mm.position_age_s:.0f}s  pnl {mm.pnl:+.2f}  fills {len(mm.fills)}  "
        f"edge {_mean([edge_bps(f, f.mid) for f in mm.fills]):+.2f}bps  {markouts}  "
        f"vol {mm.vol_bps:.2f}bps x{mm.vol_ratio:.1f} {mm.regime}  bias {mm.bias:+d}  trend {mm.trend:+d}  "
        f"pickoff b{mm.pickoff_score('buy'):.2f}/s{mm.pickoff_score('sell'):.2f}"
    )


def _side_quotes(mm: MarketMaker, side: str) -> str:
    """The inner price and how many layers are live, e.g. `buy 78,611 x3`."""
    orders = sorted((o for o in mm.orders.values() if o.side == side), key=lambda o: o.layer)
    return f"{side} {orders[0].price:,.6g} x{len(orders)}" if orders else f"{side} -"


def breakdown(mm: MarketMaker) -> str:
    """Fills, edge and short markouts per layer and per regime."""
    horizons = [h for h in mm.config.markout_horizons_ms if h <= 5000]
    header = "fills     volume    edge  " + "  ".join(f"{f'mo{h / 1000:g}s':>6}" for h in horizons)
    lines = [
        "",
        f"pulled         buy {mm.pulled_ms['buy'] / 1000:,.0f}s  sell {mm.pulled_ms['sell'] / 1000:,.0f}s"
        f"   jumps {mm.jumps}",
        "regime time    " + "  ".join(f"{r} {ms / 1000:,.0f}s" for r, ms in mm.regime_ms.items()),
        f"volatility     realized {mm.volatility.realized_bps:.2f}bps  bipower {mm.volatility.bipower_bps:.2f}bps"
        f" on a {mm.config.vol_sample_ms}ms grid, {mm.volatility.jump_share:.0%} from jumps;"
        f" Lee-Mykland jumps {mm.volatility.jumps}",
        f"reference      {mm.reference_jumps} jumps, basis {(mm.basis or 0) * 10_000:+.2f}bps",
        f"alpha          out-of-sample r2 {mm.alpha.r2 if mm.alpha.r2 is None else round(mm.alpha.r2, 3)}"
        f" over {mm.alpha.scored:,} forecasts; weights "
        + "  ".join(f"{name} {w:+.3f}" for name, w in zip(FEATURES, mm.alpha.weights))
        + f"; pulls buy {mm.alpha_pulls['buy']} sell {mm.alpha_pulls['sell']}",
        _glft_line(mm),
        f"flattens       {len(mm.flattens)} fills, ${sum(f.price * f.size for f in mm.flattens):,.2f}, cost "
        f"${sum(-edge_bps(f, f.mid) * f.price * f.size / 10_000 + f.fee for f in mm.flattens):,.4f} (crossing + fees)",
        "",
        f"layer    {header}",
    ]
    for label, group in [
        *((f"{layer:>5}  ", [f for f in mm.fills if f.layer == layer]) for layer in sorted({f.layer for f in mm.fills})),
        ("", []),
        *((f"{regime:<7}", [f for f in mm.fills if f.regime == regime]) for regime in mm.regime_ms),
    ]:
        if not label:
            lines += ["", f"regime   {header}"]
            continue
        if not group:
            continue
        ids = {id(f) for f in group}
        markouts = [
            _mean([m.bps for m in mm.markouts if id(m.fill) in ids and m.horizon_ms == h]) for h in horizons
        ]
        lines.append(
            f"{label}  {len(group):>5}  {sum(f.price * f.size for f in group):>9,.0f}  "
            f"{_mean([edge_bps(f, f.mid) for f in group]):>+6.2f}  "
            + "  ".join(f"{m:>+6.2f}" for m in markouts)
        )
    return "\n".join(lines)


def _glft_line(mm: MarketMaker) -> str:
    intensity = mm.intensity
    if intensity.k is None:
        return "glft           fill intensity not calibrated yet"
    line = f"glft           A {intensity.a:.3f}/s  k {intensity.k:.3f}/bp"
    model = mm.glft
    if model:
        line += f"  -> half spread {model.half_spread_bps:.2f}bps, skew {model.skew_bps:.3f}bps/lot"
    return line


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _clock(time_ms: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(time_ms / 1000))


def _bar(value: float) -> str:
    filled = round(abs(value) * BAR_WIDTH)
    ask_side = ("-" * filled if value < 0 else "").rjust(BAR_WIDTH)
    bid_side = ("+" * filled if value > 0 else "").ljust(BAR_WIDTH)
    return f"[{ask_side}|{bid_side}]"
