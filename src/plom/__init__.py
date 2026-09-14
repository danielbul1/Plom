import argparse
import asyncio
import time

from plom.hyperliquid import l2_books
from plom.pressure import DEFAULT_DEPTH, DEFAULT_HALF_LIFE_BPS, pressure

BAR_WIDTH = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Live order-book pressure from Hyperliquid.")
    parser.add_argument("coin", nargs="?", default="BTC")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--half-life-bps", type=float, default=DEFAULT_HALF_LIFE_BPS)
    args = parser.parse_args()

    try:
        asyncio.run(_stream(args.coin, args.depth, args.half_life_bps))
    except KeyboardInterrupt:
        pass


async def _stream(coin: str, depth: int, half_life_bps: float) -> None:
    async for bids, asks in l2_books(coin):
        reading = pressure(bids, asks, depth, half_life_bps)
        if reading is None:
            continue
        print(
            f"{time.strftime('%H:%M:%S')}  {coin}  mid {reading.mid:>12,.4f}  "
            f"pressure {reading.value:+.3f}  {_bar(reading.value)}",
            flush=True,
        )


def _bar(value: float) -> str:
    filled = round(abs(value) * BAR_WIDTH)
    ask_side = ("-" * filled if value < 0 else "").rjust(BAR_WIDTH)
    bid_side = ("+" * filled if value > 0 else "").ljust(BAR_WIDTH)
    return f"[{ask_side}|{bid_side}]"
