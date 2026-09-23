"""Contract sizes for venues that count book or trade sizes in contracts rather than coins.

Each lookup also tells us whether the venue lists the coin at all: None means it doesn't.
"""

import json
import urllib.request
from collections.abc import Callable

BOOK_IN_CONTRACTS = {"okx", "htx_perps", "blofin"}
TRADES_IN_CONTRACTS = {"okx", "blofin"}
"""HTX perp trades carry the coin quantity alongside the contract count, and we parse that."""


def _get(url: str) -> dict:
    # Some venues refuse urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "plom/0.1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def _okx(coin: str) -> float | None:
    data = _get(f"https://www.okx.com/api/v5/public/instruments?instType=SWAP&instId={coin.upper()}-USDT-SWAP")["data"]
    return float(data[0]["ctVal"]) * float(data[0].get("ctMult") or 1) if data else None


def _htx_perps(coin: str) -> float | None:
    data = _get(f"https://api.hbdm.com/linear-swap-api/v1/swap_contract_info?contract_code={coin.upper()}-USDT").get("data")
    return float(data[0]["contract_size"]) if data else None


def _blofin(coin: str) -> float | None:
    data = _get(f"https://openapi.blofin.com/api/v1/market/instruments?instId={coin.upper()}-USDT").get("data")
    return float(data[0]["contractValue"]) if data else None


LOOKUPS: dict[str, Callable[[str], float | None]] = {"okx": _okx, "htx_perps": _htx_perps, "blofin": _blofin}


def contract_size(venue: str, coin: str) -> float | None:
    """Coins per contract, 1.0 for venues that count in coins, or None when the venue lacks the coin."""
    lookup = LOOKUPS.get(venue)
    return lookup(coin) if lookup else 1.0
