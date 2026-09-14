"""Per-venue execution assumptions for the paper market maker.

Latencies are ~150ms of network round trip from here (measured on public feeds) plus any delay the
venue adds on purpose. Tick sizes are for BTC. Figures as documented in September 2026:
- Hyperliquid: base tier 1.5 bps maker / 4.5 bps taker.
  https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
- Lighter Standard: no fees, but 60 transactions a minute, a 300ms cancel delay and a maker delay
  documented as both 0 and 200ms (we take 200). Premium: 0.4 / 2.8 bps, no maker or cancel delay.
  https://apidocs.lighter.xyz/docs/account-types
- Orderly via Raydium Perps: 0 bps maker / 4.5 bps taker, 10 order requests a second.
  https://docs.raydium.io/user-flows/perpetuals-trading-fees
"""

from dataclasses import dataclass, field

NETWORK_MS = 150


@dataclass(frozen=True)
class Profile:
    venue: str
    config: dict = field(default_factory=dict)
    """Config overrides."""


PROFILES = {
    "ideal": Profile("hyperliquid", {}),
    "hyperliquid": Profile(
        "hyperliquid",
        {"maker_fee_bps": 1.5, "taker_fee_bps": 4.5, "order_latency_ms": NETWORK_MS, "cancel_latency_ms": NETWORK_MS},
    ),
    "lighter-standard": Profile(
        "lighter",
        {
            "tick_size": 0.1, "maker_fee_bps": 0.0, "taker_fee_bps": 0.0,
            "order_latency_ms": NETWORK_MS + 200, "cancel_latency_ms": NETWORK_MS + 300, "tx_per_minute": 60,
        },
    ),
    "lighter-premium": Profile(
        "lighter",
        {
            "tick_size": 0.1, "maker_fee_bps": 0.4, "taker_fee_bps": 2.8,
            "order_latency_ms": NETWORK_MS, "cancel_latency_ms": NETWORK_MS, "tx_per_minute": 4000,
        },
    ),
    "orderly-raydium": Profile(
        "orderly",
        {
            "tick_size": 0.1, "maker_fee_bps": 0.0, "taker_fee_bps": 4.5,
            "order_latency_ms": NETWORK_MS, "cancel_latency_ms": NETWORK_MS, "tx_per_minute": 600,
        },
    ),
}

DEFAULT_PROFILE = {"hyperliquid": "hyperliquid", "lighter": "lighter-standard", "orderly": "orderly-raydium"}
"""The profile used for a venue when none is named."""
