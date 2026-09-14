# Plom

Order-book tools on Hyperliquid's public data. Nothing here sends orders to an exchange.

## Order-book pressure

Distance-weighted bid depth versus ask depth over the nearest 12 book levels.
Ranges from -1 (all weight on the ask) to +1 (all weight on the bid).

```bash
uv run plom pressure BTC
uv run plom pressure ETH --depth 12 --half-life-bps 5
```

`--half-life-bps` sets how fast weight decays with distance from the mid: a level that far away counts half.

## Paper market maker

Quotes both sides around the mid and simulates fills against the real book and tape.

- **Layers**: `--layers` quotes per side. Each sits `--layer-spacing` times further out than the one inside it and is `--size-growth` times larger, so size is backloaded away from the touch. Inner layers get size first when `--max-position` limits it.
- **Spread**: the inner layer sits `--base-half-spread-bps` from the reservation price, widened to `--vol-multiplier` x one-second volatility when that is larger.
- **Inventory skew**: quotes shift against the position by up to `--inventory-skew-bps` at `--max-position`.
- **Pressure bias**: quotes shift `--pressure-skew-bps` towards book pressure once it crosses `--pressure-enter`, until it falls below `--pressure-exit`.
- **Pickoff defense**: each side scores how far the mid moves against its fills `--pickoff-horizon-ms` later, reaching 1 at `--pickoff-full-bps`, decaying with `--pickoff-decay-s` while the side doesn't fill. A picked-off side quotes up to (1 + `--pickoff-spread-mult`) times further out and cuts its inner size by up to `--pickoff-size-cut`.
- **One-sided in trends**: once the mid drifts `--trend-enter-z` expected moves over `--trend-window-s`, the side being run over is pulled (asks in an uptrend, bids in a downtrend) until the drift falls below `--trend-exit-z`.
- **Jumps**: a book-to-book mid step of `--jump-bps`, or a trade that far through the mid, widens all quotes `--jump-spread-mult` times and scales sizes by `--jump-size-mult` for `--jump-hold-ms`. A trade jump also pulls the side it swept until the next book arrives.
- **Cadence**: requote once the mid moves `--requote-move-bps` from where we last quoted, after `--requote-ttl-ms`, or after a fill or bias change, but no more often than `--requote-interval-ms`. Jumps and trend changes requote immediately.
- **Execution**: orders go live after `--latency-ms`, and a side pauses `--fill-cooldown-ms` after a fill.
- **Fills**: a trade through our price or the book crossing it fills the order; a trade at our price first eats the size queued ahead of us.

```bash
uv run plom mm BTC
uv run plom mm BTC --maker-fee-bps 1.5 --latency-ms 250
uv run plom mm --help
```

Output: position, PnL marked at mid, fills, **edge** (fill price vs mid at fill) and **markouts** (mid 1s / 5s after the fill vs fill price).
Edge minus markout decay is the adverse selection. Ctrl+C prints a summary broken down per layer.

Record live data once and replay it to compare settings:

```bash
uv run plom record BTC btc.jsonl
uv run plom mm BTC --replay btc.jsonl --base-half-spread-bps 1
```

Caveats: our orders never move the real market, queue position is estimated from visible size, and latency is a fixed guess.

## Tests

```bash
uv run pytest
```
