# Plom

Order-book tools on Hyperliquid's public data. Nothing here sends orders to an exchange.

## Order-book pressure

Distance-weighted bid depth versus ask depth over the nearest 12 book levels.
Ranges from -1 (all weight on the ask) to +1 (all weight on the bid).

```bash
uv run plom pressure BTC
uv run plom pressure ETH --venue lighter --depth 12 --half-life-bps 5
```

`--half-life-bps` sets how fast weight decays with distance from the mid: a level that far away counts half.

## Paper market maker

Quotes both sides around the mid and simulates fills against the real book and tape.

- **Layers**: `--layers` quotes per side. Each sits `--layer-spacing` times further out than the one inside it and is `--size-growth` times larger, so size is backloaded away from the touch. Inner layers get size first when `--max-position` limits it.
- **Spread**: the inner layer sits `--base-half-spread-bps` from the reservation price, widened to `--vol-multiplier` x one-second volatility when that is larger.
- **Fair price**: quotes centre on the mid, moved `--microprice-weight` of the way to the microprice once top-of-book size imbalance reaches `--microprice-imbalance`.
- **Regimes**: recent volatility over its `--vol-baseline-half-life-s` baseline. Below `--calm-below` is calm (tighter, bigger, slower TTL); above `--chaotic-above` is chaotic (wider, smaller, `--chaotic-layers` deeper-spaced layers with size backloaded harder, faster TTL). The `--calm-*` and `--chaotic-*` multipliers set how much.
- **Inventory skew**: quotes shift against the position by up to `--inventory-skew-bps` at `--max-position`.
- **Pressure bias**: quotes shift `--pressure-skew-bps` towards book pressure once it crosses `--pressure-enter`, until it falls below `--pressure-exit`.
- **Pickoff defense**: each side scores how far the mid moves against its fills `--pickoff-horizon-ms` later, reaching 1 at `--pickoff-full-bps`, decaying with `--pickoff-decay-s` while the side doesn't fill. A picked-off side quotes up to (1 + `--pickoff-spread-mult`) times further out and cuts its inner size by up to `--pickoff-size-cut`.
- **One-sided in trends**: once the mid drifts `--trend-enter-z` expected moves over `--trend-window-s`, the side being run over is pulled (asks in an uptrend, bids in a downtrend) until the drift falls below `--trend-exit-z`.
- **Jumps**: a book-to-book mid step of `--jump-bps`, or a trade that far through the mid, widens all quotes `--jump-spread-mult` times and scales sizes by `--jump-size-mult` for `--jump-hold-ms`. A trade jump also pulls the side it swept until the next book arrives.
- **Cadence**: requote once the mid moves `--requote-move-bps` from where we last quoted, after `--requote-ttl-ms`, or after a fill or bias change, but no more often than `--requote-interval-ms`. Jumps and trend changes requote immediately.
- **Execution**: orders rest after `--order-latency-ms`; a replaced or cancelled order stays fillable for `--cancel-latency-ms`. Placements, replacements and cancels beyond `--tx-per-minute` are skipped. A side pauses `--fill-cooldown-ms` after a fill.
- **Fills**: a trade through our price fills us up to the trade's size, and the opposite book reaching our price fills us up to the size resting there. A trade at our price first eats the size queued ahead of us. Size that leaves our level without trading is assumed to leave from behind us, unless `--queue-power` n > 0 splits it ahead/behind as front^n : back^n (hftbacktest's power probability queue model).

### Venue profiles

`--profile` sets fees, latencies, tick size and rate limits for a venue (see `src/plom/profiles.py` for sources). Without one, `--venue` picks its default; any explicit flag overrides the profile.

| Profile | Venue | Maker / taker | Order / cancel latency | Tx per minute |
|---|---|---|---|---|
| `ideal` | hyperliquid | 0 / 0 | 150 / 150ms | unlimited |
| `hyperliquid` | hyperliquid | 1.5 / 4.5 bps | 150 / 150ms | unlimited |
| `lighter-standard` | lighter | 0 / 0 | 350 / 450ms | 60 |
| `lighter-premium` | lighter | 0.4 / 2.8 bps | 150 / 150ms | 4,000 |
| `orderly-raydium` | orderly | 0 / 4.5 bps | 150 / 150ms | 600 |

```bash
uv run plom mm BTC
uv run plom mm BTC --replay data/btc.jsonl.gz --profile lighter-standard
uv run plom mm BTC --profile hyperliquid --order-latency-ms 250 --queue-power 2
uv run plom mm --help
```

Ctrl+C (or the end of a replay) prints an evaluation:

- **PnL attribution**: PnL = spread capture (each fill's edge against the mid at fill time) + inventory PnL (the position marked through later mid moves) - fees. Exact, not estimated.
- **Markouts** at 100ms, 1s, 5s, 30s, 1m and 5m: how far the mid moved in our favour after each fill, size-weighted, with the dollars of spread capture that survived to that horizon. Edge minus markout is adverse selection.
- **Confidence**: PnL per hour and each markout get a 95% block-bootstrap interval. Fills cluster, so whole `--block-s` blocks of market time (default 300s) are resampled, not individual fills.
- A breakdown of fills, edge and short markouts per layer and per regime.

## Comparing configurations

`plom compare` replays one recording under several configurations in parallel processes and ranks them. Each variant's PnL per hour is also compared with the profile's defaults block by block (a paired bootstrap), which separates real differences from market noise far better than comparing two totals.

```bash
uv run plom compare data/btc.jsonl.gz --profile lighter-standard --grid base-half-spread-bps=0.5,1,2 --grid layers=1,3
```

`--grid FLAG=V1,V2` takes any config flag; repeated grids are crossed. An interval that straddles zero means the data can't tell yet.

## Recording

`plom record` saves raw books and trades from several venues into one file, each line tagged with the venue and our local receive time (the only clock shared across venues). A `.gz` path compresses it.

| Venue | Book | Trades | Notes |
|---|---|---|---|
| `hyperliquid` | full L2 snapshots, ~0.5s | all | subscribes with the undocumented `fast` flag |
| `binance` | top of book, first update per 50ms | aggregated | USD-M futures, used as a reference price |
| `lighter` | snapshot + deltas, ~50ms | all, incl. liquidations | local book rebuilt from nonce-chained deltas |
| `orderly` | snapshot + deltas, ~200ms | all | local book rebuilt from `prevTs`-chained deltas |

```bash
uv run plom record BTC data/btc.jsonl.gz
uv run plom record BTC data/btc.jsonl.gz --venues hyperliquid,binance
uv run plom mm BTC --replay data/btc.jsonl.gz --venue lighter
```

Replays pick one venue with `--venue` (default `hyperliquid`). Older Hyperliquid-only recordings still replay.

Caveats: our orders never move the real market, queue position is estimated from visible (aggregated) size, and latencies are fixed estimates until measured against live orders.

## Tests

```bash
uv run pytest
```
