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
- **Reference price**: with `--reference binance` (the default), fair value moves `--reference-weight` of the way from the local price to the reference mid adjusted by a learned basis (`--basis-half-life-s`). Reference books are put on the venue's clock through our local receive time. A reference move of `--reference-jump-bps` within `--reference-jump-window-ms` pulls the side it runs towards until the venue's next book and widens quotes. `--reference none` quotes on the venue alone.
- **Forecast**: an online ridge regression predicts the mid's move over `--alpha-horizon-ms` from the reference gap, top-of-book order-flow imbalance, trade-flow imbalance and microprice (see `src/plom/alpha.py`). It learns only from moves that have already happened, so its reported R² is out of sample. `--alpha-weight` shifts fair value by the forecast, `--alpha-widen` widens the side it moves against, and a side is pulled while the forecast move against it exceeds its half spread plus maker fee plus `--alpha-pull-margin-bps`. By default it learns and reports without acting.
- **Inventory skew**: quotes shift against the position by up to `--inventory-skew-bps` at `--max-position`.
- **GLFT**: with `--glft-gamma` above 0, the half spread and inventory skew come from the closed-form GLFT approximation (Guéant, Lehalle and Fernandez-Tapia) instead of `--vol-multiplier` and `--inventory-skew-bps`, with `--base-half-spread-bps` as a floor. Its fill intensity A exp(-k δ) is calibrated online, as in hftbacktest: every `--glft-sample-ms`, how far beyond fair value trades reached on each side, decayed over `--glft-half-life-s` and fitted at `--glft-step-bps` steps (see `src/plom/glft.py`). Inventory counts in lots of `--order-size`, so gamma is per bp per lot. Layers step out by `--glft-layer-spacing` GLFT half spreads instead of growing geometrically. With gamma 0 (the default) it calibrates and reports A and k without quoting from them.
- **Position age**: the inventory skew grows by its own size every `--position-age-s` the position stays open (since it was last flat or changed sign). Once it has been open `--flatten-age-s`, a taker order cuts it by `--flatten-fraction`, landing after the order latency, walking the visible book and paying the taker fee; flattens are at most one per `--flatten-cooldown-ms`.
- **Pressure bias**: quotes shift `--pressure-skew-bps` towards book pressure once it crosses `--pressure-enter`, until it falls below `--pressure-exit`.
- **Pickoff defense**: each side scores how far the mid moves against its fills `--pickoff-horizon-ms` later, reaching 1 at `--pickoff-full-bps`, decaying with `--pickoff-decay-s` while the side doesn't fill. A picked-off side quotes up to (1 + `--pickoff-spread-mult`) times further out and cuts its inner size by up to `--pickoff-size-cut`.
- **One-sided in trends**: once the mid drifts `--trend-enter-z` expected moves over `--trend-window-s`, the side being run over is pulled (asks in an uptrend, bids in a downtrend) until the drift falls below `--trend-exit-z`.
- **Jumps**: a book-to-book mid step of `--jump-bps`, or a trade that far through the mid, widens all quotes `--jump-spread-mult` times and scales sizes by `--jump-size-mult` for `--jump-hold-ms`. A trade jump also pulls the side it swept until the next book arrives. With `--jump-alpha` above 0, the move since the last `--vol-sample-ms` sample is also a jump when the Lee-Mykland test rejects no jump: that move over the local bipower volatility of the `--jump-window-samples` samples before it, against a threshold giving about `--jump-alpha` false jumps a day. It is tested on every book, so a jump is caught when it happens, at most once a sample.
- **Jump-robust volatility**: `--vol-bipower 1` measures volatility and the regime ratio with bipower variation on a `--vol-sample-ms` grid, which ignores jumps, instead of the realized variance of book-to-book moves. The grid defaults to 10s: on one-second BTC books most mids haven't moved, and bipower variation reads far below realized variance on every venue from tick discreteness rather than jumps. The report shows both, and the share of variance that came from jumps (see `src/plom/volatility.py`).
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

## Lead-lag

`plom leadlag` measures, from a multi-venue recording, how far each venue's mid lags the reference (peak cross-correlation of mid returns on a 25ms grid of receive times), and how much of the basis-adjusted gap each venue closes over the next 250ms to 5s.

```bash
uv run plom leadlag data/btc.jsonl.gz
```

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
