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
- **Reference price**: with `--reference composite` (the default), fair value moves `--reference-weight` of the way from the local price to a composite reference adjusted by a learned basis (`--basis-half-life-s`). The composite is built like the aggregated price charting services show, from Coinbase, OKX, HTX spot and perps, BloFin, Aster and Hyperliquid, leaving out the venue being quoted: each venue's mid is moved onto a common level by its own slowly learned basis, and the composite is the median of the venues heard from in the last second (see `src/plom/composite.py`). A single venue such as `--reference binance` also works. Reference books are put on the venue's clock through our local receive time. A reference move of `--reference-jump-bps` within `--reference-jump-window-ms` pulls the side it runs towards until the venue's next book, and at least `--reference-jump-hold-ms`, and widens quotes. The hold matters on venues that lag and send books slowly: Bitunix, sending books about three times a second, trails the composite by 300-500ms and after a sharp move is still 1.5-2.5bps behind when its next book arrives. `--reference none` quotes on the venue alone.
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
| `bitunix` / `bitunix-eth` | bitunix | 2 / 5 bps (VIP 1) | 150 / 150ms (assumed, unmeasured) | 600 |

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

## Rechecking every stage

`scripts/recheck.py` replays a recording, against the composite reference unless `--reference` says otherwise, under the defaults and each B4-B8 idea (reference weight, forecast use, GLFT, position age and flattening, bipower volatility and Lee-Mykland jumps, and pulling on a lagging venue as the reference jumps, including how cancel latency erodes it) for several venue profiles in parallel, and writes one Markdown report: the differences the data can see, lead-lag, a table per stage and venue, and the defaults' full evaluation. A recording still being written can be used.

```bash
uv run python scripts/recheck.py data/btc.jsonl.gz
uv run python scripts/recheck.py data/btc.jsonl.gz --profiles hyperliquid,lighter-standard --out recheck.md
```

## Lead-lag

`plom leadlag` measures, from a multi-venue recording, how far each venue's mid lags the reference (by default the composite, built without the venue being measured) (peak cross-correlation of mid returns on a 25ms grid of receive times), and how much of the basis-adjusted gap each venue closes over the next 250ms to 5s.

```bash
uv run plom leadlag data/btc.jsonl.gz
```

## Opportunity on a lagging venue

`plom opportunity` asks, before any strategy is built, what a venue's quotes stand to lose or win around the composite's sharp moves (at least `--move-bps` within 250ms), from its own recorded book and trades, for each assumed latency in `--latencies-ms`:

- stale side: how often a quote left at the touch the move runs towards gets hit before our cancel lands, and how far the venue's mid has moved past it two seconds on;
- favorable side: how often a quote joined at the other touch after the latency fills within 500ms (trades there must exceed the size queued ahead), and its edge;
- the expected bps per move of each, net of each maker fee in `--fees-bps`.

The venue is timed by its own timestamps plus its fastest typical delivery delay, since venues that batch messages deliver some events long after they happen.

```bash
uv run plom opportunity data/btc.jsonl.gz --venues bitunix,lighter,hyperliquid --move-bps 1.5
uv run plom opportunity data/btc.jsonl.gz --venues bitunix --leaders binance,bybit
```

The composite leaves out Binance and Bybit, as aggregators do. `--leaders binance,bybit` (also on `leadlag`) adds them, to ask whether seeing the biggest venues moves the composite's signal earlier.

## Recording

`plom record` saves raw books and trades from several venues into one file, each line tagged with the venue and our local receive time (the only clock shared across venues). A `.gz` path compresses it.

| Venue | Book | Trades | Notes |
|---|---|---|---|
| `hyperliquid` | full L2 snapshots, ~0.5s | all | subscribes with the undocumented `fast` flag |
| `lighter` | snapshot + deltas, ~50ms | all, incl. liquidations | local book rebuilt from nonce-chained deltas |
| `orderly` | snapshot + deltas, ~200ms | all | local book rebuilt from `prevTs`-chained deltas |
| `coinbase` | 50 levels, snapshot + deltas batched every 50ms | all | spot BTC-USD, in real dollars |
| `okx` | top of book, first update per 50ms | all | USDT perp; sizes in contracts |
| `htx_spot`, `htx_perps` | top of book, first update per 50ms | all | gzip frames; perp book sizes in contracts |
| `blofin` | five levels, first snapshot per 50ms | all | USDT perp; sizes in contracts |
| `aster` | top of book, first update per 50ms | aggregated | USDT perp, Binance-style API |
| `binance` | top of book, first update per 50ms | aggregated | USD-M futures; not recorded by default |
| `bybit` | best bid/ask, first update per 50ms; 50 levels with depth | all | USDT perp; not recorded by default |

By default every venue but Binance and Bybit is recorded. A venue that fails, for instance by refusing the server's region, is retried with backoff and never stops the others. Lighter refuses connections from some jurisdictions.

```bash
uv run plom record BTC data/btc.jsonl.gz
uv run plom record BTC data/btc.jsonl.gz --venues hyperliquid,coinbase,okx
uv run plom mm BTC --replay data/btc.jsonl.gz --venue lighter
```

Replays pick one venue with `--venue` (default `hyperliquid`). Older Hyperliquid-only recordings still replay.

Caveats: our orders never move the real market, queue position is estimated from visible (aggregated) size, and latencies are fixed estimates until measured against live orders.

## Hub

`plom serve` streams every venue's full book (up to 200 levels a side; OKX, BloFin, HTX and Aster switch from top of book to their depth channels, and delta streams reconnect for a new snapshot on a sequence gap) for several coins at once and serves the aggregate over REST and WebSocket, like the aggregator APIs sold to traders, but from our own direct feeds (tens to hundreds of milliseconds old rather than seconds).

```bash
PLOM_TOKEN=... uv run plom serve --coins BTC,ETH,SOL --port 8000
curl -H "Authorization: Bearer $PLOM_TOKEN" "localhost:8000/api/v1/market/tick?symbol=BTC-USD"
```

- `GET /api/v1/market/tick`: the composite price, each fresh venue's mid, spread, age and learned basis, and volume over the last second.
- `GET /api/v1/market/dom`: every venue's book summed into price buckets (`bucket`, default about 1bp), with each level's size by venue. Venues trade at different levels, so the raw merge can look crossed; `adjusted=true` first moves each venue onto the composite's level.
- `GET /api/v1/market/heatmap`: one column a second of the adjusted merged book, for up to 30 minutes (`seconds`, thinned by `step`). Each column is a price grid of 300 buckets centred on the composite price (bucket i is `low + i * bucket`), with bid and ask size per bucket. Venues lead and lag each other by about a basis point, more than any one venue's spread, so bids and asks overlap slightly near the price.
- `GET /api/v1/market/tape`: recent trades from every venue, sizes in coins (contract venues are scaled by their contract size), paged by `after_seq`.
- `GET /api/history/{symbol}`: OHLCV candles (`interval` 1m, 5m, 15m, 1h, 4h or 1d; `from` and `to` in Unix ms, default the last 24 hours; `limit` up to 2,000) with taker buy and sell volume and trade counts. See below.
- `GET /api/v1/premium/liquidation_clusters`: estimated liquidation notional per price bucket (`bucket_pct`), longs below the price and shorts above, and by leverage. See below.
- `GET /api/v1/premium/position_zones`: where the most positions were opened in the last `window_s`, by entry price.
- `GET /api/v1/premium/liquidations`: real forced liquidations reported by Binance, Bybit and OKX.
- `snapshot`, `snapshot/batch`, `latest`, `universe`, `status` (each feed's state) and `health` (no token needed).
- WebSocket `/api/ws?token=...`: send `{"event": "subscribe_symbols", "symbols": ["BTC-USD"]}` to receive `tick` every 250ms, new `tape` prints, and `dom` and the newest `heatmap` column every second.

Candles are built live from every venue's trades, with each venue's prices moved onto the composite's level by its basis, and kept in SQLite under `PLOM_DATA` (default `data/`). On start, and whenever a requested window is less than 80% covered, gaps are backfilled from Coinbase (no 4h) and Hyperliquid (its latest 5,000 candles only): highs and lows across both, opens and closes their median. Backfilled candles cover two venues rather than all of them and carry no buy/sell split, so their volume is lower than live candles'. A live candle whose interval began before the hub started is marked `partial`, and is replaced by a backfilled one once it closes. Windows over 7 days return what is stored at once with `"backfilling": true`.

Liquidation zones are a model, as every liquidation heatmap is: no venue publishes positions' entries or leverage. When a venue's open interest rises (OKX, Bybit, Hyperliquid, and Binance where reachable), positions were opened at about the price its trades printed since the last reading, split into longs and shorts by its taker flow, and spread over leverage tiers (5x to 100x); each liquidates near entry(1 ∓ 1/L ± mmr). When open interest falls by some share, the venue's positions shrink by that share. A level is removed once the price trades through it, and levels fade with age, high leverage faster. On start the model replays OKX's last 72 hours of 5-minute open interest, taker volume and candles, so it doesn't begin empty. Real liquidations are kept alongside, to see and to check the model against (`src/plom/hub/liquidations.py`, `src/plom/hub/positioning.py`).

The dashboard at `/` shows it all live for BTC and ETH: candles with volume coloured by the taker side, a heatmap of the adjusted merged book with the composite price over it (or, on its other tab, the estimated liquidation map with the last hour's real liquidations), each venue's mid, spread and age against the composite, the merged book as a ladder with each level's size split by venue, and the tape. It asks for the token once (or takes `?token=`) and keeps it in the browser. The chart library (TradingView Lightweight Charts, Apache 2.0) is served from the hub itself.

The token comes from `PLOM_TOKEN`; without it the API is open. `PLOM_COINS`, `PLOM_VENUES` and `PORT` set the defaults for `--coins`, `--venues` and `--port`.

## Running on a server

The `Dockerfile` runs `scripts/railway-start.sh`: an hourly recording of each coin in `PLOM_RECORD_COINS` (default BTC and ETH) from `PLOM_RECORD_VENUES` into `$PLOM_DATA/recordings`, deleting files older than `PLOM_KEEP_DAYS` (default 3), with the hub in front on `$PORT`. `railway.json` builds it on Railway with a health check on `/api/v1/health`; mount a volume at `/data` and set `PLOM_TOKEN`. Only list venues the server's region can reach: one that refuses the connection stops the whole recording it is in.

The hub lists its recordings at `/api/v1/recordings` and serves each by name; `PLOM_TOKEN=... uv run python scripts/fetch_recordings.py <hub URL> data/remote` downloads the finished ones that aren't already local.

## Tests

```bash
uv run pytest
```
