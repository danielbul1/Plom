# Plom

## Order-book pressure

Distance-weighted bid depth versus ask depth over the nearest 12 book levels, streamed live from Hyperliquid.
Ranges from -1 (all weight on the ask) to +1 (all weight on the bid).

```bash
uv run plom BTC
uv run plom ETH --depth 12 --half-life-bps 5
```

`--half-life-bps` sets how fast weight decays with distance from the mid: a level that far away counts half.

```bash
uv run pytest
```
