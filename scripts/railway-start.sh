#!/bin/sh
# Start Plom on a server: hourly recordings of each coin in the background, and the hub in front.
#
#   PLOM_DATA           where candles and recordings live (default /data, a Railway volume)
#   PLOM_RECORD_COINS   coins to record (default BTC,ETH)
#   PLOM_RECORD_VENUES  venues to record; a venue that refuses the connection would stop a whole
#                       recording, so only list ones reachable from the server's region
#   PLOM_KEEP_DAYS      delete recordings older than this (default 3; two coins take ~1.3GB a day)
#   PORT, PLOM_TOKEN, PLOM_COINS, PLOM_VENUES   as for `plom serve`
set -u
DATA="${PLOM_DATA:-/data}"
RECORDINGS="$DATA/recordings"
COINS="${PLOM_RECORD_COINS:-BTC,ETH}"
VENUES="${PLOM_RECORD_VENUES:-hyperliquid,orderly,coinbase,okx,htx_spot,htx_perps,blofin,aster,bitunix}"
KEEP_DAYS="${PLOM_KEEP_DAYS:-3}"
mkdir -p "$RECORDINGS"
export PLOM_DATA="$DATA"

record_loop() {
  coin="$1"
  name=$(echo "$coin" | tr '[:upper:]' '[:lower:]')
  while true; do
    find "$RECORDINGS" -name '*.jsonl.gz' -mtime +"$KEEP_DAYS" -delete
    timeout 3600 plom record "$coin" "$RECORDINGS/${name}_$(date -u +%Y%m%d_%H%M).jsonl.gz" \
      --venues "$VENUES" --status-every-s 600
    sleep 1  # Also keeps a recording that fails at once from spinning.
  done
}

for coin in $(echo "$COINS" | tr ',' ' '); do
  record_loop "$coin" &
done

exec plom serve --port "${PORT:-8000}"
