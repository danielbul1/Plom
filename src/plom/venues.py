from plom import aster, binance, bitunix, blofin, coinbase, htx, hyperliquid, lighter, okx, orderly

VENUES = {
    venue.NAME: venue
    for venue in (hyperliquid, binance, lighter, orderly, coinbase, okx, htx.Spot, htx.Perps, blofin, aster, bitunix)
}
"""Each venue has `messages(coin)`, an async stream of raw messages, and a `Parser` class."""
