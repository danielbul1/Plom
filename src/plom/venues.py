from plom import binance, hyperliquid, lighter, orderly

VENUES = {venue.NAME: venue for venue in (hyperliquid, binance, lighter, orderly)}
"""Each venue module has `messages(coin)`, an async stream of raw messages, and a `Parser` class."""
