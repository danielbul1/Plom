"""Paper market maker: quotes around the mid and simulates its fills against the real tape.

Nothing is sent to the exchange. Fills are inferred from public data:
- a trade through our price, or the opposite side of the book reaching it, fills the whole order;
- a trade at our price first eats the visible size that was queued ahead of us.
Our orders don't move the real market, so results are optimistic for sizes that would.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from plom.hyperliquid import Book, Trade
from plom.pressure import pressure

Side = Literal["buy", "sell"]
SIDES: tuple[Side, Side] = ("buy", "sell")
Key = tuple[Side, int]
"""An order slot: side and layer (0 is nearest the mid)."""
EPSILON = 1e-9


@dataclass(frozen=True)
class Config:
    order_size: float = 0.001
    """Size of the inner layer."""
    max_position: float = 0.01
    layers: int = 3
    layer_spacing: float = 4.0
    """Each layer sits layer_spacing times further from the reservation price than the one inside it."""
    size_growth: float = 2.0
    """Each layer is size_growth times larger than the one inside it."""
    base_half_spread_bps: float = 0.5
    """Distance of the inner layer from the reservation price."""
    vol_multiplier: float = 1.0
    """Half spread widens to vol_multiplier x one-second volatility when that is larger."""
    vol_half_life_s: float = 30.0
    inventory_skew_bps: float = 2.0
    """How far quotes shift against the position when it is at max_position."""
    pressure_skew_bps: float = 1.0
    """How far quotes shift towards the book-pressure bias."""
    pressure_enter: float = 0.3
    pressure_exit: float = 0.1
    latency_ms: int = 150
    requote_interval_ms: int = 500
    fill_cooldown_ms: int = 1000
    maker_fee_bps: float = 0.0
    markout_horizons_ms: tuple[int, ...] = (1000, 5000)


@dataclass
class Order:
    side: Side
    layer: int
    price: float
    size: float
    """Remaining size. A pending order with size 0 is a cancel."""
    live_from_ms: int
    queue_ahead: float = 0.0


@dataclass(frozen=True)
class Fill:
    time_ms: int
    side: Side
    layer: int
    price: float
    size: float
    fee: float
    mid: float


@dataclass(frozen=True)
class Markout:
    fill: Fill
    horizon_ms: int
    bps: float
    """How far the mid moved in the fill's favour after horizon_ms."""


@dataclass
class MarketMaker:
    config: Config = field(default_factory=Config)
    now_ms: int = 0
    book: Book | None = None
    mid: float | None = None
    pressure: float = 0.0
    bias: int = 0
    """-1, 0 or +1: which way book pressure leans, with hysteresis."""
    position: float = 0.0
    cash: float = 0.0
    fees: float = 0.0
    volume: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    markouts: list[Markout] = field(default_factory=list)
    orders: dict[Key, Order] = field(default_factory=dict)
    pending: dict[Key, Order] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._var_per_s = 0.0
        self._vol_ref: tuple[int, float] | None = None
        self._last_requote_ms: int | None = None
        self._last_fill_ms: dict[Side, int | None] = {"buy": None, "sell": None}
        self._unsettled: dict[int, deque[Fill]] = {h: deque() for h in self.config.markout_horizons_ms}

    @property
    def pnl(self) -> float:
        return self.cash + self.position * (self.mid or 0.0)

    @property
    def vol_bps(self) -> float:
        """One-second volatility of the mid, in bps."""
        return math.sqrt(self._var_per_s)

    def on_book(self, book: Book) -> None:
        self.now_ms = max(self.now_ms, book.time_ms)
        self.book = book
        self._activate_pending()
        for order in list(self.orders.values()):
            self._match_book(order, book)
        if not book.bids or not book.asks:
            return
        self._update_mid(book)
        self._update_bias(book)
        self._settle_markouts()
        self._requote()

    def on_trade(self, trade: Trade) -> None:
        self.now_ms = max(self.now_ms, trade.time_ms)
        self._activate_pending()
        resting: Side = "buy" if trade.side == "sell" else "sell"
        for order in sorted(
            (o for o in self.orders.values() if o.side == resting), key=lambda o: o.layer
        ):
            if trade.time_ms < order.live_from_ms:
                continue
            if _better(resting, order.price, trade.price):
                self._fill(order, order.size)
            elif trade.price == order.price:
                order.queue_ahead -= trade.size
                if order.queue_ahead < 0:
                    self._fill(order, min(order.size, -order.queue_ahead))
                    order.queue_ahead = 0.0

    def _activate_pending(self) -> None:
        if self.book is None:
            return
        for key, order in list(self.pending.items()):
            if order.live_from_ms > self.now_ms:
                continue
            del self.pending[key]
            if order.size <= EPSILON:
                self.orders.pop(key, None)
                continue
            levels = self.book.bids if order.side == "buy" else self.book.asks
            order.queue_ahead = dict(levels).get(order.price, 0.0)
            self.orders[key] = order

    def _match_book(self, order: Order, book: Book) -> None:
        own = book.bids if order.side == "buy" else book.asks
        opposite = book.asks if order.side == "buy" else book.bids
        if opposite and not _better(order.side, opposite[0][0], order.price):
            self._fill(order, order.size)
            return
        visible = dict(own)
        if order.price in visible:
            order.queue_ahead = min(order.queue_ahead, visible[order.price])
        elif own and not _better(order.side, own[-1][0], order.price):
            # Inside the visible range but the level is gone: nobody is ahead of us.
            order.queue_ahead = 0.0

    def _fill(self, order: Order, size: float) -> None:
        if size <= EPSILON:
            return
        notional = order.price * size
        fee = notional * self.config.maker_fee_bps / 10_000
        sign = 1 if order.side == "buy" else -1
        self.position += sign * size
        self.cash -= sign * notional + fee
        self.fees += fee
        self.volume += notional
        fill = Fill(self.now_ms, order.side, order.layer, order.price, size, fee, self.mid or order.price)
        self.fills.append(fill)
        for queue in self._unsettled.values():
            queue.append(fill)
        self._last_fill_ms[order.side] = self.now_ms
        order.size -= size
        key = (order.side, order.layer)
        if order.size <= EPSILON and self.orders.get(key) is order:
            del self.orders[key]

    def _update_mid(self, book: Book) -> None:
        self.mid = (book.bids[0][0] + book.asks[0][0]) / 2
        if self._vol_ref is None:
            self._vol_ref = (self.now_ms, self.mid)
            return
        ref_ms, ref_mid = self._vol_ref
        dt_s = (self.now_ms - ref_ms) / 1000
        if dt_s <= 0:
            return
        move_bps = (self.mid / ref_mid - 1) * 10_000
        alpha = 1 - 0.5 ** (dt_s / self.config.vol_half_life_s)
        self._var_per_s = (1 - alpha) * self._var_per_s + alpha * move_bps**2 / dt_s
        self._vol_ref = (self.now_ms, self.mid)

    def _update_bias(self, book: Book) -> None:
        reading = pressure(book.bids, book.asks)
        if reading is None:
            return
        self.pressure = reading.value
        if self.bias == 0:
            if abs(self.pressure) >= self.config.pressure_enter:
                self.bias = 1 if self.pressure > 0 else -1
        elif self.bias * self.pressure < self.config.pressure_exit:
            self.bias = 0

    def _settle_markouts(self) -> None:
        for horizon, queue in self._unsettled.items():
            while queue and queue[0].time_ms + horizon <= self.now_ms:
                fill = queue.popleft()
                self.markouts.append(Markout(fill, horizon, edge_bps(fill, self.mid)))

    def _requote(self) -> None:
        c = self.config
        if self._last_requote_ms is not None and self.now_ms - self._last_requote_ms < c.requote_interval_ms:
            return
        self._last_requote_ms = self.now_ms
        bids, asks = quotes(
            self.mid, self.book.bids[0][0], self.book.asks[0][0],
            self.vol_bps, self.position, self.bias, c,
        )
        capacity = {"buy": c.max_position - self.position, "sell": c.max_position + self.position}
        live_from = self.now_ms + c.latency_ms
        for side, prices in (("buy", bids), ("sell", asks)):
            last_fill = self._last_fill_ms[side]
            if last_fill is not None and self.now_ms - last_fill < c.fill_cooldown_ms:
                continue
            wanted = _ladder(prices, c, capacity[side])
            for key in {k for k in (*self.orders, *self.pending) if k[0] == side}:
                current = self.pending.get(key) or self.orders.get(key)
                if key[1] not in wanted and current.size > EPSILON:
                    self.pending[key] = Order(side, key[1], current.price, 0.0, live_from)
            for layer, (price, size) in wanted.items():
                key = (side, layer)
                current = self.pending.get(key) or self.orders.get(key)
                if current is None or current.size <= EPSILON or current.price != price:
                    self.pending[key] = Order(side, layer, price, size, live_from)


def quotes(
    mid: float,
    best_bid: float,
    best_ask: float,
    vol_bps: float,
    position: float,
    bias: int,
    config: Config,
) -> tuple[list[float], list[float]]:
    """Bid and ask prices per layer, inner first, around a reservation price skewed by inventory and bias.

    Never crosses the book or our own quotes, and no two layers on a side share a price.
    """
    tick = tick_size(mid)
    half_spread_bps = max(config.base_half_spread_bps, config.vol_multiplier * vol_bps)
    skew_bps = (
        -config.inventory_skew_bps * position / config.max_position
        + config.pressure_skew_bps * bias
    )
    reservation = mid * (1 + skew_bps / 10_000)
    bid_ticks: list[int] = []
    ask_ticks: list[int] = []
    for layer in range(config.layers):
        offset = half_spread_bps * config.layer_spacing**layer / 10_000
        bid = math.floor(reservation * (1 - offset) / tick + EPSILON)
        ask = math.ceil(reservation * (1 + offset) / tick - EPSILON)
        if layer == 0:
            bid = min(bid, round(best_ask / tick) - 1)
            ask = max(ask, round(best_bid / tick) + 1, bid + 1)
        else:
            bid = min(bid, bid_ticks[-1] - 1)
            ask = max(ask, ask_ticks[-1] + 1)
        bid_ticks.append(bid)
        ask_ticks.append(ask)
    return [_price(t, tick) for t in bid_ticks], [_price(t, tick) for t in ask_ticks]


def tick_size(price: float) -> float:
    """Hyperliquid prices carry at most 5 significant figures."""
    return 10.0 ** (math.floor(math.log10(price)) - 4)


def edge_bps(fill: Fill, mid: float) -> float:
    """How far `mid` sits in the fill's favour, in bps of the fill price."""
    sign = 1 if fill.side == "buy" else -1
    return sign * (mid - fill.price) / fill.price * 10_000


def _ladder(prices: list[float], config: Config, capacity: float) -> dict[int, tuple[float, float]]:
    """Price and size per layer, growing outwards, filling inner layers first until capacity runs out."""
    wanted = {}
    for layer, price in enumerate(prices):
        size = min(config.order_size * config.size_growth**layer, capacity)
        if size <= EPSILON:
            break
        capacity -= size
        wanted[layer] = (price, size)
    return wanted


def _price(ticks: int, tick: float) -> float:
    return round(ticks * tick, max(0, -math.floor(math.log10(tick))))


def _better(side: Side, price: float, than: float) -> bool:
    """Whether `price` is a strictly better price than `than` for a resting `side` order."""
    return price > than if side == "buy" else price < than
