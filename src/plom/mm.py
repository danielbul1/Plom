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
EPSILON = 1e-9


@dataclass(frozen=True)
class Config:
    order_size: float = 0.001
    max_position: float = 0.01
    base_half_spread_bps: float = 0.5
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
    price: float
    size: float
    """Remaining size. A pending order with size 0 is a cancel."""
    live_from_ms: int
    queue_ahead: float = 0.0


@dataclass(frozen=True)
class Fill:
    time_ms: int
    side: Side
    price: float
    size: float
    fee: float
    mid: float


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
    orders: dict[Side, Order | None] = field(default_factory=lambda: {"buy": None, "sell": None})
    pending: dict[Side, Order | None] = field(default_factory=lambda: {"buy": None, "sell": None})
    markouts: dict[int, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._var_per_s = 0.0
        self._vol_ref: tuple[int, float] | None = None
        self._last_requote_ms: int | None = None
        self._last_fill_ms: dict[Side, int | None] = {"buy": None, "sell": None}
        self._unsettled: dict[int, deque[Fill]] = {h: deque() for h in self.config.markout_horizons_ms}
        self.markouts = {h: [] for h in self.config.markout_horizons_ms}

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
        for side in SIDES:
            if order := self.orders[side]:
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
        order = self.orders[resting]
        if order is None or trade.time_ms < order.live_from_ms:
            return
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
        for side in SIDES:
            order = self.pending[side]
            if order is None or order.live_from_ms > self.now_ms:
                continue
            self.pending[side] = None
            if order.size <= EPSILON:
                self.orders[side] = None
                continue
            levels = self.book.bids if side == "buy" else self.book.asks
            order.queue_ahead = dict(levels).get(order.price, 0.0)
            self.orders[side] = order

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
        fill = Fill(self.now_ms, order.side, order.price, size, fee, self.mid or order.price)
        self.fills.append(fill)
        for queue in self._unsettled.values():
            queue.append(fill)
        self._last_fill_ms[order.side] = self.now_ms
        order.size -= size
        if order.size <= EPSILON and self.orders[order.side] is order:
            self.orders[order.side] = None

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
                self.markouts[horizon].append(edge_bps(fill, self.mid))

    def _requote(self) -> None:
        c = self.config
        if self._last_requote_ms is not None and self.now_ms - self._last_requote_ms < c.requote_interval_ms:
            return
        self._last_requote_ms = self.now_ms
        bid, ask = quotes(
            self.mid, self.book.bids[0][0], self.book.asks[0][0],
            self.vol_bps, self.position, self.bias, c,
        )
        wanted = {
            "buy": (bid, min(c.order_size, c.max_position - self.position)),
            "sell": (ask, min(c.order_size, c.max_position + self.position)),
        }
        for side, (price, size) in wanted.items():
            last_fill = self._last_fill_ms[side]
            if last_fill is not None and self.now_ms - last_fill < c.fill_cooldown_ms:
                continue
            current = self.pending[side] or self.orders[side]
            has_order = current is not None and current.size > EPSILON
            live_from = self.now_ms + c.latency_ms
            if size <= EPSILON:
                if has_order:
                    self.pending[side] = Order(side, current.price, 0.0, live_from)
            elif not has_order or current.price != price:
                self.pending[side] = Order(side, price, size, live_from)


def quotes(
    mid: float,
    best_bid: float,
    best_ask: float,
    vol_bps: float,
    position: float,
    bias: int,
    config: Config,
) -> tuple[float, float]:
    """Bid and ask prices: a spread around a reservation price skewed by inventory and bias.

    Never crosses the book, so both quotes would rest as post-only orders.
    """
    tick = tick_size(mid)
    half_spread_bps = max(config.base_half_spread_bps, config.vol_multiplier * vol_bps)
    skew_bps = (
        -config.inventory_skew_bps * position / config.max_position
        + config.pressure_skew_bps * bias
    )
    reservation = mid * (1 + skew_bps / 10_000)
    bid_ticks = math.floor(reservation * (1 - half_spread_bps / 10_000) / tick + EPSILON)
    ask_ticks = math.ceil(reservation * (1 + half_spread_bps / 10_000) / tick - EPSILON)
    bid_ticks = min(bid_ticks, round(best_ask / tick) - 1)
    ask_ticks = max(ask_ticks, round(best_bid / tick) + 1)
    return _price(bid_ticks, tick), _price(ask_ticks, tick)


def tick_size(price: float) -> float:
    """Hyperliquid prices carry at most 5 significant figures."""
    return 10.0 ** (math.floor(math.log10(price)) - 4)


def _price(ticks: int, tick: float) -> float:
    return round(ticks * tick, max(0, -math.floor(math.log10(tick))))


def _better(side: Side, price: float, than: float) -> bool:
    """Whether `price` is a strictly better price than `than` for a resting `side` order."""
    return price > than if side == "buy" else price < than


def edge_bps(fill: Fill, mid: float) -> float:
    """How far `mid` sits in the fill's favour, in bps of the fill price."""
    sign = 1 if fill.side == "buy" else -1
    return sign * (mid - fill.price) / fill.price * 10_000
