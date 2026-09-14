"""Paper market maker: quotes around the mid and simulates its fills against the real tape.

Nothing is sent to the exchange. Fills are inferred from public data:
- a trade through our price, or the opposite side of the book reaching it, fills the whole order;
- a trade at our price first eats the visible size that was queued ahead of us.
Our orders don't move the real market, so results are optimistic for sizes that would.
"""

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Literal

from plom.market import Book, Trade
from plom.pressure import pressure

Side = Literal["buy", "sell"]
SIDES: tuple[Side, Side] = ("buy", "sell")
Key = tuple[Side, int]
"""An order slot: side and layer (0 is nearest the mid)."""
Regime = Literal["calm", "normal", "chaotic"]
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
    microprice_imbalance: float = 0.5
    """Quote around the microprice instead of the mid once top-of-book size imbalance reaches this."""
    microprice_weight: float = 1.0
    """How far to move from the mid towards the microprice: 0 is the mid, 1 is the microprice."""

    pickoff_horizon_ms: int = 1000
    """A fill is judged by how the mid moved this long after it."""
    pickoff_half_life_fills: float = 10.0
    pickoff_decay_s: float = 60.0
    """Half-life of a side's pickoff score while it gets no new fills."""
    pickoff_full_bps: float = 2.0
    """Average adverse markout at which a side counts as fully picked off."""
    pickoff_spread_mult: float = 1.0
    """A fully picked-off side quotes (1 + this) times further out."""
    pickoff_size_cut: float = 0.5
    """A fully picked-off side cuts its inner layer size by this fraction."""

    trend_window_s: float = 10.0
    trend_enter_z: float = 2.0
    """Pull the side being run over once the mid drifts this many expected moves over trend_window_s."""
    trend_exit_z: float = 1.0
    trend_floor_bps: float = 1.0
    """Smallest expected move, so a quiet market doesn't make every tick look like a trend."""

    vol_baseline_half_life_s: float = 600.0
    """Regimes compare recent volatility (vol_half_life_s) with this slower baseline."""
    calm_below: float = 0.7
    """Calm while recent volatility is below this fraction of the baseline."""
    chaotic_above: float = 1.5
    """Chaotic while recent volatility is above this multiple of the baseline."""
    calm_spread_mult: float = 0.8
    calm_size_mult: float = 1.25
    calm_ttl_mult: float = 2.0
    chaotic_spread_mult: float = 2.0
    chaotic_size_mult: float = 0.5
    chaotic_ttl_mult: float = 0.5
    chaotic_layers: int = 2
    chaotic_spacing_mult: float = 1.5
    chaotic_size_growth_mult: float = 1.5
    """Backload size harder in chaos: layers grow faster away from the touch."""

    latency_ms: int = 150
    requote_interval_ms: int = 500
    """Minimum time between requotes, except urgent ones (a jump or a trend change)."""
    requote_move_bps: float = 0.3
    """Requote once the mid has moved this far from where we last quoted..."""
    requote_ttl_ms: int = 5000
    """...or after this long, or after a fill or bias change."""
    jump_bps: float = 3.0
    """A book-to-book mid step, or a trade this far through the mid, counts as a jump."""
    jump_hold_ms: int = 3000
    jump_spread_mult: float = 2.0
    """How much wider all quotes sit for jump_hold_ms after a jump."""
    jump_size_mult: float = 0.5
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
    regime: Regime = "normal"


@dataclass(frozen=True)
class Markout:
    fill: Fill
    horizon_ms: int
    bps: float
    """How far the mid moved in the fill's favour after horizon_ms."""


@dataclass
class _TimeEwma:
    """Exponentially weighted mean over time, corrected for starting from nothing."""

    half_life_s: float
    _sum: float = 0.0
    _weight: float = 0.0

    def add(self, value: float, dt_s: float) -> None:
        alpha = 1 - 0.5 ** (dt_s / self.half_life_s)
        self._sum = (1 - alpha) * self._sum + alpha * value
        self._weight = (1 - alpha) * self._weight + alpha

    @property
    def mean(self) -> float:
        return self._sum / self._weight if self._weight > 0 else 0.0


@dataclass
class MarketMaker:
    config: Config = field(default_factory=Config)
    now_ms: int = 0
    book: Book | None = None
    mid: float | None = None
    fair: float | None = None
    """What we quote around: the mid, pulled towards the microprice when the top of book is lopsided."""
    regime: Regime = "normal"
    pressure: float = 0.0
    bias: int = 0
    """-1, 0 or +1: which way book pressure leans, with hysteresis."""
    trend: int = 0
    """-1, 0 or +1: which way the mid is running, with hysteresis. The side it runs over is pulled."""
    position: float = 0.0
    cash: float = 0.0
    fees: float = 0.0
    volume: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    markouts: list[Markout] = field(default_factory=list)
    orders: dict[Key, Order] = field(default_factory=dict)
    pending: dict[Key, Order] = field(default_factory=dict)
    pulled_ms: dict[Side, int] = field(default_factory=lambda: {"buy": 0, "sell": 0})
    """Time each side spent pulled by a trend or a sweep."""
    swept: dict[Side, bool] = field(default_factory=lambda: {"buy": False, "sell": False})
    """A trade jumped through this side since the last book, so its quotes are pulled until a fresh one."""
    jumps: int = 0
    regime_ms: dict[Regime, int] = field(default_factory=lambda: {"calm": 0, "normal": 0, "chaotic": 0})

    def __post_init__(self) -> None:
        self._quoted_fair: float | None = None
        self._dirty = False
        self._jump_until_ms: int | None = None
        self._vol = _TimeEwma(self.config.vol_half_life_s)
        self._vol_baseline = _TimeEwma(self.config.vol_baseline_half_life_s)
        self._vol_ref: tuple[int, float] | None = None
        self._mids: deque[tuple[int, float]] = deque()
        self._last_book_ms: int | None = None
        self._last_requote_ms: int | None = None
        self._last_fill_ms: dict[Side, int | None] = {"buy": None, "sell": None}
        self._pickoff: dict[Side, tuple[float, int]] = {"buy": (0.0, 0), "sell": (0.0, 0)}
        horizons = {*self.config.markout_horizons_ms, self.config.pickoff_horizon_ms}
        self._unsettled: dict[int, deque[Fill]] = {h: deque() for h in horizons}

    @property
    def pnl(self) -> float:
        return self.cash + self.position * (self.mid or 0.0)

    @property
    def vol_bps(self) -> float:
        """Recent one-second volatility of the mid, in bps."""
        return math.sqrt(self._vol.mean)

    @property
    def vol_ratio(self) -> float:
        """Recent volatility relative to its slow baseline; 1 before there is a baseline."""
        baseline = self._vol_baseline.mean
        return math.sqrt(self._vol.mean / baseline) if baseline > 0 else 1.0

    def pickoff_bps(self, side: Side) -> float:
        """Recent average adverse markout of this side's fills, decayed while it doesn't fill."""
        value, since_ms = self._pickoff[side]
        return value * 0.5 ** ((self.now_ms - since_ms) / 1000 / self.config.pickoff_decay_s)

    def pickoff_score(self, side: Side) -> float:
        """0 when this side's fills aren't followed by adverse moves, 1 when fully picked off."""
        return min(1.0, max(0.0, self.pickoff_bps(side) / self.config.pickoff_full_bps))

    def is_pulled(self, side: Side) -> bool:
        running_over = (self.trend > 0 and side == "sell") or (self.trend < 0 and side == "buy")
        return running_over or self.swept[side]

    @property
    def is_jumping(self) -> bool:
        return self._jump_until_ms is not None and self.now_ms < self._jump_until_ms

    def on_book(self, book: Book) -> None:
        self.now_ms = max(self.now_ms, book.time_ms)
        self.book = book
        self._activate_pending()
        for order in list(self.orders.values()):
            self._match_book(order, book)
        if not book.bids or not book.asks:
            return
        if self._last_book_ms is not None:
            elapsed = self.now_ms - self._last_book_ms
            self.regime_ms[self.regime] += elapsed
            for side in SIDES:
                if self.is_pulled(side):
                    self.pulled_ms[side] += elapsed
        self._last_book_ms = self.now_ms
        was_swept = any(self.swept.values())
        self.swept = {"buy": False, "sell": False}
        previous_mid, trend, bias, regime = self.mid, self.trend, self.bias, self.regime
        self._update_mid(book)
        self.fair = fair_price(book, self.config)
        self.regime = classify_regime(self.vol_ratio, self.config)
        jumped = previous_mid is not None and abs(self.mid / previous_mid - 1) * 10_000 >= self.config.jump_bps
        if jumped:
            self._start_jump()
        self._update_trend()
        self._update_bias(book)
        self._settle_markouts()
        if self.bias != bias or self.regime != regime:
            self._dirty = True
        turned_chaotic = self.regime == "chaotic" and regime != "chaotic"
        self._maybe_requote(urgent=jumped or was_swept or turned_chaotic or self.trend != trend)

    def on_trade(self, trade: Trade) -> None:
        stale = trade.time_ms + 1000 < self.now_ms
        self.now_ms = max(self.now_ms, trade.time_ms)
        self._activate_pending()
        self._fill_from_trade(trade)
        if stale or self.mid is None:
            return
        sign = 1 if trade.side == "buy" else -1
        if sign * (trade.price / self.mid - 1) * 10_000 >= self.config.jump_bps:
            self.swept["buy" if trade.side == "sell" else "sell"] = True
            self._start_jump()
            self._maybe_requote(urgent=True)

    def _fill_from_trade(self, trade: Trade) -> None:
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
        fill = Fill(
            self.now_ms, order.side, order.layer, order.price, size, fee, self.mid or order.price, self.regime
        )
        self.fills.append(fill)
        for queue in self._unsettled.values():
            queue.append(fill)
        self._last_fill_ms[order.side] = self.now_ms
        self._dirty = True
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
        variance_per_s = ((self.mid / ref_mid - 1) * 10_000) ** 2 / dt_s
        self._vol.add(variance_per_s, dt_s)
        self._vol_baseline.add(variance_per_s, dt_s)
        self._vol_ref = (self.now_ms, self.mid)

    def _update_trend(self) -> None:
        c = self.config
        window_ms = c.trend_window_s * 1000
        self._mids.append((self.now_ms, self.mid))
        # Keep one sample at or before the window start to measure the drift from.
        while len(self._mids) > 1 and self._mids[1][0] <= self.now_ms - window_ms:
            self._mids.popleft()
        start_ms, start_mid = self._mids[0]
        if self.now_ms - start_ms < window_ms:
            return
        drift_bps = (self.mid / start_mid - 1) * 10_000
        expected_bps = max(self.vol_bps * math.sqrt(c.trend_window_s), c.trend_floor_bps)
        if self.trend == 0:
            if abs(drift_bps) >= c.trend_enter_z * expected_bps:
                self.trend = 1 if drift_bps > 0 else -1
        elif self.trend * drift_bps < c.trend_exit_z * expected_bps:
            self.trend = 0

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
        c = self.config
        for horizon, queue in self._unsettled.items():
            while queue and queue[0].time_ms + horizon <= self.now_ms:
                fill = queue.popleft()
                markout = Markout(fill, horizon, edge_bps(fill, self.mid))
                self.markouts.append(markout)
                if horizon == c.pickoff_horizon_ms:
                    alpha = 1 - 0.5 ** (1 / c.pickoff_half_life_fills)
                    adverse = (1 - alpha) * self.pickoff_bps(fill.side) + alpha * -markout.bps
                    self._pickoff[fill.side] = (adverse, self.now_ms)
                    self._dirty = True

    def _start_jump(self) -> None:
        self.jumps += 1
        self._jump_until_ms = self.now_ms + self.config.jump_hold_ms

    def _maybe_requote(self, urgent: bool = False) -> None:
        c = self.config
        if self._jump_until_ms is not None and not self.is_jumping:
            self._jump_until_ms = None
            self._dirty = True
        last = self._last_requote_ms
        if not urgent and last is not None:
            if self.now_ms - last < c.requote_interval_ms:
                return
            moved = abs(self.fair / self._quoted_fair - 1) * 10_000 >= c.requote_move_bps
            ttl_mult = {"calm": c.calm_ttl_mult, "normal": 1.0, "chaotic": c.chaotic_ttl_mult}[self.regime]
            if not (self._dirty or moved or self.now_ms - last >= c.requote_ttl_ms * ttl_mult):
                return
        self._requote()

    def _requote(self) -> None:
        self._last_requote_ms = self.now_ms
        self._quoted_fair = self.fair
        self._dirty = False
        jump = self.is_jumping
        c = regime_config(self.config, self.regime)
        half = half_spread_bps(self.vol_bps, c) * (c.jump_spread_mult if jump else 1.0)
        spreads = {side: half * (1 + c.pickoff_spread_mult * self.pickoff_score(side)) for side in SIDES}
        bids, asks = quotes(
            self.fair, self.book.bids[0][0], self.book.asks[0][0],
            skew_bps(self.position, self.bias, c), spreads["buy"], spreads["sell"], c,
        )
        capacity = {"buy": c.max_position - self.position, "sell": c.max_position + self.position}
        live_from = self.now_ms + c.latency_ms
        for side, prices in (("buy", bids), ("sell", asks)):
            if self.is_pulled(side):
                wanted = {}
            else:
                last_fill = self._last_fill_ms[side]
                if last_fill is not None and self.now_ms - last_fill < c.fill_cooldown_ms:
                    self._dirty = True
                    continue
                inner_size = 1 - c.pickoff_size_cut * self.pickoff_score(side)
                size_mult = c.jump_size_mult if jump else 1.0
                wanted = _ladder(prices, c, capacity[side], inner_size, size_mult)
            for key in {k for k in (*self.orders, *self.pending) if k[0] == side}:
                current = self.pending.get(key) or self.orders.get(key)
                if key[1] not in wanted and current.size > EPSILON:
                    self.pending[key] = Order(side, key[1], current.price, 0.0, live_from)
            for layer, (price, size) in wanted.items():
                key = (side, layer)
                current = self.pending.get(key) or self.orders.get(key)
                if current is None or current.size <= EPSILON or current.price != price:
                    self.pending[key] = Order(side, layer, price, size, live_from)


def fair_price(book: Book, config: Config) -> float:
    """The mid, moved towards the size-weighted microprice when top-of-book sizes are lopsided."""
    (bid, bid_size), (ask, ask_size) = book.bids[0], book.asks[0]
    mid = (bid + ask) / 2
    total = bid_size + ask_size
    if total <= 0 or abs(bid_size - ask_size) / total < config.microprice_imbalance:
        return mid
    microprice = (bid * ask_size + ask * bid_size) / total
    return mid + config.microprice_weight * (microprice - mid)


def classify_regime(vol_ratio: float, config: Config) -> Regime:
    if vol_ratio >= config.chaotic_above:
        return "chaotic"
    if vol_ratio < config.calm_below:
        return "calm"
    return "normal"


def regime_config(config: Config, regime: Regime) -> Config:
    """The config to quote with in a regime: calm is tighter and bigger; chaos is wider, smaller and deeper."""
    match regime:
        case "calm":
            return replace(
                config,
                base_half_spread_bps=config.base_half_spread_bps * config.calm_spread_mult,
                order_size=config.order_size * config.calm_size_mult,
            )
        case "chaotic":
            return replace(
                config,
                base_half_spread_bps=config.base_half_spread_bps * config.chaotic_spread_mult,
                order_size=config.order_size * config.chaotic_size_mult,
                layers=min(config.layers, config.chaotic_layers),
                layer_spacing=config.layer_spacing * config.chaotic_spacing_mult,
                size_growth=config.size_growth * config.chaotic_size_growth_mult,
            )
        case _:
            return config


def half_spread_bps(vol_bps: float, config: Config) -> float:
    """Distance of the inner layer from the reservation price before any side-specific widening."""
    return max(config.base_half_spread_bps, config.vol_multiplier * vol_bps)


def skew_bps(position: float, bias: int, config: Config) -> float:
    """How far the reservation price sits above the mid: against inventory, towards book pressure."""
    return -config.inventory_skew_bps * position / config.max_position + config.pressure_skew_bps * bias


def quotes(
    mid: float,
    best_bid: float,
    best_ask: float,
    skew_bps: float,
    bid_half_spread_bps: float,
    ask_half_spread_bps: float,
    config: Config,
) -> tuple[list[float], list[float]]:
    """Bid and ask prices per layer, inner first, around the mid shifted by skew_bps.

    Never crosses the book or our own quotes, and no two layers on a side share a price.
    """
    tick = tick_size(mid)
    reservation = mid * (1 + skew_bps / 10_000)
    bid_ticks: list[int] = []
    ask_ticks: list[int] = []
    for layer in range(config.layers):
        spacing = config.layer_spacing**layer / 10_000
        bid = math.floor(reservation * (1 - bid_half_spread_bps * spacing) / tick + EPSILON)
        ask = math.ceil(reservation * (1 + ask_half_spread_bps * spacing) / tick - EPSILON)
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


def _ladder(
    prices: list[float],
    config: Config,
    capacity: float,
    inner_size: float = 1.0,
    size_mult: float = 1.0,
) -> dict[int, tuple[float, float]]:
    """Price and size per layer, growing outwards, filling inner layers first until capacity runs out.

    inner_size scales the inner layer only; size_mult scales every layer.
    """
    wanted = {}
    for layer, price in enumerate(prices):
        if capacity <= EPSILON:
            break
        scale = (inner_size if layer == 0 else config.size_growth**layer) * size_mult
        size = min(config.order_size * scale, capacity)
        if size <= EPSILON:
            continue
        capacity -= size
        wanted[layer] = (price, size)
    return wanted


def _price(ticks: int, tick: float) -> float:
    return round(ticks * tick, max(0, -math.floor(math.log10(tick))))


def _better(side: Side, price: float, than: float) -> bool:
    """Whether `price` is a strictly better price than `than` for a resting `side` order."""
    return price > than if side == "buy" else price < than
