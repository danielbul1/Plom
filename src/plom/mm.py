"""Paper market maker: quotes around the mid and simulates its fills against the real tape.

Nothing is sent to the exchange. Fills are inferred from public data, conservatively:
- a trade through our price fills us up to the trade's size; the opposite side of the book reaching
  our price fills us up to the size resting there;
- a trade at our price first eats the visible size queued ahead of us;
- an order rests only after order_latency_ms, and a replaced or cancelled one stays fillable for
  cancel_latency_ms.
Our orders don't move the real market, so results are optimistic for sizes that would.
"""

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Literal

from plom.alpha import AlphaConfig, OnlineAlpha
from plom.glft import FillIntensity, Glft, glft
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
    glft_gamma: float = 0.0
    """GLFT risk aversion, per bp per lot of order_size. Above 0, once fill intensity is calibrated, the
    half spread and inventory skew come from GLFT (see plom.glft) instead of vol_multiplier and
    inventory_skew_bps; base_half_spread_bps stays a floor. 0 calibrates and reports without using it."""
    glft_layer_spacing: float = 1.0
    """With GLFT, each layer sits this many GLFT half spreads beyond the one inside it."""
    glft_sample_ms: int = 100
    glft_half_life_s: float = 600.0
    glft_step_bps: float = 0.25
    glft_buckets: int = 40
    glft_min_hits: float = 10.0
    position_age_s: float = math.inf
    """The inventory skew grows by its own size for every position_age_s the position has been open."""
    flatten_age_s: float = math.inf
    """Once the position has been open this long, cross the spread to cut it by flatten_fraction."""
    flatten_fraction: float = 1.0
    flatten_cooldown_ms: int = 5000
    pressure_skew_bps: float = 1.0
    """How far quotes shift towards the book-pressure bias."""
    pressure_enter: float = 0.3
    pressure_exit: float = 0.1
    microprice_imbalance: float = 0.5
    """Quote around the microprice instead of the mid once top-of-book size imbalance reaches this."""
    microprice_weight: float = 1.0
    """How far to move from the mid towards the microprice: 0 is the mid, 1 is the microprice."""

    reference_weight: float = 0.5
    """With a reference venue, move fair value this far from the local price towards the reference
    mid adjusted by the learned basis: 0 ignores the reference, 1 quotes around it."""
    basis_half_life_s: float = 300.0
    """How fast the learned venue-minus-reference basis adapts."""
    reference_stale_ms: int = 1000
    """Ignore the reference when its last book is older than this."""
    reference_jump_bps: float = 2.0
    """A reference move this large within reference_jump_window_ms counts as a jump and pulls the side
    it runs towards until the venue's next book."""
    reference_jump_window_ms: int = 250

    alpha_horizon_ms: int = 1000
    """The online forecast predicts the mid's move over this horizon (see plom.alpha)."""
    alpha_sample_ms: int = 100
    alpha_half_life_samples: float = 3000.0
    alpha_ridge: float = 1.0
    alpha_warmup_samples: int = 600
    flow_half_life_ms: int = 1000
    alpha_weight: float = 0.0
    """Shift fair value by this fraction of the forecast move; 0 learns and reports without using it."""
    alpha_widen: float = 0.0
    """Widen a side's half spread by this multiple of the forecast move against it."""
    alpha_pull_margin_bps: float = math.inf
    """Pull a side while the forecast move against it exceeds its half spread plus maker fee plus this."""

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

    tick_size: float = 0.0
    """Price increment; 0 uses Hyperliquid's five-significant-figure rule."""
    order_latency_ms: int = 150
    """From deciding to quote until the order rests on the book."""
    cancel_latency_ms: int = 150
    """From deciding to cancel or replace until the old order stops being fillable."""
    tx_per_minute: float = math.inf
    """Placements, replacements and cancels allowed per minute; actions beyond it are skipped."""
    queue_power: float = 0.0
    """0 assumes size leaving our level without trading left from behind us. Above 0, it is split
    ahead of and behind us in proportion front**n : back**n, as in hftbacktest's power queue model."""
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
    taker_fee_bps: float = 0.0
    markout_horizons_ms: tuple[int, ...] = (100, 1000, 5000, 30_000, 60_000, 300_000)


@dataclass
class Order:
    side: Side
    layer: int
    price: float
    size: float
    """Remaining size. A pending order with size 0 is a cancel."""
    live_from_ms: int
    dead_from_ms: float = math.inf
    """When a cancel or replacement takes effect; the order is fillable in [live_from_ms, dead_from_ms)."""
    queue_ahead: float = 0.0
    level_size: float = 0.0
    """Visible size at our price in the last book, to tell cancels from trades."""
    traded_here: float = 0.0
    """Size traded at our price since the last book."""

    def is_live(self, time_ms: float) -> bool:
        return self.live_from_ms <= time_ms < self.dead_from_ms and self.size > EPSILON


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
    """What we quote around: the mid pulled towards the microprice when the top of book is lopsided,
    then towards the basis-adjusted reference when there is one."""
    local_fair: float | None = None
    reference_mid: float | None = None
    reference_ms: int | None = None
    reference_jumps: int = 0
    alpha_pulls: dict[Side, int] = field(default_factory=lambda: {"buy": 0, "sell": 0})
    """Requotes at which the forecast pulled a side."""
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
    spread_capture: float = 0.0
    """Sum of each fill's edge against the mid at fill time, in quote currency."""
    inventory_pnl: float = 0.0
    """PnL from holding the position through mid moves. pnl = spread_capture + inventory_pnl - fees."""
    fills: list[Fill] = field(default_factory=list)
    """Maker fills."""
    flattens: list[Fill] = field(default_factory=list)
    """Taker fills from crossing the spread to flatten an old position, with layer -1."""
    markouts: list[Markout] = field(default_factory=list)
    orders: dict[Key, Order] = field(default_factory=dict)
    pending: dict[Key, Order] = field(default_factory=dict)
    retiring: list[Order] = field(default_factory=list)
    """Replaced orders that are still fillable until their cancel takes effect."""
    tx_sent: int = 0
    tx_skipped: int = 0
    pulled_ms: dict[Side, int] = field(default_factory=lambda: {"buy": 0, "sell": 0})
    """Time each side spent pulled by a trend or a sweep."""
    swept: dict[Side, bool] = field(default_factory=lambda: {"buy": False, "sell": False})
    """A trade jumped through this side since the last book, so its quotes are pulled until a fresh one."""
    jumps: int = 0
    regime_ms: dict[Regime, int] = field(default_factory=lambda: {"calm": 0, "normal": 0, "chaotic": 0})

    def __post_init__(self) -> None:
        self._quoted_fair: float | None = None
        self._tx_tokens = self.config.tx_per_minute
        self._tx_refilled_ms: int | None = None
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
        self._basis = _TimeEwma(self.config.basis_half_life_s)
        c = self.config
        self.alpha = OnlineAlpha(AlphaConfig(
            c.alpha_horizon_ms, c.alpha_sample_ms, c.alpha_half_life_samples, c.alpha_ridge,
            c.alpha_warmup_samples, c.flow_half_life_ms,
        ))
        self._quoted_prediction = 0.0
        self.intensity = FillIntensity(
            c.glft_sample_ms, c.glft_half_life_s, c.glft_step_bps, c.glft_buckets, c.glft_min_hits
        )
        self._position_since_ms: int | None = None
        self._flatten: tuple[Side, float, int] | None = None
        """A pending taker order: side, size, and when it lands."""
        self._last_flatten_ms: int | None = None
        self._basis_ms: int | None = None
        self._reference_mids: deque[tuple[int, float]] = deque()
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

    @property
    def glft(self) -> Glft | None:
        """GLFT half spread and skew at current volatility; None while off or not yet calibrated."""
        if self.config.glft_gamma <= 0 or self.intensity.k is None:
            return None
        c = self.config
        return glft(self.vol_bps, self.intensity.a, self.intensity.k, c.glft_gamma, c.maker_fee_bps)

    @property
    def position_age_s(self) -> float:
        """How long the position has been open: since it was last flat or changed sign."""
        return 0.0 if self._position_since_ms is None else (self.now_ms - self._position_since_ms) / 1000

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
    def basis(self) -> float | None:
        """Learned log(venue mid / reference mid)."""
        return self._basis.mean if self._basis_ms is not None else None

    @property
    def reference_fresh(self) -> bool:
        return self.reference_ms is not None and self.now_ms - self.reference_ms <= self.config.reference_stale_ms

    @property
    def is_jumping(self) -> bool:
        return self._jump_until_ms is not None and self.now_ms < self._jump_until_ms

    def on_book(self, book: Book) -> None:
        self.now_ms = max(self.now_ms, book.time_ms)
        self.book = book
        self._activate_pending()
        for order in self._resting():
            self._match_book(order, book)
        self._execute_flatten(book)
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
        self.local_fair = fair_price(book, self.config)
        self._update_basis()
        self.alpha.on_book(self.now_ms, book, self.mid, self._gap_bps())
        if abs(self.alpha.prediction_bps - self._quoted_prediction) >= self.config.requote_move_bps:
            self._dirty = True
        self._update_fair()
        self.intensity.on_book(self.now_ms, self.fair)
        self.regime = classify_regime(self.vol_ratio, self.config)
        jumped = previous_mid is not None and abs(self.mid / previous_mid - 1) * 10_000 >= self.config.jump_bps
        if jumped:
            self._start_jump()
        self._update_trend()
        self._update_bias(book)
        self._settle_markouts()
        if self.bias != bias or self.regime != regime:
            self._dirty = True
        self._maybe_flatten()
        turned_chaotic = self.regime == "chaotic" and regime != "chaotic"
        self._maybe_requote(urgent=jumped or was_swept or turned_chaotic or self.trend != trend)

    def on_reference(self, book: Book) -> None:
        """A book from the reference venue, with time_ms already translated to this venue's clock."""
        if not book.bids or not book.asks:
            return
        self.now_ms = max(self.now_ms, book.time_ms)
        self._activate_pending()
        self.reference_mid = (book.bids[0][0] + book.asks[0][0]) / 2
        self.reference_ms = self.now_ms
        c = self.config
        window = self._reference_mids
        window.append((self.now_ms, self.reference_mid))
        while len(window) > 1 and window[1][0] <= self.now_ms - c.reference_jump_window_ms:
            window.popleft()
        move_bps = (self.reference_mid / window[0][1] - 1) * 10_000
        jumped = abs(move_bps) >= c.reference_jump_bps
        if jumped:
            self.reference_jumps += 1
            self.swept["sell" if move_bps > 0 else "buy"] = True
            self._start_jump()
            window.clear()
            window.append((self.now_ms, self.reference_mid))
        if self.book is None or self.mid is None:
            return
        self._update_fair()
        self._maybe_requote(urgent=jumped)

    def _update_basis(self) -> None:
        if not self.reference_fresh:
            return
        dt_s = 0.0 if self._basis_ms is None else (self.now_ms - self._basis_ms) / 1000
        if self._basis_ms is None:
            self._basis.add(math.log(self.mid / self.reference_mid), 1.0)
        elif dt_s > 0:
            self._basis.add(math.log(self.mid / self.reference_mid), dt_s)
        self._basis_ms = self.now_ms

    def _gap_bps(self) -> float:
        """Basis-adjusted reference minus local fair price, in bps; 0 without a fresh reference."""
        if self.basis is None or not self.reference_fresh or self.local_fair is None:
            return 0.0
        return (self.reference_mid * math.exp(self.basis) / self.local_fair - 1) * 10_000

    def _update_fair(self) -> None:
        if self.local_fair is None:
            return
        gap = self._gap_bps() * self.config.reference_weight + self.alpha.prediction_bps * self.config.alpha_weight
        self.fair = self.local_fair * (1 + gap / 10_000)

    def on_trade(self, trade: Trade) -> None:
        stale = trade.time_ms + 1000 < self.now_ms
        self.now_ms = max(self.now_ms, trade.time_ms)
        self._activate_pending()
        self._fill_from_trade(trade)
        self.alpha.on_trade(self.now_ms, trade)
        if stale or self.mid is None:
            return
        self.intensity.on_trade(trade)
        sign = 1 if trade.side == "buy" else -1
        if sign * (trade.price / self.mid - 1) * 10_000 >= self.config.jump_bps:
            self.swept["buy" if trade.side == "sell" else "sell"] = True
            self._start_jump()
            self._maybe_requote(urgent=True)

    def _resting(self, side: Side | None = None, time_ms: float | None = None) -> list[Order]:
        """Orders fillable at time_ms (default now), best price first when a side is given."""
        at = self.now_ms if time_ms is None else time_ms
        orders = [o for o in (*self.orders.values(), *self.retiring) if o.is_live(at) and side in (None, o.side)]
        return sorted(orders, key=lambda o: o.price, reverse=side == "buy")

    def _fill_from_trade(self, trade: Trade) -> None:
        resting: Side = "buy" if trade.side == "sell" else "sell"
        remaining = trade.size
        for order in self._resting(resting, trade.time_ms):
            if remaining <= EPSILON:
                break
            if _better(resting, order.price, trade.price):
                filled = min(order.size, remaining)
            elif trade.price == order.price:
                order.traded_here += trade.size
                order.queue_ahead -= trade.size
                if order.queue_ahead >= 0:
                    continue
                filled = min(order.size, -order.queue_ahead, remaining)
                order.queue_ahead = 0.0
            else:
                continue
            self._fill(order, filled)
            remaining -= filled

    def _activate_pending(self) -> None:
        if self.book is None:
            return
        self.retiring = [o for o in self.retiring if o.is_live(self.now_ms)]
        for key, order in list(self.pending.items()):
            if order.live_from_ms > self.now_ms:
                continue
            del self.pending[key]
            current = self.orders.pop(key, None)
            if current is not None and current.is_live(self.now_ms):
                self.retiring.append(current)
            if order.size <= EPSILON:
                continue
            levels = self.book.bids if order.side == "buy" else self.book.asks
            order.queue_ahead = order.level_size = dict(levels).get(order.price, 0.0)
            self.orders[key] = order
        for key, order in list(self.orders.items()):
            if order.dead_from_ms <= self.now_ms:
                del self.orders[key]

    def _match_book(self, order: Order, book: Book) -> None:
        own = book.bids if order.side == "buy" else book.asks
        opposite = book.asks if order.side == "buy" else book.bids
        crossing = sum(size for price, size in opposite if not _better(order.side, price, order.price))
        if crossing > EPSILON:
            self._fill(order, min(order.size, crossing))
            if order.size <= EPSILON:
                return
        visible = dict(own)
        if order.price in visible:
            new_size = visible[order.price]
            left_without_trading = order.level_size - order.traded_here - new_size
            if self.config.queue_power > 0 and left_without_trading > 0:
                order.queue_ahead = _queue_after_cancels(
                    order.queue_ahead, order.level_size - order.traded_here, left_without_trading,
                    self.config.queue_power,
                )
            order.queue_ahead = min(order.queue_ahead, new_size)
            order.level_size = new_size
        elif own and not _better(order.side, own[-1][0], order.price):
            # Inside the visible range but the level is gone: nobody is ahead of us.
            order.queue_ahead = order.level_size = 0.0
        order.traded_here = 0.0

    def _fill(self, order: Order, size: float) -> None:
        if size <= EPSILON:
            return
        fill = self._trade(order.side, order.layer, order.price, size, self.config.maker_fee_bps)
        self.fills.append(fill)
        for queue in self._unsettled.values():
            queue.append(fill)
        self._last_fill_ms[order.side] = self.now_ms
        self._dirty = True
        order.size -= size
        if order.size <= EPSILON:
            key = (order.side, order.layer)
            if self.orders.get(key) is order:
                del self.orders[key]
            elif order in self.retiring:
                self.retiring.remove(order)

    def _trade(self, side: Side, layer: int, price: float, size: float, fee_bps: float) -> Fill:
        """Book a fill into position, cash, fees and PnL attribution."""
        notional = price * size
        fee = notional * fee_bps / 10_000
        sign = 1 if side == "buy" else -1
        previous = self.position
        self.position += sign * size
        self.cash -= sign * notional + fee
        self.fees += fee
        self.volume += notional
        self.spread_capture += sign * ((self.mid or price) - price) * size
        if abs(self.position) <= EPSILON:
            self._position_since_ms = None
        elif self._position_since_ms is None or previous * self.position < 0:
            self._position_since_ms = self.now_ms
        return Fill(self.now_ms, side, layer, price, size, fee, self.mid or price, self.regime)

    def _maybe_flatten(self) -> None:
        c = self.config
        if self._flatten is not None or self.position_age_s < c.flatten_age_s:
            return
        if self._last_flatten_ms is not None and self.now_ms - self._last_flatten_ms < c.flatten_cooldown_ms:
            return
        if not self._spend_tx():
            return
        side: Side = "sell" if self.position > 0 else "buy"
        self._flatten = (side, abs(self.position) * c.flatten_fraction, self.now_ms + c.order_latency_ms)
        self._last_flatten_ms = self.now_ms

    def _execute_flatten(self, book: Book) -> None:
        """Fill a landed flatten against the book, walking levels up to their visible size."""
        if self._flatten is None or self._flatten[2] > self.now_ms:
            return
        side, size, _ = self._flatten
        self._flatten = None
        still_reduces = (side == "sell") == (self.position > 0)
        size = min(size, abs(self.position)) if still_reduces else 0.0
        for price, available in book.asks if side == "buy" else book.bids:
            if size <= EPSILON:
                break
            taken = min(size, available)
            self.flattens.append(self._trade(side, -1, price, taken, self.config.taker_fee_bps))
            size -= taken
        self._dirty = True

    def _update_mid(self, book: Book) -> None:
        mid = (book.bids[0][0] + book.asks[0][0]) / 2
        if self.mid is not None:
            self.inventory_pnl += self.position * (mid - self.mid)
        self.mid = mid
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
        self._quoted_prediction = self.alpha.prediction_bps
        self._dirty = False
        jump = self.is_jumping
        c = regime_config(self.config, self.regime)
        model = self.glft
        half = half_spread_bps(self.vol_bps, c, model) * (c.jump_spread_mult if jump else 1.0)
        spreads = {side: half * (1 + c.pickoff_spread_mult * self.pickoff_score(side)) for side in SIDES}
        against = {"buy": -self.alpha.prediction_bps, "sell": self.alpha.prediction_bps}
        for side in SIDES:
            spreads[side] += c.alpha_widen * max(0.0, against[side])
        alpha_pulled = {
            side: against[side] > spreads[side] + c.maker_fee_bps + c.alpha_pull_margin_bps for side in SIDES
        }
        age_mult = 1 + self.position_age_s / c.position_age_s
        bids, asks = quotes(
            self.fair, self.book.bids[0][0], self.book.asks[0][0],
            skew_bps(self.position, self.bias, self.config, model, age_mult), spreads["buy"], spreads["sell"], c,
            c.glft_layer_spacing * half if model else None,
        )
        capacity = {"buy": c.max_position - self.position, "sell": c.max_position + self.position}
        live_from = self.now_ms + c.order_latency_ms
        cancel_at = self.now_ms + c.cancel_latency_ms
        for side, prices in (("buy", bids), ("sell", asks)):
            if alpha_pulled[side]:
                self.alpha_pulls[side] += 1
            if self.is_pulled(side) or alpha_pulled[side]:
                wanted = {}
            else:
                last_fill = self._last_fill_ms[side]
                if last_fill is not None and self.now_ms - last_fill < c.fill_cooldown_ms:
                    self._dirty = True
                    continue
                inner_size = 1 - c.pickoff_size_cut * self.pickoff_score(side)
                size_mult = c.jump_size_mult if jump else 1.0
                wanted = _ladder(prices, c, capacity[side], inner_size, size_mult)
            for key in sorted({k for k in (*self.orders, *self.pending) if k[0] == side}):
                current = self.pending.get(key) or self.orders.get(key)
                if key[1] not in wanted and current.size > EPSILON and self._spend_tx():
                    self._retire(key, cancel_at)
                    self.pending[key] = Order(side, key[1], current.price, 0.0, cancel_at)
            for layer, (price, size) in wanted.items():
                key = (side, layer)
                current = self.pending.get(key) or self.orders.get(key)
                if current is None or current.size <= EPSILON or current.price != price:
                    if not self._spend_tx():
                        continue
                    self._retire(key, cancel_at)
                    self.pending[key] = Order(side, layer, price, size, live_from)

    def _retire(self, key: Key, at_ms: int) -> None:
        """Schedule the order resting in this slot to stop being fillable at at_ms."""
        order = self.orders.get(key)
        if order is not None:
            order.dead_from_ms = min(order.dead_from_ms, at_ms)

    def _spend_tx(self) -> bool:
        """Take one transaction from a token bucket refilled at tx_per_minute; False if it's empty."""
        c = self.config
        if not math.isinf(c.tx_per_minute):
            if self._tx_refilled_ms is not None:
                refill = (self.now_ms - self._tx_refilled_ms) * c.tx_per_minute / 60_000
                self._tx_tokens = min(c.tx_per_minute, self._tx_tokens + refill)
            self._tx_refilled_ms = self.now_ms
            if self._tx_tokens < 1:
                self.tx_skipped += 1
                self._dirty = True
                return False
            self._tx_tokens -= 1
        self.tx_sent += 1
        return True


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
                glft_layer_spacing=config.glft_layer_spacing * config.chaotic_spacing_mult,
                size_growth=config.size_growth * config.chaotic_size_growth_mult,
            )
        case _:
            return config


def half_spread_bps(vol_bps: float, config: Config, model: Glft | None = None) -> float:
    """Distance of the inner layer from the reservation price before any side-specific widening."""
    wanted = model.half_spread_bps if model else config.vol_multiplier * vol_bps
    return max(config.base_half_spread_bps, wanted)


def skew_bps(position: float, bias: int, config: Config, model: Glft | None = None, age_mult: float = 1.0) -> float:
    """How far the reservation price sits above the mid: against inventory, towards book pressure.

    With GLFT the inventory part is its per-lot skew times the position in lots of order_size.
    age_mult scales the inventory part as the position ages.
    """
    if model:
        inventory = model.skew_bps * position / config.order_size
    else:
        inventory = config.inventory_skew_bps * position / config.max_position
    return -inventory * age_mult + config.pressure_skew_bps * bias


def quotes(
    mid: float,
    best_bid: float,
    best_ask: float,
    skew_bps: float,
    bid_half_spread_bps: float,
    ask_half_spread_bps: float,
    config: Config,
    layer_step_bps: float | None = None,
) -> tuple[list[float], list[float]]:
    """Bid and ask prices per layer, inner first, around the mid shifted by skew_bps.

    Layers sit layer_spacing times further out than the one inside them, or, given layer_step_bps,
    that much further out. Never crosses the book or our own quotes, and no two layers on a side
    share a price.
    """
    tick = config.tick_size or tick_size(mid)
    reservation = mid * (1 + skew_bps / 10_000)
    bid_ticks: list[int] = []
    ask_ticks: list[int] = []
    for layer in range(config.layers):
        if layer_step_bps is None:
            bid_depth = bid_half_spread_bps * config.layer_spacing**layer
            ask_depth = ask_half_spread_bps * config.layer_spacing**layer
        else:
            bid_depth = bid_half_spread_bps + layer * layer_step_bps
            ask_depth = ask_half_spread_bps + layer * layer_step_bps
        bid = math.floor(reservation * (1 - bid_depth / 10_000) / tick + EPSILON)
        ask = math.ceil(reservation * (1 + ask_depth / 10_000) / tick - EPSILON)
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


def _queue_after_cancels(front: float, level: float, cancelled: float, power: float) -> float:
    """Queue ahead after `cancelled` size left a level of `level` without trading.

    The share of it that came from behind us is back**n / (back**n + front**n).
    """
    back = max(0.0, level - front)
    if front + back <= 0:
        return 0.0
    behind = back**power / (back**power + front**power)
    estimate = front - (1 - behind) * cancelled + min(back - behind * cancelled, 0.0)
    return max(0.0, estimate)


def _price(ticks: int, tick: float) -> float:
    return round(ticks * tick, max(0, -math.floor(math.log10(tick))))


def _better(side: Side, price: float, than: float) -> bool:
    """Whether `price` is a strictly better price than `than` for a resting `side` order."""
    return price > than if side == "buy" else price < than
