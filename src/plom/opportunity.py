"""How much a lagging venue's quotes stand to lose, or win, around the composite's sharp moves.

Before building a strategy on a venue that trails the composite, this measures the opportunity from
the venue's own recorded book and trades. The composite is timed by our local receive time; the
venue by its own timestamps plus its fastest typical delivery delay, since venues that batch messages
deliver some events long after they happen. Our actions are delayed by an assumed latency:

- Stale side: a quote resting at the venue's touch on the side the move runs towards (the ask when
  the composite jumps up). If a trade reaches it before our cancel lands, it fills. Its loss is how
  far the venue's mid has moved past it `markout_ms` after the move.
- Favorable side: a quote we join at the touch on the other side, landing after the latency. It
  fills only if trades there within `window_ms` exceed the size already queued ahead of us. Its
  edge is how far the venue's mid moved in our favour by the markout, less the maker fee.

Both are upper-bound-ish in different ways: the stale fill ignores queue position (we assume we're
first, as a resting quote would often be), and neither accounts for our own orders moving anything.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plom import composite, recording
from plom.market import Book, Trade
from plom.venues import VENUES

MOVE_WINDOW_MS = 250
MOVE_GAP_MS = 2000


@dataclass(frozen=True)
class Outcome:
    latency_ms: int
    events: int
    stale_fill_rate: float
    """Share of moves in which a trade hit a quote left at the stale touch before the cancel landed."""
    stale_loss_bps: float
    """Mean loss of those stale fills at the markout."""
    good_fill_rate: float
    """Share of moves in which a quote joined at the favorable touch would have filled."""
    good_edge_bps: float
    """Mean gross edge of those fills at the markout, before fees."""

    def per_event_bps(self, maker_fee_bps: float) -> tuple[float, float]:
        """Expected bps per move: (lost by leaving the stale quote, won by joining the good side after fees)."""
        return (
            -self.stale_fill_rate * (self.stale_loss_bps + maker_fee_bps),
            self.good_fill_rate * (self.good_edge_bps - maker_fee_bps),
        )


@dataclass(frozen=True)
class Report:
    venue: str
    hours: float
    threshold_bps: float
    outcomes: list[Outcome]


def measure(
    path: Path, venue: str, threshold_bps: float = 1.5, latencies_ms: Sequence[int] = (50, 150, 300),
    window_ms: int = 500, markout_ms: int = 2000,
) -> Report:
    comp = composite.Composite(tuple(v for v in composite.VENUES if v != venue))
    parsers = {v: VENUES[v].Parser() for v in (*comp.venues, venue)}
    ref_t, ref_px = [], []
    book_t, book_recv, bid, bid_sz, ask, ask_sz = [], [], [], [], [], []
    trade_t, trade_px, trade_sz, trade_buy = [], [], [], []
    for source, recv_ms, message in recording.read(path):
        if source not in parsers or recv_ms is None:
            continue
        for event in parsers[source].events(message):
            if source == venue:
                if isinstance(event, Book) and event.bids and event.asks:
                    book_t.append(event.time_ms)
                    book_recv.append(recv_ms)
                    bid.append(event.bids[0][0]); bid_sz.append(event.bids[0][1])
                    ask.append(event.asks[0][0]); ask_sz.append(event.asks[0][1])
                elif isinstance(event, Trade):
                    trade_t.append(event.time_ms); trade_px.append(event.price)
                    trade_sz.append(event.size); trade_buy.append(event.side == "buy")
            elif isinstance(event, Book) and (merged := comp.on_book(source, recv_ms, event)) is not None:
                ref_t.append(recv_ms)
                ref_px.append(merged.bids[0][0])
    if not ref_t or not book_t:
        raise ValueError(f"need composite venues and {venue} in {path}")
    ref_t, ref_px = np.array(ref_t), np.array(ref_px)
    # Venues that batch their messages deliver some events hundreds of ms after they happened, so
    # time the venue by its own timestamps, put on our clock by the fastest typical delivery.
    offset = float(np.percentile(np.array(book_recv) - np.array(book_t), 5))
    book = {k: np.array(v) for k, v in dict(t=book_t, bid=bid, bid_sz=bid_sz, ask=ask, ask_sz=ask_sz).items()}
    trades = {k: np.array(v) for k, v in dict(t=trade_t, px=trade_px, sz=trade_sz, buy=trade_buy).items()}
    book["t"] = book["t"] + offset
    trades["t"] = trades["t"] + offset
    order = np.argsort(book["t"], kind="stable")
    book = {k: v[order] for k, v in book.items()}
    order = np.argsort(trades["t"], kind="stable")
    trades = {k: v[order] for k, v in trades.items()}
    events = sharp_moves(ref_t, ref_px, threshold_bps)
    hours = (ref_t[-1] - ref_t[0]) / 3_600_000
    outcomes = [_outcome(events, book, trades, latency, window_ms, markout_ms) for latency in latencies_ms]
    return Report(venue, hours, threshold_bps, outcomes)


def sharp_moves(times: np.ndarray, prices: np.ndarray, threshold_bps: float) -> list[tuple[float, int]]:
    """(time, direction) of each move of at least threshold_bps within MOVE_WINDOW_MS, one per MOVE_GAP_MS."""
    before = np.searchsorted(times, times - MOVE_WINDOW_MS, side="left")
    move_bps = (prices / prices[before] - 1) * 10_000
    events: list[tuple[float, int]] = []
    for i in np.flatnonzero(np.abs(move_bps) >= threshold_bps):
        if not events or times[i] - events[-1][0] >= MOVE_GAP_MS:
            events.append((float(times[i]), 1 if move_bps[i] > 0 else -1))
    return events


def _at(series: dict[str, np.ndarray], t: float) -> int | None:
    i = int(np.searchsorted(series["t"], t, side="right")) - 1
    return i if i >= 0 else None


def _outcome(events, book, trades, latency_ms, window_ms, markout_ms) -> Outcome:
    stale_losses, good_edges, counted = [], [], 0
    for t, up in events:
        now, later = _at(book, t), _at(book, t + markout_ms)
        if now is None or later is None or book["t"][-1] < t + markout_ms:
            continue
        counted += 1
        mid_later = (book["bid"][later] + book["ask"][later]) / 2
        # Stale: our quote at the touch the move runs towards, until the cancel lands.
        stale_px = book["ask"][now] if up > 0 else book["bid"][now]
        lo, hi = np.searchsorted(trades["t"], [t, t + latency_ms], side="left")
        hit = trades["buy"][lo:hi] if up > 0 else ~trades["buy"][lo:hi]
        reached = trades["px"][lo:hi] >= stale_px if up > 0 else trades["px"][lo:hi] <= stale_px
        if np.any(hit & reached):
            stale_losses.append(up * (mid_later - stale_px) / stale_px * 10_000)
        # Favorable: join the other touch once the order lands, behind what is already queued there.
        joined = _at(book, t + latency_ms)
        if joined is None:
            continue
        good_px = book["bid"][joined] if up > 0 else book["ask"][joined]
        queue = book["bid_sz"][joined] if up > 0 else book["ask_sz"][joined]
        lo, hi = np.searchsorted(trades["t"], [t + latency_ms, t + window_ms], side="left")
        against = ~trades["buy"][lo:hi] if up > 0 else trades["buy"][lo:hi]
        through = trades["px"][lo:hi] <= good_px if up > 0 else trades["px"][lo:hi] >= good_px
        if trades["sz"][lo:hi][against & through].sum() > queue:
            good_edges.append(up * (mid_later - good_px) / good_px * 10_000)
    n = max(counted, 1)
    return Outcome(
        latency_ms, counted,
        len(stale_losses) / n, float(np.mean(stale_losses)) if stale_losses else 0.0,
        len(good_edges) / n, float(np.mean(good_edges)) if good_edges else 0.0,
    )


def format_report(report: Report, fees_bps: Sequence[float]) -> str:
    per_hour = sum(o.events for o in report.outcomes[:1]) / report.hours if report.hours else 0.0
    lines = [
        f"--- {report.venue}: {report.outcomes[0].events if report.outcomes else 0} composite moves >= "
        f"{report.threshold_bps:g}bps in {report.hours:.1f}h ({per_hour:.1f}/h) ---",
        "latency  stale fill  stale loss   good fill  good edge   expected bps per move (stale / good) at maker fee "
        + ", ".join(f"{f:g}" for f in fees_bps),
    ]
    for o in report.outcomes:
        cells = "   ".join("{:+.2f} / {:+.2f}".format(*o.per_event_bps(fee)) for fee in fees_bps)
        lines.append(
            f"{o.latency_ms:>5}ms   {o.stale_fill_rate:>8.0%}  {o.stale_loss_bps:>+8.2f}bps  "
            f"{o.good_fill_rate:>8.0%}  {o.good_edge_bps:>+8.2f}bps   {cells}"
        )
    return "\n".join(lines)
