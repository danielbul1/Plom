"""Measure how far each venue's mid lags a reference venue, and how well the gap predicts catch-up.

Everything is timed by our local receive time, the only clock shared across venues, so the lags
include each feed's delivery latency to us — which is what a strategy running here would face.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plom import composite, recording
from plom.market import Book
from plom.venues import VENUES

GRID_MS = 25
LAGS_MS = range(-1000, 2001, 25)
HORIZONS_MS = (250, 500, 1000, 2000, 5000)
MOVE_WINDOW_MS = 250
MOVE_GAP_MS = 2000
"""Sharp moves closer together than this count as one."""
AFTER_MS = (0, 100, 250, 500, 1000, 2000, 3000)
BASIS_HALF_LIFE_S = 300.0


@dataclass(frozen=True)
class LeadLag:
    venue: str
    minutes: float
    peak_lag_ms: int
    """Lag at which the venue's mid returns correlate best with the reference's; positive means it follows."""
    peak_correlation: float
    correlations: dict[int, float]
    catch_up: dict[int, tuple[float, float]]
    """Per horizon: (beta, r2) of the venue's future mid move on today's basis-adjusted gap to the reference."""
    gap_bps_p50: float
    gap_bps_p90: float
    moves: "Moves | None" = None


@dataclass(frozen=True)
class Moves:
    """How the venue followed the reference's sharp moves, signed so the reference moved up."""

    threshold_bps: float
    count: int
    reference_bps: float
    """Mean size of the reference's move over MOVE_WINDOW_MS."""
    venue_bps: dict[int, float]
    """Per ms after the move: how far the venue had moved since it began."""
    gap_bps: dict[int, float]
    """Per ms after the move: the basis-adjusted gap still open (positive: the venue lags)."""


def mid_series(
    path: Path, venues: Sequence[str], grid_ms: int = GRID_MS, leaders: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Each venue's mid sampled on a shared grid of local receive times, holding the last value.

    `composite` is the composite of the composite venues not otherwise listed.
    """
    comp = None
    if composite.NAME in venues:
        comp = composite.Composite(tuple(v for v in (*composite.VENUES, *leaders) if v not in venues))
    sources = [v for v in venues if v != composite.NAME] + list(comp.venues if comp else ())
    parsers = {venue: VENUES[venue].Parser() for venue in sources}
    times: dict[str, list[float]] = {venue: [] for venue in venues}
    mids: dict[str, list[float]] = {venue: [] for venue in venues}
    for venue, recv_ms, message in recording.read(path):
        if venue not in parsers or recv_ms is None:
            continue
        for event in parsers[venue].events(message):
            if not (isinstance(event, Book) and event.bids and event.asks):
                continue
            if venue in times:
                times[venue].append(recv_ms)
                mids[venue].append((event.bids[0][0] + event.asks[0][0]) / 2)
            if comp is not None and (book := comp.on_book(venue, recv_ms, event)) is not None:
                times[composite.NAME].append(recv_ms)
                mids[composite.NAME].append(book.bids[0][0])
    if any(not times[venue] for venue in venues):
        missing = [venue for venue in venues if not times[venue]]
        raise ValueError(f"no books with receive times for {', '.join(missing)}")
    start = max(t[0] for t in times.values())
    end = min(t[-1] for t in times.values())
    grid = np.arange(start, end, grid_ms)
    sampled = {}
    for venue in venues:
        t = np.asarray(times[venue])
        index = np.searchsorted(t, grid, side="right") - 1
        sampled[venue] = np.log(np.asarray(mids[venue])[index])
    return grid, sampled


def measure(
    path: Path, reference: str, venues: Sequence[str], move_bps: float = 1.5, leaders: Sequence[str] = (),
) -> list[LeadLag]:
    if reference == composite.NAME:
        # Measure each venue against a composite without it, reading the recording once per venue.
        return [
            _measure(venue, *mid_series(path, [reference, venue], leaders=leaders), reference, move_bps)
            for venue in venues
        ]
    grid, log_mids = mid_series(path, [reference, *venues])
    return [_measure(venue, grid, log_mids, reference, move_bps) for venue in venues]


def sharp_moves(ref: np.ndarray, mid: np.ndarray, basis: np.ndarray, threshold_bps: float, warmup: int) -> Moves | None:
    """Follow the venue after each move of the reference of at least threshold_bps within MOVE_WINDOW_MS."""
    steps, gap_steps, last = MOVE_WINDOW_MS // GRID_MS, MOVE_GAP_MS // GRID_MS, max(AFTER_MS) // GRID_MS
    move = (ref[steps:] - ref[:-steps]) * 10_000
    events: list[int] = []
    for i in np.flatnonzero(np.abs(move) >= threshold_bps):
        if i >= warmup and i + steps + last < len(ref) and (not events or i - events[-1] >= gap_steps):
            events.append(int(i))
    if not events:
        return None
    start = np.array(events)
    sign = np.sign(move[start])
    venue_bps, gap_bps = {}, {}
    for after in AFTER_MS:
        t = start + steps + after // GRID_MS
        venue_bps[after] = float(np.mean((mid[t] - mid[start]) * sign) * 10_000)
        gap_bps[after] = float(np.mean((ref[t] + basis[start] - mid[t]) * sign) * 10_000)
    return Moves(threshold_bps, len(events), float(np.mean(np.abs(move[start]))), venue_bps, gap_bps)


def _measure(venue: str, grid: np.ndarray, log_mids: dict[str, np.ndarray], reference: str, move_bps: float = 1.5) -> LeadLag:
    ref = log_mids[reference]
    ref_returns = np.diff(ref)
    alpha = 1 - 0.5 ** (GRID_MS / 1000 / BASIS_HALF_LIFE_S)
    mid = log_mids[venue]
    returns = np.diff(mid)
    correlations = {lag: _lagged_correlation(ref_returns, returns, lag // GRID_MS) for lag in LAGS_MS}
    peak_lag = max(correlations, key=lambda lag: correlations[lag])
    basis = _ewma(mid - ref, alpha)  # Causal: uses only the past.
    gap = ref + basis - mid
    catch_up = {}
    warmup = int(BASIS_HALF_LIFE_S * 1000 / GRID_MS)
    for horizon in HORIZONS_MS:
        steps = horizon // GRID_MS
        x = gap[warmup:-steps:4]
        y = (mid[warmup + steps:] - mid[warmup:-steps])[::4]
        catch_up[horizon] = _regress_through_origin(x, y)
    gap_bps = np.abs(gap[warmup:]) * 10_000
    return LeadLag(
        venue=venue,
        minutes=(grid[-1] - grid[0]) / 60_000,
        peak_lag_ms=peak_lag,
        peak_correlation=correlations[peak_lag],
        correlations=correlations,
        catch_up=catch_up,
        gap_bps_p50=float(np.percentile(gap_bps, 50)) if gap_bps.size else 0.0,
        gap_bps_p90=float(np.percentile(gap_bps, 90)) if gap_bps.size else 0.0,
        moves=sharp_moves(ref, mid, basis, move_bps, warmup),
    )


def format_report(reference: str, results: Sequence[LeadLag]) -> str:
    lines = []
    for r in results:
        lines += [
            "",
            f"--- {r.venue} vs {reference} ({r.minutes:.0f} min) ---",
            f"peak correlation of mid returns at lag {r.peak_lag_ms:+d}ms: {r.peak_correlation:.3f}"
            f"  ({'follows' if r.peak_lag_ms > 0 else 'leads' if r.peak_lag_ms < 0 else 'moves with'} {reference})",
            "correlation by lag: " + "  ".join(
                f"{lag:+d}ms {r.correlations[lag]:.2f}" for lag in (-250, 0, 100, 250, 500, 750, 1000, 1500, 2000)
            ),
            f"basis-adjusted gap to {reference}: median {r.gap_bps_p50:.2f}bps, 90th percentile {r.gap_bps_p90:.2f}bps",
            "catch-up: future mid move = beta x gap now",
        ]
        for horizon, (beta, r2) in r.catch_up.items():
            lines.append(f"  {horizon:>5}ms  beta {beta:+.2f}  r2 {r2:.3f}")
        m = r.moves
        if m is None:
            lines.append(f"sharp moves: none of {reference} reached the threshold")
            continue
        lines.append(
            f"after {m.count} sharp {reference} moves (>= {m.threshold_bps:g}bps within {MOVE_WINDOW_MS}ms, "
            f"mean {m.reference_bps:.2f}bps), {r.venue} had moved / still lagged by:"
        )
        for after in AFTER_MS:
            lines.append(f"  +{after:>4}ms  moved {m.venue_bps[after]:+.2f}bps  gap {m.gap_bps[after]:+.2f}bps")
    return "\n".join(lines)


def _lagged_correlation(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """corr(a[t], b[t + lag])."""
    if lag >= 0:
        x, y = a[: len(a) - lag], b[lag:]
    else:
        x, y = a[-lag:], b[: len(b) + lag]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    level = values[0]
    for i, value in enumerate(values):
        level = (1 - alpha) * level + alpha * value
        out[i] = level
    return out


def _regress_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    denominator = float(x @ x)
    if denominator == 0:
        return 0.0, 0.0
    beta = float(x @ y) / denominator
    total = float(y @ y)
    r2 = 1 - float(((y - beta * x) ** 2).sum()) / total if total else 0.0
    return beta, r2
