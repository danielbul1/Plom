"""Re-run the B4-B7 comparisons on a recording and write one Markdown report.

Each variant changes one idea against a venue profile's defaults and replays the recording in its own
process. Its PnL is compared with the defaults block by block (a paired block bootstrap), so a
difference whose interval excludes zero is one the data can see. The report also holds lead-lag
against the reference and the defaults' full evaluation per venue (alpha R², GLFT A and k, jumps).

A recording that is still being written can be used: reading stops quietly at its unfinished end.

    uv run python scripts/recheck.py data/btc_multi_2.jsonl.gz
    uv run python scripts/recheck.py data/btc_multi_2.jsonl.gz --profiles hyperliquid --workers 4
"""

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from plom import composite, evaluate, leadlag, runner
from plom.cli import breakdown
from plom.mm import Config, MarketMaker
from plom.profiles import PROFILES

DEFAULT_PROFILES = ("hyperliquid", "lighter-standard", "orderly-raydium")
EXPERIMENTS: dict[str, list[dict[str, float]]] = {
    "B4 reference price": [
        {"reference_weight": 0.0},
        {"reference_weight": 1.0},
    ],
    "B5 forecast": [
        {"alpha_weight": 0.5},
        {"alpha_weight": 1.0},
        {"alpha_widen": 1.0},
        {"alpha_pull_margin_bps": 0.0},
    ],
    "B6 GLFT and position age": [
        {"glft_gamma": 0.01},
        {"glft_gamma": 0.1},
        {"glft_gamma": 1.0},
        {"glft_gamma": 0.1, "glft_layer_spacing": 0.5},
        {"position_age_s": 30.0},
        {"position_age_s": 120.0},
        {"flatten_age_s": 60.0},
        {"flatten_age_s": 300.0},
    ],
    "B7 volatility and jumps": [
        {"vol_bipower": 1},
        {"jump_alpha": 0.01},
        {"jump_bps": math.inf},
        {"jump_alpha": 0.01, "jump_bps": math.inf},
    ],
}


def run(path: Path, venue: str, reference: str, config: Config, label: str, block_s: float, detailed: bool) -> tuple[evaluate.Evaluation, str]:
    """Replay one variant; with `detailed`, also return its full evaluation and breakdown as text."""
    mm = MarketMaker(config)
    tracker = evaluate.Tracker(block_s)
    dispatcher = runner.Dispatcher(mm)
    for is_reference, recv_ms, event in runner.replay(path, venue, reference):
        dispatcher.feed(is_reference, recv_ms, event)
        tracker.observe(mm)
    result = evaluate.evaluate(mm, tracker, label)
    text = evaluate.format_report(result) + "\n" + breakdown(mm) if detailed else ""
    return result, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replay", type=Path)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES), help="comma-separated venue profiles")
    parser.add_argument("--reference", default=composite.NAME, help="reference venue, or composite (the default)")
    parser.add_argument("--block-s", type=float, default=evaluate.BLOCK_S)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, help="default: recheck-<time>.md next to the recording")
    args = parser.parse_args()
    profiles = args.profiles.split(",")
    unknown = set(profiles) - set(PROFILES)
    if unknown:
        raise SystemExit(f"unknown profiles: {', '.join(sorted(unknown))}")
    out = args.out or args.replay.parent / f"recheck-{time.strftime('%Y%m%d-%H%M')}.md"
    started = time.monotonic()

    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for name in profiles:
            profile = PROFILES[name]
            base = replace(Config(), **profile.config)
            key = (name, "", "defaults")
            futures[pool.submit(run, args.replay, profile.venue, args.reference, base, "defaults", args.block_s, True)] = key
            for experiment, variants in EXPERIMENTS.items():
                for overrides in variants:
                    label = " ".join(f"{flag}={value:g}" for flag, value in overrides.items())
                    config = replace(base, **overrides)
                    futures[pool.submit(run, args.replay, profile.venue, args.reference, config, label, args.block_s, False)] = (name, experiment, label)
        print(f"{len(futures)} replays of {args.replay} on {args.workers} workers", flush=True)
        venues = sorted({PROFILES[name].venue for name in profiles})
        lead_lag = leadlag.format_report(args.reference, leadlag.measure(args.replay, args.reference, venues))
        for done, future in enumerate(as_completed(futures), 1):
            name, experiment, label = futures[future]
            runs.append((name, experiment, label, *future.result()))
            print(f"  {done}/{len(futures)}  {name}  {label}", flush=True)

    report = format_report(args.replay, args.reference, profiles, runs, lead_lag, args.block_s, time.monotonic() - started)
    out.write_text(report, encoding="utf-8")
    print(f"\nwrote {out}")


def format_report(path, reference, profiles, runs, lead_lag, block_s, elapsed_s) -> str:
    by_key = {(name, label): (result, text) for name, _, label, result, text in runs}
    first = by_key[(profiles[0], "defaults")][0]
    lines = [
        f"# Recheck of `{path.name}`",
        "",
        f"{time.strftime('%Y-%m-%d %H:%M')}: {first.hours:.2f}h of market time in {len(first.block_pnls)} blocks of "
        f"{block_s:g}s, {len(runs)} replays in {elapsed_s / 60:.1f} minutes. Reference: {reference}.",
        "",
        "Intervals are 95% block bootstraps. *vs defaults* pairs each block with the same block under the "
        "profile's defaults; **better** or **worse** means that interval excludes zero. "
        f"Out of {len(runs) - len(profiles)} comparisons, about {(len(runs) - len(profiles)) / 20:.0f} would do so by chance "
        "alone, so trust a difference that holds across recordings or venues.",
        "",
        "## Differences the data can see",
        "",
    ]
    verdicts = []
    for name in profiles:
        base = by_key[(name, "defaults")][0]
        for experiment, variants in EXPERIMENTS.items():
            for overrides in variants:
                label = " ".join(f"{flag}={value:g}" for flag, value in overrides.items())
                result = by_key[(name, label)][0]
                difference = evaluate.paired_difference(result, base)
                if difference and (difference.low > 0 or difference.high < 0):
                    verdicts.append(f"- {name}, {experiment}: `{label}` {_verdict(difference)} by {difference} $/h")
    lines += verdicts or ["None."]
    lines += ["", "## Lead-lag", "", "```text", lead_lag.strip("\n"), "```"]

    for name in profiles:
        base, text = by_key[(name, "defaults")]
        lines += [
            "",
            f"## {name}",
            "",
            f"Defaults: {base.fills:,} fills, PnL ${base.pnl:+,.3f} = spread capture ${base.spread_capture:+,.3f} "
            f"+ inventory ${base.inventory_pnl:+,.3f} - fees ${base.fees:,.3f}; per hour {base.pnl_per_hour or 'n/a'}.",
        ]
        for experiment, variants in EXPERIMENTS.items():
            lines += [
                "",
                f"### {experiment}",
                "",
                "| variant | fills | fees $ | PnL $/h [95%] | vs defaults $/h [95%] | | markout 1s | markout 60s |",
                "|---|---:|---:|---|---|---|---:|---:|",
            ]
            for overrides in variants:
                label = " ".join(f"{flag}={value:g}" for flag, value in overrides.items())
                result = by_key[(name, label)][0]
                difference = evaluate.paired_difference(result, base)
                lines.append(
                    f"| `{label}` | {result.fills:,} | {result.fees:.3f} | {result.pnl_per_hour or 'n/a'} | "
                    f"{difference or 'n/a'} | {_verdict(difference)} | {_markout(result, 1000)} | {_markout(result, 60_000)} |"
                )
        lines += ["", "<details><summary>Defaults: full evaluation</summary>", "", "```text", text.strip("\n"), "```", "", "</details>"]
    return "\n".join(lines) + "\n"


def _verdict(difference: evaluate.Interval | None) -> str:
    if difference is None:
        return ""
    return "**better**" if difference.low > 0 else "**worse**" if difference.high < 0 else ""


def _markout(result: evaluate.Evaluation, horizon_ms: int) -> str:
    stats = next((h for h in result.horizons if h.horizon_ms == horizon_ms), None)
    return f"{stats.mean_bps.mean:+.2f}" if stats and stats.mean_bps else "n/a"


if __name__ == "__main__":
    main()
