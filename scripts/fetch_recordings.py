"""Download the hub's recordings that aren't already here, skipping the ones still being written.

    PLOM_TOKEN=... uv run python scripts/fetch_recordings.py https://hub.example.com data/remote
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

STILL_WRITING_S = 120
"""A file modified this recently is probably the hour still being recorded."""


def get(url: str, token: str) -> urllib.request.addinfourl:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "plom/0.1"})
    return urllib.request.urlopen(request, timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hub", help="the hub's base URL")
    parser.add_argument("out", type=Path, nargs="?", default=Path("data/remote"))
    parser.add_argument("--include-current", action="store_true", help="also fetch files still being written")
    args = parser.parse_args()
    token = os.environ["PLOM_TOKEN"]
    args.out.mkdir(parents=True, exist_ok=True)
    with get(f"{args.hub.rstrip('/')}/api/v1/recordings", token) as response:
        listing = json.load(response)["recordings"]
    now_ms = time.time() * 1000
    for entry in listing:
        target = args.out / entry["name"]
        if target.exists() and target.stat().st_size == entry["bytes"]:
            continue
        if not args.include_current and now_ms - entry["modified_ms"] < STILL_WRITING_S * 1000:
            print(f"skip {entry['name']} (still being written)")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        with get(f"{args.hub.rstrip('/')}/api/v1/recordings/{entry['name']}", token) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.rename(target)
        print(f"got {entry['name']} ({entry['bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
