"""REST and WebSocket API over a Hub, shaped like the aggregator APIs it replaces.

Auth is one shared token: `Authorization: Bearer <token>`, or `?token=` where headers can't be set
(browser WebSockets). Without a token configured, everything is open, which is only for local use.
"""

import asyncio
import contextlib
import secrets
import time
from functools import partial

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from plom.hub import candles
from plom.hub.runner import Hub
from plom.hub.state import HEATMAP_LEVELS, HEATMAP_SECONDS, SymbolState

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"
TICK_EVERY_S = 0.25
HISTORY_MAX = 2000
BACKFILL_BELOW = 0.8
"""Backfill a history window holding fewer than this share of its candles."""
SYNC_BACKFILL_MS = 7 * 86_400_000
"""Windows up to this long wait for their backfill; longer ones get it in the background."""
DOM_EVERY_TICKS = 4


def create_app(hub: Hub, token: str | None, start_hub: bool = True, recordings: Path | None = None) -> FastAPI:
    """`recordings` is the directory `plom record` writes to, served for download (with the token)."""
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_hub:
            hub.start()
        yield
        if start_hub:
            await hub.stop()

    app = FastAPI(title="Plom Hub", lifespan=lifespan)

    def authorized(supplied: str | None) -> bool:
        return token is None or (supplied is not None and secrets.compare_digest(supplied, token))

    def auth(request: Request, token_param: str | None = Query(None, alias="token")) -> None:
        header = request.headers.get("authorization", "")
        supplied = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else token_param
        if not authorized(supplied):
            raise HTTPException(401, "missing or bad token")

    def state(symbol: str) -> SymbolState:
        found = hub.states.get(symbol.upper())
        if found is None:
            raise HTTPException(404, f"unknown symbol {symbol}; see /api/v1/universe")
        return found

    def now_ms() -> float:
        return time.time() * 1000

    @app.get("/api/v1/health")
    def health() -> dict:
        live = sum(status == "live" for status in hub.status.values())
        return {"ok": True, "ts_ms": round(now_ms()), "feeds_live": live, "feeds": len(hub.status)}

    @app.get("/api/v1/meta", dependencies=[Depends(auth)])
    def meta() -> dict:
        return {
            "ok": True, "version": "v1", "venues": list(hub.venues),
            "ws": {"path": "/api/ws", "subscribe_event": "subscribe_symbols", "unsubscribe_event": "unsubscribe_symbols",
                   "server_events": ["tick", "dom", "tape", "heatmap"], "tick_every_ms": TICK_EVERY_S * 1000},
        }

    @app.get("/api/v1/status", dependencies=[Depends(auth)])
    def status() -> dict:
        by_symbol: dict[str, dict[str, str]] = {}
        for (symbol, venue), value in sorted(hub.status.items()):
            by_symbol.setdefault(symbol, {})[venue] = value
        return by_symbol

    @app.get("/api/v1/universe", dependencies=[Depends(auth)])
    def universe() -> dict:
        return {"symbols": list(hub.states)}

    @app.get("/api/v1/market/tick", dependencies=[Depends(auth)])
    def tick(symbol: str) -> dict:
        return state(symbol).tick(now_ms())

    @app.get("/api/v1/market/dom", dependencies=[Depends(auth)])
    def dom(symbol: str, bucket: float | None = None, depth: int = Query(50, ge=1, le=500), adjusted: bool = False) -> dict:
        return state(symbol).dom(now_ms(), bucket, depth, adjusted)

    @app.get("/api/v1/market/heatmap", dependencies=[Depends(auth)])
    def heatmap(symbol: str, seconds: float = Query(300, gt=0, le=HEATMAP_SECONDS), step: int = Query(1, ge=1, le=60)) -> dict:
        columns = state(symbol).heatmap_since(now_ms(), seconds, step)
        return {"symbol": symbol.upper(), "levels": HEATMAP_LEVELS, "adjusted": True, "columns": [c.to_json() for c in columns]}

    @app.get("/api/v1/market/tape", dependencies=[Depends(auth)])
    def tape(symbol: str, after_seq: int = 0, limit: int = Query(200, ge=1, le=5000)) -> dict:
        prints = state(symbol).prints_after(after_seq, limit)
        return {"symbol": symbol.upper(), "trades": [p.to_json() for p in prints]}

    @app.get("/api/v1/market/latest", dependencies=[Depends(auth)])
    def latest(symbol: str) -> dict:
        s = state(symbol)
        return {
            "symbol": s.symbol, "tick_ms": round(s.price_ms) if s.price_ms else None,
            "tape_ms": round(s.tape[-1].recv_ms) if s.tape else None, "tape_seq": s.seq,
            "venues_ms": {v: round(t) for v, (t, _) in sorted(s.books.items())},
        }

    def snapshot_of(s: SymbolState, include: set[str]) -> dict:
        now = now_ms()
        parts = {
            "tick": lambda: s.tick(now),
            "dom": lambda: s.dom(now),
            "tape": lambda: [p.to_json() for p in s.prints_after(0, 200)],
            "heatmap": lambda: [c.to_json() for c in s.heatmap_since(now, 60)],
        }
        return {name: build() for name, build in parts.items() if name in include}

    @app.get("/api/v1/market/snapshot", dependencies=[Depends(auth)])
    def snapshot(symbol: str) -> dict:
        return {"symbol": symbol.upper(), **snapshot_of(state(symbol), {"tick", "dom", "tape", "heatmap"})}

    @app.get("/api/v1/market/snapshot/batch", dependencies=[Depends(auth)])
    def snapshot_batch(symbols: str, include: str = "tick") -> dict:
        wanted = [s.strip() for s in symbols.split(",") if s.strip()][:50]
        parts = set(include.split(","))
        return {"snapshots": {state(s).symbol: snapshot_of(state(s), parts) for s in wanted}}

    app.mount("/static", StaticFiles(directory=DASHBOARD.parent), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))

    @app.get("/api/v1/recordings", dependencies=[Depends(auth)])
    def list_recordings() -> dict:
        files = sorted(recordings.glob("*.jsonl*")) if recordings and recordings.is_dir() else []
        return {"recordings": [
            {"name": f.name, "bytes": f.stat().st_size, "modified_ms": round(f.stat().st_mtime * 1000)} for f in files
        ]}

    @app.get("/api/v1/recordings/{name}", dependencies=[Depends(auth)])
    def get_recording(name: str) -> FileResponse:
        path = recordings / name if recordings else None
        # Only plain names of files directly in the directory: no paths, no "..".
        if path is None or name != Path(name).name or not name.endswith((".jsonl", ".jsonl.gz")) or not path.is_file():
            raise HTTPException(404, "no such recording")
        return FileResponse(path, media_type="application/gzip", filename=name)

    backfilling: set[tuple[str, str]] = set()

    @app.get("/api/history/{symbol}", dependencies=[Depends(auth)])
    async def history(
        symbol: str, interval: str = "1m", from_ms: int | None = Query(None, alias="from"),
        to_ms: int | None = Query(None, alias="to"), limit: int = Query(500, ge=1, le=HISTORY_MAX),
    ) -> dict:
        s = state(symbol)
        if hub.store is None:
            raise HTTPException(503, "no candle store configured")
        length = candles.INTERVALS.get(interval)
        if length is None:
            raise HTTPException(400, f"interval must be one of {', '.join(candles.INTERVALS)}")
        to_ms = to_ms or int(now_ms())
        from_ms = max(from_ms or to_ms - 86_400_000, to_ms - limit * length)
        found = hub.store.read(s.symbol, interval, from_ms, to_ms)
        expected = max(1, (to_ms - from_ms) // length)
        key = (s.symbol, interval)
        if len(found) < BACKFILL_BELOW * expected and key not in backfilling:
            fill = partial(candles.backfill, hub.store, s.symbol, interval, from_ms, to_ms, dict(s.composite.basis))
            backfilling.add(key)
            if to_ms - from_ms <= SYNC_BACKFILL_MS:
                try:
                    await asyncio.to_thread(fill)
                finally:
                    backfilling.discard(key)
                found = hub.store.read(s.symbol, interval, from_ms, to_ms)
            else:
                task = asyncio.create_task(asyncio.to_thread(fill))
                task.add_done_callback(lambda _: backfilling.discard(key))
        return {
            "symbol": s.symbol, "interval": interval, "from_ms": from_ms, "to_ms": to_ms, "count": len(found),
            "backfilling": key in backfilling, "candles": [c.to_json() for c in found],
        }

    @app.websocket("/api/ws")
    async def ws(socket: WebSocket, token_param: str | None = Query(None, alias="token")) -> None:
        if not authorized(token_param):
            await socket.close(code=4401)
            return
        await socket.accept()
        subscribed: dict[str, int] = {}
        """Symbol -> the last tape sequence number sent."""
        options = {"adjusted": False}

        async def receive() -> None:
            while True:
                message = await socket.receive_json()
                symbols = [s.upper() for s in message.get("symbols", []) if s.upper() in hub.states]
                if message.get("event") == "subscribe_symbols":
                    options["adjusted"] = bool(message.get("adjusted", options["adjusted"]))
                    for s in symbols:
                        subscribed.setdefault(s, hub.states[s].seq)
                elif message.get("event") == "unsubscribe_symbols":
                    for s in symbols:
                        subscribed.pop(s, None)
                await socket.send_json({"event": "subscribed", "symbols": sorted(subscribed)})

        async def send() -> None:
            count = 0
            last_column: dict[str, int] = {}
            while True:
                await asyncio.sleep(TICK_EVERY_S)
                now = now_ms()
                for symbol in list(subscribed):
                    s = hub.states[symbol]
                    await socket.send_json({"event": "tick", "data": s.tick(now)})
                    prints = s.prints_after(subscribed[symbol])
                    if prints:
                        subscribed[symbol] = prints[-1].seq
                        await socket.send_json({"event": "tape", "symbol": symbol, "data": [p.to_json() for p in prints]})
                    if count % DOM_EVERY_TICKS == 0:
                        await socket.send_json({"event": "dom", "data": s.dom(now, adjusted=options["adjusted"])})
                        if s.heatmap and s.heatmap[-1].ts_ms > last_column.get(symbol, 0):
                            last_column[symbol] = s.heatmap[-1].ts_ms
                            await socket.send_json({"event": "heatmap", "symbol": symbol, "data": s.heatmap[-1].to_json()})
                count += 1

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()

    return app
