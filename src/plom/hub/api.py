"""REST and WebSocket API over a Hub, shaped like the aggregator APIs it replaces.

Auth is one shared token: `Authorization: Bearer <token>`, or `?token=` where headers can't be set
(browser WebSockets). Without a token configured, everything is open, which is only for local use.
"""

import asyncio
import contextlib
import secrets
import time

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from plom.hub.runner import Hub
from plom.hub.state import SymbolState

TICK_EVERY_S = 0.25
DOM_EVERY_TICKS = 4


def create_app(hub: Hub, token: str | None, start_hub: bool = True) -> FastAPI:
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
                   "server_events": ["tick", "dom", "tape"], "tick_every_ms": TICK_EVERY_S * 1000},
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
        }
        return {name: build() for name, build in parts.items() if name in include}

    @app.get("/api/v1/market/snapshot", dependencies=[Depends(auth)])
    def snapshot(symbol: str) -> dict:
        return {"symbol": symbol.upper(), **snapshot_of(state(symbol), {"tick", "dom", "tape"})}

    @app.get("/api/v1/market/snapshot/batch", dependencies=[Depends(auth)])
    def snapshot_batch(symbols: str, include: str = "tick") -> dict:
        wanted = [s.strip() for s in symbols.split(",") if s.strip()][:50]
        parts = set(include.split(","))
        return {"snapshots": {state(s).symbol: snapshot_of(state(s), parts) for s in wanted}}

    @app.websocket("/api/ws")
    async def ws(socket: WebSocket, token_param: str | None = Query(None, alias="token")) -> None:
        if not authorized(token_param):
            await socket.close(code=4401)
            return
        await socket.accept()
        subscribed: dict[str, int] = {}
        """Symbol -> the last tape sequence number sent."""

        async def receive() -> None:
            while True:
                message = await socket.receive_json()
                symbols = [s.upper() for s in message.get("symbols", []) if s.upper() in hub.states]
                if message.get("event") == "subscribe_symbols":
                    for s in symbols:
                        subscribed.setdefault(s, hub.states[s].seq)
                elif message.get("event") == "unsubscribe_symbols":
                    for s in symbols:
                        subscribed.pop(s, None)
                await socket.send_json({"event": "subscribed", "symbols": sorted(subscribed)})

        async def send() -> None:
            count = 0
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
                        await socket.send_json({"event": "dom", "data": s.dom(now)})
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
