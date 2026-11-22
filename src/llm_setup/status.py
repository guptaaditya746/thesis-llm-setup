"""The status API: polls each model's /health and /metrics, classifies it, keeps one day of samples.

    GET /healthz      liveness of this API
    GET /v1/status    {"overall", "updatedAt", "models": {<alias or alias@port>: {...}}}
    GET /v1/events    the supervisor's recent events
    GET /metrics      the same numbers in Prometheus text format

Run by the supervisor as ``python -m llm_setup.status <session folder>``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from .metrics import parse_metrics
from .profile import Profile, load_profile
from .runtime import read_events

POLL_S = 5


def classify(previous: str, reachable: bool, metrics: dict[str, Any],
             last_bad_at: float | None = None, now: float | None = None) -> str:
    """healthy, busy, saturated, unavailable, or recovering (healthy within a minute of trouble)."""
    if not reachable:
        return "unavailable"
    waiting = metrics.get("waiting", 0)
    kv = metrics.get("kvCache", 0)
    queue = metrics.get("queueP95Seconds", 0) or 0
    if waiting > 4 or kv >= .92:
        state = "saturated"
    elif waiting >= 2 or kv >= .80 or queue >= 2:
        state = "busy"
    else:
        state = "healthy"
    instant = now if now is not None else time.time()
    if state == "healthy" and previous in {"unavailable", "saturated"} and last_bad_at and instant - last_bad_at < 60:
        return "recovering"
    return state


class History:
    """Compact SQLite samples, pruned after ``retention_seconds``."""

    def __init__(self, path: str | Path, retention_seconds: int = 86400) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_seconds
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS samples (ts REAL NOT NULL, alias TEXT NOT NULL, data TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts)")

    def add(self, alias: str, data: dict[str, Any], now: float | None = None) -> None:
        timestamp = now if now is not None else time.time()
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO samples VALUES (?, ?, ?)", (timestamp, alias, json.dumps(data)))
            db.execute("DELETE FROM samples WHERE ts < ?", (timestamp - self.retention_seconds,))

    def recent(self, alias: str, seconds: int = 86400, now: float | None = None) -> list[dict[str, Any]]:
        cutoff = (now if now is not None else time.time()) - seconds
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT data FROM samples WHERE alias=? AND ts>=? ORDER BY ts", (alias, cutoff)).fetchall()
        return [json.loads(row[0]) for row in rows]


class StatusMonitor:
    def __init__(self, profile: Profile, history: History) -> None:
        self.targets = {key: instance.url for key, instance in profile.status_keys().items()}
        self.history = history
        self.last: dict[str, dict[str, Any]] = {}
        self.last_bad_at: dict[str, float] = {}

    async def poll_once(self) -> dict[str, Any]:
        now = time.time()
        async with httpx.AsyncClient(timeout=2) as client:
            for key, backend in self.targets.items():
                metrics: dict[str, Any] = {}
                reachable = False
                try:
                    health = await client.get(backend + "/health")
                    response = await client.get(backend + "/metrics")
                    reachable = health.is_success and response.is_success
                    metrics = parse_metrics(response.text) if response.is_success else {}
                except httpx.HTTPError:
                    pass
                state = classify(self.last.get(key, {}).get("state", "unknown"), reachable, metrics,
                                 self.last_bad_at.get(key), now)
                if state in {"unavailable", "saturated"}:
                    self.last_bad_at[key] = now
                item = {"state": state, "backend": backend, **{name: metrics.get(name) for name in (
                    "running", "waiting", "kvCache", "queueP95Seconds", "latencyP95Seconds", "preemptions")}}
                if reachable:
                    item["lastHealthyAt"] = _iso(now)
                self.last[key] = {name: value for name, value in item.items() if value is not None}
                self.history.add(key, self.last[key], now)
        return self.document()

    def document(self) -> dict[str, Any]:
        states = [entry.get("state", "unknown") for entry in self.last.values()]
        overall = "unknown" if not states else (
            "unavailable" if "unavailable" in states else "saturated" if "saturated" in states
            else "busy" if "busy" in states else "healthy" if all(s == "healthy" for s in states)
            else "recovering")
        return {"overall": overall, "updatedAt": _iso(), "models": self.last}


def _iso(timestamp: float | None = None) -> str:
    value = datetime.fromtimestamp(timestamp, UTC) if timestamp is not None else datetime.now(UTC)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


_PROMETHEUS = (("running", "llm_requests_running", "gauge", "Current running requests"),
               ("waiting", "llm_requests_waiting", "gauge", "Current waiting requests"),
               ("kvCache", "llm_kv_cache_usage_ratio", "gauge", "KV cache usage ratio"),
               ("queueP95Seconds", "llm_queue_time_seconds_p95", "gauge", "Backend request queue time p95"),
               ("latencyP95Seconds", "llm_request_latency_seconds_p95", "gauge", "Backend end-to-end latency p95"),
               ("preemptions", "llm_preemptions_total", "counter", "Requests paused because the KV cache was full"))


def create_app(profile: Profile, history: History, session: Path | None = None) -> FastAPI:
    monitor = StatusMonitor(profile, history)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async def loop() -> None:
            while True:
                try:
                    await monitor.poll_once()
                except Exception:
                    logging.getLogger(__name__).exception("status poll failed")
                await asyncio.sleep(POLL_S)

        task = asyncio.create_task(loop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(lifespan=lifespan)
    app.state.monitor = monitor

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        if not monitor.last:
            await monitor.poll_once()
        return monitor.document()

    @app.get("/v1/events")
    async def events() -> dict[str, Any]:
        return {"events": read_events(session)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus() -> str:
        lines = ["# HELP llm_backend_up Backend health", "# TYPE llm_backend_up gauge"]
        for _, metric, kind, text in _PROMETHEUS:
            lines += [f"# HELP {metric} {text}", f"# TYPE {metric} {kind}"]
        for key, item in monitor.last.items():
            lines.append(f'llm_backend_up{{alias="{key}"}} {int(item["state"] not in {"unavailable", "unknown"})}')
            lines += [f'{metric}{{alias="{key}"}} {item[name]}' for name, metric, _, _ in _PROMETHEUS
                      if item.get(name) is not None]
        return "\n".join(lines) + "\n"

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="llm_setup.status")
    parser.add_argument("session", type=Path, help="the session folder (runtime/<session-id>, node-local in a Slurm job)")
    args = parser.parse_args()
    profile = load_profile(args.session / "profile.yaml")
    app = create_app(profile, History(args.session / "status.sqlite3"), args.session)
    uvicorn.run(app, host="127.0.0.1", port=profile.status_port, log_level="warning")


if __name__ == "__main__":
    main()
