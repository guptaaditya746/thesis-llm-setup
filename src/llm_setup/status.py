"""Polling, status classification, and local FastAPI endpoints."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from .history import History
from .metrics import parse_metrics
from .models import model_targets

STATES = {"healthy", "busy", "saturated", "unavailable", "recovering", "unknown"}


def classify(previous: str, reachable: bool, metrics: dict[str, Any],
             last_bad_at: float | None = None, now: float | None = None) -> str:
    import time
    instant = now if now is not None else time.time()
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
    if state == "healthy" and previous in {"unavailable", "saturated"} and last_bad_at and instant - last_bad_at < 60:
        return "recovering"
    return state


class StatusMonitor:
    def __init__(self, profile: dict[str, Any], history: History):
        self.targets = model_targets(profile)
        self.history = history
        self.last: dict[str, dict[str, Any]] = {}
        self.last_bad_at: dict[str, float] = {}
        self.gateway = profile["gateway"]

    async def poll_once(self) -> dict[str, Any]:
        import time
        now = time.time()
        async with httpx.AsyncClient(timeout=2) as client:
            for alias, target in self.targets.items():
                old = self.last.get(alias, {})
                metrics: dict[str, Any] = {}
                reachable = False
                try:
                    health = await client.get(target["backend"] + "/health")
                    metric_response = await client.get(target["backend"] + "/metrics")
                    reachable = health.is_success and metric_response.is_success
                    metrics = parse_metrics(metric_response.text) if metric_response.is_success else {}
                except httpx.HTTPError:
                    pass
                state = classify(old.get("state", "unknown"), reachable, metrics,
                                 self.last_bad_at.get(alias), now)
                if state in {"unavailable", "saturated"}:
                    self.last_bad_at[alias] = now
                item = {"state": state, "backend": target["backend"],
                        "running": metrics.get("running"), "waiting": metrics.get("waiting"),
                        "kvCache": metrics.get("kvCache"),
                        "queueP95Seconds": metrics.get("queueP95Seconds"),
                        "latencyP95Seconds": metrics.get("latencyP95Seconds"),
                        "preemptions": metrics.get("preemptions")}
                if reachable:
                    item["lastHealthyAt"] = _iso(now)
                self.last[alias] = {k: v for k, v in item.items() if v is not None}
                self.history.add(alias, self.last[alias], now)
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


def create_app(profile: dict[str, Any], history: History) -> FastAPI:
    monitor = StatusMonitor(profile, history)
    app = FastAPI()
    app.state.monitor = monitor

    @app.on_event("startup")
    async def start_polling() -> None:
        async def loop() -> None:
            while True:
                await monitor.poll_once()
                await asyncio.sleep(5)
        app.state.poller = asyncio.create_task(loop())

    @app.on_event("shutdown")
    async def stop_polling() -> None:
        task = getattr(app.state, "poller", None)
        if task:
            task.cancel()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        if not monitor.last:
            await monitor.poll_once()
        return monitor.document()

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus() -> str:
        lines = [
            "# HELP llm_backend_up Backend health", "# TYPE llm_backend_up gauge",
            "# HELP llm_requests_running Current running requests", "# TYPE llm_requests_running gauge",
            "# HELP llm_requests_waiting Current waiting requests", "# TYPE llm_requests_waiting gauge",
            "# HELP llm_kv_cache_usage_ratio KV cache usage ratio", "# TYPE llm_kv_cache_usage_ratio gauge",
            "# HELP llm_queue_time_seconds_p95 Backend request queue time p95",
            "# TYPE llm_queue_time_seconds_p95 gauge",
            "# HELP llm_request_latency_seconds_p95 Backend end-to-end latency p95",
            "# TYPE llm_request_latency_seconds_p95 gauge",
            "# HELP llm_preemptions_total Requests paused because the KV cache was full",
            "# TYPE llm_preemptions_total counter",
        ]
        for alias, item in monitor.last.items():
            lines.append(f'llm_backend_up{{alias="{alias}"}} {int(item["state"] not in {"unavailable", "unknown"})}')
            for key, metric in (("running", "llm_requests_running"), ("waiting", "llm_requests_waiting"),
                                ("kvCache", "llm_kv_cache_usage_ratio"),
                                ("queueP95Seconds", "llm_queue_time_seconds_p95"),
                                ("latencyP95Seconds", "llm_request_latency_seconds_p95"),
                                ("preemptions", "llm_preemptions_total")):
                if item.get(key) is not None:
                    lines.append(f'{metric}{{alias="{alias}"}} {item[key]}')
        return "\n".join(lines) + "\n"

    return app
