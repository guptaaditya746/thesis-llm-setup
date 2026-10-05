"""Keep a started session running: restart what crashes, report what changes, stop cleanly.

Used by ``llm-setup start --foreground`` (and so by the Slurm batch job): the command stays in the
foreground for the whole session. It restarts a service whose process exits (the gateway and status
API at once, a model with growing delays and at most ``MODEL_RESTARTS_PER_HOUR`` times an hour), says
when a running service stops answering its health check and when it answers again, prints a summary
line every few minutes, and on SIGTERM (``scancel``, the end of the Slurm time limit) or Ctrl-C stops
every process of the session. Every change is printed and appended to ``events.jsonl`` in the session
folder, which ``llm-setup events`` and the status API (``/v1/events``) show.
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import httpx

from .supervisor import Say, Service, Session, clock, record_event, terminate

MODEL_RESTARTS_PER_HOUR = 3
SERVICE_RESTARTS_PER_HOUR = 20
UNHEALTHY_AFTER = 6          # failed health checks in a row (one a check interval) before it is reported


class Watchdog:
    def __init__(self, session: Session, say: Say, status_url: str, check_s: float = 10.0,
                 summary_s: float = 300.0, root: Path = Path("runtime"),
                 backoff_s: tuple[float, float] = (30.0, 5.0)) -> None:
        """``backoff_s``: the first restart delay for a model and for the gateway/status API; it doubles
        with each restart in the last hour (at most 300 s and 60 s)."""
        self.backoff_s = backoff_s
        self.session = session
        self.say = say
        self.status_url = status_url
        self.check_s = check_s
        self.summary_s = summary_s
        self.root = root
        self.stop = threading.Event()
        self.reason = "stopped"
        self.failures: dict[str, int] = {}
        self.restart_at: dict[str, float] = {}
        self.started = time.monotonic()

    # Signals ---------------------------------------------------------------------------------------
    def install_signals(self) -> None:
        def handler(number: int, _frame: object) -> None:
            self.reason = f"received {signal.Signals(number).name}"
            self.stop.set()

        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
            signal.signal(number, handler)

    # Events ----------------------------------------------------------------------------------------
    def event(self, service: str, what: str, detail: str = "") -> None:
        record_event(self.session.path, service, what, detail)
        self.say(f"{service}: {what}" + (f" ({detail})" if detail else ""))

    # The loop --------------------------------------------------------------------------------------
    def run(self) -> int:
        """Supervise until a signal or `llm-setup stop`; returns the exit code for the batch job."""
        last_summary = time.monotonic()
        self.say(f"watching {len(self.session.services)} services; Ctrl-C or `scancel` stops the session")
        while not self.stop.wait(self.check_s):
            if self._stopped_elsewhere():
                self.reason = "stopped with `llm-setup stop`"
                break
            for service in self.session.services.values():
                self.check(service)
            if time.monotonic() - last_summary >= self.summary_s:
                last_summary = time.monotonic()
                self.say(self.summary())
        self.event("session", "stopping", self.reason)
        stopped = terminate(self.session, say=self.say)
        current = self.root / "current"
        if current.exists() and current.read_text(encoding="utf-8").strip() == self.session.id:
            current.unlink()
        self.event("session", "stopped", ", ".join(stopped) or "nothing was running")
        failed = [name for name, item in self.session.services.items() if item.state == "failed"]
        return 1 if failed else 0

    def check(self, service: Service) -> None:
        if service.state == "failed":
            return
        if service.state == "restarting":
            due = self.restart_at.get(service.name)
            if due is not None:                         # waiting out the delay before the next launch
                if time.monotonic() >= due:
                    del self.restart_at[service.name]
                    service.restarts.append(time.time())
                    service.launch()
                    service.state = "restarting"
                    self.session.save()
                    pid = service.process.pid if service.process else "?"
                    self.event(service.name, "restarted", f"restart {len(service.restarts)}, pid {pid}")
                return
            code = service.exited()
            if code is not None:
                self.event(service.name, "exited again", f"status {code}; last log line: {_last_line(service)}")
                self._schedule_restart(service)
            elif service.healthy():
                service.state = "ready"
                self.failures[service.name] = 0
                self.event(service.name, "healthy again", f"{clock(time.monotonic() - service.started)} after restart")
            return
        code = service.exited()
        if code is not None:
            self.event(service.name, "exited", f"status {code}; last log line: {_last_line(service)}")
            self._schedule_restart(service)
            return
        if service.healthy():
            if service.state == "unhealthy":
                self.event(service.name, "healthy again")
            service.state = "ready"
            self.failures[service.name] = 0
            return
        self.failures[service.name] = self.failures.get(service.name, 0) + 1
        if self.failures[service.name] == UNHEALTHY_AFTER and service.state == "ready":
            service.state = "unhealthy"
            self.event(service.name, "not answering its health check",
                       f"process alive for {int(UNHEALTHY_AFTER * self.check_s)} s without an answer")

    def _schedule_restart(self, service: Service) -> None:
        model = service.name in {"heavy", "lite", "embed"}
        limit = MODEL_RESTARTS_PER_HOUR if model else SERVICE_RESTARTS_PER_HOUR
        recent = [moment for moment in service.restarts if time.time() - moment < 3600]
        if len(recent) >= limit:
            service.state = "failed"
            self.event(service.name, "gave up restarting",
                       f"{len(recent)} restarts in the last hour; read {service.log}")
            return
        first = self.backoff_s[0] if model else self.backoff_s[1]
        delay = min(300.0 if model else 60.0, first * 2 ** len(recent))
        service.state = "restarting"
        self.restart_at[service.name] = time.monotonic() + delay
        self.event(service.name, "restarting", f"in {int(delay)} s")

    def _stopped_elsewhere(self) -> bool:
        current = self.root / "current"
        try:
            return current.read_text(encoding="utf-8").strip() != self.session.id
        except OSError:
            return True

    def summary(self) -> str:
        states = {name: item.state for name, item in self.session.services.items()}
        restarts = sum(len(item.restarts) for item in self.session.services.values())
        line = (f"up {clock(time.monotonic() - self.started)} · "
                + ", ".join(f"{name} {state}" for name, state in states.items())
                + f" · restarts {restarts}")
        try:
            report = httpx.get(self.status_url, timeout=3).json()
            models = report.get("models", {})
            line += " · " + ", ".join(
                f"{alias} {item.get('state')} run {item.get('running', '-')} wait {item.get('waiting', '-')}"
                for alias, item in models.items())
        except (httpx.HTTPError, ValueError):
            line += " · status API not answering"
        return line


def _last_line(service: Service) -> str:
    from .phases import log_lines

    lines = log_lines(service.log)
    return lines[-1][:160] if lines else "empty log"
