"""Starting and stopping a session.

``start`` launches every model instance in parallel, waits for each to answer /health, then starts
GLiNER (optional: a failure leaves the stack running without it), the LiteLLM gateway and the status
API, in that order. On any failure it stops what it started. The watchdog (watchdog.py) keeps a
started session running.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import vllm
from .gateway import render_litellm
from .logs import current_phase, tail
from .process import GATEWAY_SECRETS, child_environment, owned, signal_owned
from .profile import Profile
from .runtime import Runtime, alive_services, record_event, save_processes

Say = Callable[[str], None]
# How long a service may take to answer its health check after launch, by kind.
STARTUP_TIMEOUT_S = {"model": 1800, "gliner": 900, "gateway": 180, "status": 60}


def clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass
class Service:
    """One supervised process: how to launch it and how to tell it is ready."""

    name: str
    kind: str                       # model, gliner, gateway or status
    command: list[str]
    env: dict[str, str]
    log: Path
    health: str
    timeout: int
    process: subprocess.Popen[Any] | None = None
    started: float = 0.0
    restarts: list[float] = field(default_factory=list)
    state: str = "stopped"          # starting, ready, unhealthy, restarting, failed, stopped

    def launch(self) -> None:
        with self.log.open("ab") as handle:
            self.process = subprocess.Popen(self.command, stdout=handle, stderr=subprocess.STDOUT, env=self.env,
                                            start_new_session=True)
        self.started = time.monotonic()
        self.state = "starting"

    def exited(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def healthy(self) -> bool:
        try:
            return httpx.get(self.health, timeout=2).is_success
        except httpx.HTTPError:
            return False


@dataclass
class Session:
    id: str
    path: Path
    services: dict[str, Service]

    def save(self) -> None:
        save_processes(self.path, self.id, {name: {"pid": item.process.pid, "command": item.command,
                                                   "log": str(item.log)}
                                            for name, item in self.services.items() if item.process is not None})


def build_services(profile: Profile, env: dict[str, str], path: Path, session_id: str, key: str) -> dict[str, Service]:
    base = {**os.environ, **env}
    services: dict[str, Service] = {}
    for number, instance in enumerate(profile.instances):
        services[instance.name] = Service(
            instance.name, "model", vllm.command(instance), vllm.environment(base, instance, session_id, number),
            path / f"{instance.name}.log", f"{instance.url}/health", STARTUP_TIMEOUT_S["model"])
    if profile.gliner:
        gliner = profile.gliner
        command = [sys.executable, "-m", "llm_setup.gliner_server", "--model", gliner.model,
                   "--port", str(gliner.port), "--max-chars", str(gliner.max_chars)]
        if gliner.revision:
            command += ["--revision", gliner.revision]
        services["gliner"] = Service(
            "gliner", "gliner", command, child_environment(base, session_id, gpu=gliner.gpu, drop=GATEWAY_SECRETS),
            path / "gliner.log", f"http://127.0.0.1:{gliner.port}/health", STARTUP_TIMEOUT_S["gliner"])
    gateway_env = child_environment(base, session_id, drop=("HF_TOKEN",))
    gateway_env.update(LLM_BACKEND_KEY=key, LITELLM_MASTER_KEY=key)
    services["gateway"] = Service(
        "gateway", "gateway", ["litellm", "--config", str(path / "litellm-config.yaml"), "--host", "127.0.0.1",
                               "--port", str(profile.gateway_port)],
        gateway_env, path / "gateway.log", f"http://127.0.0.1:{profile.gateway_port}/health/liveliness",
        STARTUP_TIMEOUT_S["gateway"])
    services["status"] = Service(
        "status", "status", [sys.executable, "-m", "llm_setup.status", str(path)],
        child_environment(base, session_id, drop=("HF_TOKEN", *GATEWAY_SECRETS)), path / "status.log",
        f"http://127.0.0.1:{profile.status_port}/healthz", STARTUP_TIMEOUT_S["status"])
    return services


def wait_ready(services: list[Service], say: Say, interval: float = 15.0, poll: float = 2.0) -> None:
    """Wait until every service answers its health check, printing each one's phase every ``interval``.
    A service that exits or exceeds its timeout raises with the end of its log and a hint."""
    waiting = list(services)
    last_report = time.monotonic()
    while waiting:
        for service in list(waiting):
            code = service.exited()
            if code is not None:
                raise RuntimeError(f"{service.name} exited with status {code} before it became healthy"
                                   f"{vllm.failure_hint(service.log)}\n--- last lines of {service.log} ---\n"
                                   f"{tail(service.log)}")
            if service.healthy():
                service.state = "ready"
                waiting.remove(service)
                say(f"{service.name}: ready after {clock(time.monotonic() - service.started)}")
            elif time.monotonic() - service.started > service.timeout:
                raise RuntimeError(f"{service.name} not healthy after {clock(service.timeout)} "
                                   f"({current_phase(service.log)})\n--- last lines of {service.log} ---\n"
                                   f"{tail(service.log)}")
        if waiting and time.monotonic() - last_report >= interval:
            last_report = time.monotonic()
            say(" · ".join(f"{item.name}: {current_phase(item.log)} ({clock(time.monotonic() - item.started)})"
                           for item in waiting))
        if waiting:
            time.sleep(poll)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def clear_stale(runtime: Runtime, say: Say) -> None:
    """Forget a recorded session none of whose processes is alive (an allocation that ended, a closed
    terminal). A session with live processes blocks a new start."""
    session_id = runtime.current_id()
    if not session_id:
        return
    alive = alive_services(runtime.root / session_id, session_id)
    if alive:
        raise RuntimeError(f"session {session_id} is still running ({', '.join(alive)}); "
                           "run `llm-setup stop` first, or `scancel` its Slurm job")
    runtime.clear_current()
    record_event(runtime.root / session_id, "session", "cleared", "no recorded process was alive at the next start")
    say(f"previous session {session_id} ended without `stop` (no process alive); cleared")


def start(profile: Profile, env: dict[str, str], say: Say = print, runtime: Runtime | None = None) -> Session:
    key = env.get("LITELLM_MASTER_KEY", "")
    if not key or key == "replace-with-a-local-secret":
        raise RuntimeError("set a non-placeholder LITELLM_MASTER_KEY before start")
    runtime = runtime or Runtime()
    runtime.root.mkdir(parents=True, exist_ok=True)
    clear_stale(runtime, say)
    busy = [f"{name} :{port}" for name, port in profile.ports().items() if port_in_use(port)]
    if busy:
        raise RuntimeError(f"ports already in use: {', '.join(busy)}; another server (or a session that was not "
                           "stopped) holds them. `ss -ltnp | grep :<port>` and `ps -eo user,pid,cmd` show the "
                           "holder; on a shared node it can be another user's job: move only our port (e.g. "
                           "GLINER_PORT=8091) instead of stopping theirs. `llm-setup doctor` shows more")
    session_id, path = runtime.new_session()
    (path / "profile.yaml").write_text(yaml.safe_dump(profile.data, sort_keys=False), encoding="utf-8")
    (path / "litellm-config.yaml").write_text(yaml.safe_dump(render_litellm(profile), sort_keys=False),
                                              encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({
        "sessionId": session_id, "profile": profile.name, "profileFile": profile.source,
        "host": socket.gethostname(), "slurmJobId": os.environ.get("SLURM_JOB_ID")}, indent=2), encoding="utf-8")
    session = Session(session_id, path, build_services(profile, env, path, session_id, key))
    say(f"session {session_id} on {socket.gethostname()}; logs in {path}")
    record_event(path, "session", "starting", socket.gethostname())
    try:
        models = [service for service in session.services.values() if service.kind == "model"]
        for service in models:
            service.launch()
        session.save()
        say("models: " + ", ".join(f"{item.name} on GPU {item.gpu}" for item in profile.instances)
            + " (loading in parallel; usually 5-20 min)")
        wait_ready(models, say)
        if profile.gliner:
            # After the models: vLLM sizes its memory from what is free when it starts, and GLiNER
            # (an estimated 1.5-2 GB) fits in what the model on its GPU leaves.
            say(f"gliner: {profile.gliner.model} on GPU {profile.gliner.gpu}, port {profile.gliner.port}")
            start_optional(session, session.services["gliner"], say)
        for name in ("gateway", "status"):
            session.services[name].launch()
            session.save()
            wait_ready([session.services[name]], say)
        runtime.mark_current(session_id)
        record_event(path, "session", "ready")
        say(f"ready: gateway http://127.0.0.1:{profile.gateway_port}/v1 · "
            f"status http://127.0.0.1:{profile.status_port}/v1/status")
        return session
    except BaseException as exc:
        terminate(session, say=lambda _: None)
        record_event(path, "session", "start failed", str(exc).splitlines()[0][:300])
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"startup failed in session {session_id}: {exc}\nlogs: {path}") from exc


def start_optional(session: Session, service: Service, say: Say) -> bool:
    """Start a service the models do not need (GLiNER). If it does not become healthy it is stopped,
    marked failed (the watchdog leaves it alone) and recorded, and the stack starts without it."""
    service.launch()
    session.save()
    try:
        wait_ready([service], say)
        return True
    except RuntimeError as exc:
        process = service.process
        if process is not None and process.poll() is None and owned(process.pid, session.id):
            signal_owned(process.pid, signal.SIGTERM)
        service.state = "failed"
        first = str(exc).splitlines()[0][:300]
        record_event(session.path, service.name, "start failed", first)
        say(f"{service.name}: start failed, continuing without it ({first}); `llm-setup logs {service.name}`")
        return False


# Slurm's KillWait on gammaweb is 30 s: after scancel it SIGKILLs every process in the step. Escalate
# ourselves before that, so vLLM workers are not killed by Slurm mid-teardown (which can leave them
# unkillable and the job stuck in CG).
TERMINATE_GRACE_S = 20.0


def terminate(session: Session, say: Say = print, grace_s: float = TERMINATE_GRACE_S) -> list[str]:
    """Stop the session's own processes: SIGTERM, then SIGKILL after ``grace_s``."""
    stopped = []
    for service in reversed(list(session.services.values())):
        process = service.process
        if process is not None and process.poll() is None and owned(process.pid, session.id):
            signal_owned(process.pid, signal.SIGTERM)
            stopped.append(service.name)
    deadline = time.monotonic() + grace_s
    for service in session.services.values():
        process = service.process
        if process is None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if owned(process.pid, session.id):
                say(f"{service.name} did not stop in {int(grace_s)} s; killing it")
                signal_owned(process.pid, signal.SIGKILL)
        service.state = "stopped"
    return stopped


def stop(runtime: Runtime | None = None) -> list[str]:
    """Stop the running session from another shell. A foreground watchdog sees `current` disappear and
    exits without restarting anything."""
    runtime = runtime or Runtime()
    session_id = runtime.current_id()
    if not session_id:
        return []
    path = runtime.root / session_id
    try:
        records = json.loads((path / "processes.json").read_text(encoding="utf-8"))["services"]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"cannot read {path}/processes.json ({exc}); nothing was signalled and session "
                           f"{session_id} stays current. Check its processes (`ps -eo pid,cmd`) before "
                           f"removing {runtime.current_file}") from exc
    runtime.clear_current()
    record_event(path, "session", "stop requested", "llm-setup stop")
    stopped = []
    for name, data in reversed(list(records.items())):
        if owned(int(data["pid"]), session_id):
            signal_owned(int(data["pid"]), signal.SIGTERM)
            stopped.append(name)
    return stopped
