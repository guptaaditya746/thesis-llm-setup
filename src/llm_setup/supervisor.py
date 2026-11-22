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
from .mirror import STOP_REQUEST
from .process import GATEWAY_SECRETS, child_environment, owned, signal_owned
from .profile import Profile
from .runtime import Runtime, alive_services, read_metadata, record_event, save_processes

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
    mirror: subprocess.Popen[Any] | None = None
    mirror_target: Path | None = None

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
    path = runtime.session_path(session_id)
    host = read_metadata(path).get("host")
    if host and host != socket.gethostname():
        say(f"runtime/current names session {session_id} on {host}, not this node; leaving it to that node")
        return
    alive = alive_services(path, session_id)
    if alive:
        raise RuntimeError(f"session {session_id} is still running ({', '.join(alive)}); "
                           "run `llm-setup stop` first, or `scancel` its Slurm job")
    runtime.clear_current(only=session_id, shared=True)
    if runtime.is_local(session_id):
        record_event(path, "session", "cleared", "no recorded process was alive at the next start")
    say(f"previous session {session_id} ended without `stop` (no process alive); cleared")


def start(profile: Profile, env: dict[str, str], say: Say = print, runtime: Runtime | None = None) -> Session:
    key = env.get("LITELLM_MASTER_KEY", "")
    if not key or key == "replace-with-a-local-secret":
        raise RuntimeError("set a non-placeholder LITELLM_MASTER_KEY before start")
    runtime = runtime or Runtime.default()
    runtime.root.mkdir(parents=True, exist_ok=True)
    runtime.work.mkdir(parents=True, exist_ok=True)
    clear_stale(runtime, say)
    busy = [f"{name} :{port}" for name, port in profile.ports().items() if port_in_use(port)]
    if busy:
        raise RuntimeError(f"ports already in use: {', '.join(busy)}; another server (or a session that was not "
                           "stopped) holds them. `ss -ltnp | grep :<port>` and `ps -eo user,pid,cmd` show the "
                           "holder; on a shared node it can be another user's job: move only our port (e.g. "
                           "GLINER_PORT=8091) instead of stopping theirs. `llm-setup doctor` shows more")
    session_id, path = runtime.new_session()
    target = runtime.root / session_id if runtime.mirrored else None
    (path / "profile.yaml").write_text(yaml.safe_dump(profile.data, sort_keys=False), encoding="utf-8")
    (path / "litellm-config.yaml").write_text(yaml.safe_dump(render_litellm(profile), sort_keys=False),
                                              encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({
        "sessionId": session_id, "profile": profile.name, "profileFile": profile.source,
        "host": socket.gethostname(), "slurmJobId": os.environ.get("SLURM_JOB_ID"),
        "work": str(path.resolve()), "mirror": str(target.resolve()) if target else None}, indent=2),
        encoding="utf-8")
    session = Session(session_id, path, build_services(profile, env, path, session_id, key))
    if target is not None:
        session.mirror_target = target
        session.mirror = start_mirror(path, target, session_id)
    say(f"session {session_id} on {socket.gethostname()}; logs in {path}"
        + (f", copied every minute to {runtime.root / session_id}" if session.mirror else ""))
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
        stop_mirror(session)
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


# How long the supervisor waits for the mirror's last copy. A hung shared filesystem can block that copy
# for ever; the supervisor then exits anyway and leaves the mirror to Slurm.
MIRROR_FINISH_S = 5.0


def start_mirror(source: Path, target: Path, session_id: str) -> subprocess.Popen[Any]:
    """The one process that writes the session to the shared filesystem (llm_setup.mirror)."""
    env = child_environment(dict(os.environ), session_id, drop=("HF_TOKEN", *GATEWAY_SECRETS))
    with (source / "mirror.log").open("ab") as handle:
        return subprocess.Popen([sys.executable, "-m", "llm_setup.mirror", str(source), str(target)],
                                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)


def stop_mirror(session: Session) -> None:
    mirror = session.mirror
    if mirror is None or mirror.poll() is not None:
        return
    mirror.send_signal(signal.SIGTERM)
    try:
        mirror.wait(timeout=MIRROR_FINISH_S)
    except subprocess.TimeoutExpired:
        pass


def terminate(session: Session, say: Say = print, grace_s: float = TERMINATE_GRACE_S) -> list[str]:
    """Stop the session's own processes: SIGTERM, then SIGKILL after ``grace_s``; then the mirror's last
    copy (``stop_mirror``) once the caller has recorded the stop."""
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


def stop(runtime: Runtime | None = None, on_node: bool = False) -> list[str]:
    """Stop the running session from another shell. A foreground watchdog sees `current` disappear and
    exits without restarting anything. A shell that sees only the mirror on the shared filesystem (not
    the session's node-local folder) asks the session's mirror to stop it, within a minute. ``on_node``:
    the caller runs on the session's node and sees its folder (the mirror passing such a request on)."""
    runtime = runtime or Runtime.default()
    session_id = runtime.current_id()
    if not session_id:
        return []
    path = runtime.session_path(session_id)
    try:
        records = json.loads((path / "processes.json").read_text(encoding="utf-8"))["services"]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"cannot read {path}/processes.json ({exc}); nothing was signalled and session "
                           f"{session_id} stays current. Check its processes (`ps -eo pid,cmd`) before "
                           f"removing {runtime.current_file}") from exc
    meta = read_metadata(path)
    work = Path(meta["work"]) if meta.get("work") else path
    if not on_node and meta.get("mirror") and (meta.get("host") != socket.gethostname() or not work.is_dir()):
        # This shell sees only the shared copy: the session's mirror, on its node, does the stop.
        (path / STOP_REQUEST).write_text("llm-setup stop\n", encoding="utf-8")
        return [f"{name} (asked; stops within a minute)" for name in records]
    current = work.parent / "current"
    try:
        if current.read_text(encoding="utf-8").strip() == session_id:
            current.unlink()
    except FileNotFoundError:
        pass
    record_event(path, "session", "stop requested", "llm-setup stop")
    stopped = []
    for name, data in reversed(list(records.items())):
        if owned(int(data["pid"]), session_id):
            signal_owned(int(data["pid"]), signal.SIGTERM)
            stopped.append(name)
    return stopped
