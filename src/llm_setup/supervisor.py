"""Session process startup and cleanup."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from .models import model_instances
from .phases import current_phase, tail
from .process import process_matches

# A timeout or a rate limit means the backend is already saturated; retrying
# it from the gateway only adds load and hides the wait from the caller, who
# owns its own deadline. Only transient server errors are retried, once.
RETRY_POLICY = {
    "TimeoutErrorRetries": 0,
    "RateLimitErrorRetries": 0,
    "BadRequestErrorRetries": 0,
    "AuthenticationErrorRetries": 0,
    "ContentPolicyViolationErrorRetries": 0,
    "InternalServerErrorRetries": 1,
    "ServiceUnavailableErrorRetries": 1,
}


def render_litellm(profile: dict[str, Any]) -> dict[str, Any]:
    models = []
    instance_numbers: dict[str, int] = {}
    for target in model_instances(profile):
        alias = target["alias"]
        role = target["role"]
        instance_numbers[role] = instance_numbers.get(role, 0) + 1
        task = "embedding" if target.get("task") == "embed" else "chat"
        deployment = {"model_name": alias, "litellm_params": {
            "model": f"openai/{alias}",
            "api_base": f"http://127.0.0.1:{target['port']}/v1", "api_key": "os.environ/LLM_BACKEND_KEY",
            "drop_params": True,
            "max_parallel_requests": target["max_num_active_seqs"] + target["max_num_queued_reqs"],
        }, "model_info": {"mode": task}}
        models.append(deployment)
        if profile["models"][role].get("replicas"):
            pinned = {**deployment, "model_name": f"{alias}-{instance_numbers[role]}"}
            models.append(pinned)
    routing_strategy = os.environ.get("LITELLM_ROUTING_STRATEGY", "least-busy")
    return {"model_list": models,
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            "litellm_settings": {"request_timeout": 600},
            "router_settings": {"routing_strategy": routing_strategy, "num_retries": 1,
                                "retry_policy": RETRY_POLICY, "fallbacks": []}}


def _command(role: str, item: dict[str, Any], session: Path) -> list[str]:
    command = ["vllm", "serve", item["model"], "--host", "127.0.0.1", "--port", str(item["port"]),
               "--served-model-name", item["alias"], "--tensor-parallel-size", "1",
               "--gpu-memory-utilization", str(item["gpu_memory_utilization"]),
               "--max-model-len", str(item["max_model_len"]),
               "--max-num-seqs", str(item["max_num_active_seqs"])]
    if item.get("task") == "embed":
        command += ["--runner", "pooling"]
    kv_cache_dtype = item.get("kv_cache_dtype", "auto")
    if item.get("task") != "embed" and kv_cache_dtype != "auto":
        command += ["--kv-cache-dtype", kv_cache_dtype]
    # Tool calling (the harness agent loop sends tool_choice="auto") needs the
    # model's parser; without it vLLM answers HTTP 400 for tool requests.
    if item.get("task") != "embed" and item.get("tool_call_parser"):
        command += ["--enable-auto-tool-choice", "--tool-call-parser", item["tool_call_parser"]]
    if item.get("revision"):
        command += ["--revision", item["revision"]]
    command.extend(item.get("extra_args", []))
    return command


def _vllm_environment(base: dict[str, str], item: dict[str, Any], session_id: str) -> dict[str, str]:
    child_env = base.copy()
    child_env.pop("LITELLM_MASTER_KEY", None)
    child_env.pop("LLM_BACKEND_KEY", None)
    child_env.pop("VLLM_API_KEY", None)
    child_env["HF_TOKEN"] = base.get("HF_TOKEN", "")
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["LLM_SETUP_SESSION_ID"] = session_id
    child_env["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
    # FlashInfer's sampler JIT-compiles on first request and requires nvcc. Many
    # Slurm runtime images have CUDA drivers but not the CUDA toolkit. The native
    # PyTorch sampler avoids that runtime compiler dependency. Operators may
    # explicitly opt back in when nvcc and a compatible toolkit are available.
    child_env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return child_env


def _failure_hint(log_path: Path | None) -> str:
    if log_path is None:
        return ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
    except OSError:
        return ""
    fp8_kv_markers = ("FLASHINFER backend", "FlashInfer attention", "kv_cache_dtype=fp8",
                      "kv_cache_dtype='fp8'", "--kv-cache-dtype fp8")
    if "Could not find nvcc" in tail and any(marker in tail for marker in fp8_kv_markers):
        return (
            "; FlashInfer attention kernels could not be compiled because nvcc is unavailable. "
            "This happens with kv_cache_dtype fp8 on pre-Hopper GPUs: set kv_cache_dtype: auto "
            "in the profile, or install flashinfer-jit-cache"
        )
    if "Could not find nvcc" in tail:
        return (
            "; FlashInfer sampler JIT failed because nvcc is unavailable. "
            "Native PyTorch sampling is now enabled by default; set "
            "VLLM_USE_FLASHINFER_SAMPLER=1 only when a compatible CUDA toolkit is installed"
        )
    return ""


# How long a service may take to answer its health check after launch.
STARTUP_TIMEOUT_S = {"embed": 1800, "heavy": 1800, "lite": 1800, "gateway": 180, "status": 60}
Say = Callable[[str], None]


def clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass
class Service:
    """One supervised process: how to launch it and how to tell it is ready."""

    name: str
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
        handle = self.log.open("ab")
        self.process = subprocess.Popen(self.command, stdout=handle, stderr=subprocess.STDOUT, env=self.env,
                                        start_new_session=True)
        handle.close()
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
        records = {name: {"pid": item.process.pid, "command": item.command, "log": str(item.log)}
                   for name, item in self.services.items() if item.process is not None}
        _save_records(self.path, self.id, records)


def build_services(profile: dict[str, Any], env: dict[str, str], session: Path, session_id: str,
                   key: str) -> dict[str, Service]:
    base = {**os.environ, **env}
    services: dict[str, Service] = {}
    for item in model_instances(profile):
        name, role = item["name"], item["role"]
        services[name] = Service(name, _command(role, item, session), _vllm_environment(base, item, session_id),
                                 session / f"{name}.log", f"http://127.0.0.1:{item['port']}/health",
                                 max(STARTUP_TIMEOUT_S.get(role, 1800), 1800))
    gateway_env = {k: v for k, v in base.items() if k != "HF_TOKEN"}
    gateway_env.update({"LLM_SETUP_SESSION_ID": session_id, "LLM_BACKEND_KEY": key, "LITELLM_MASTER_KEY": key})
    services["gateway"] = Service(
        "gateway", ["litellm", "--config", str(session / "litellm-config.yaml"), "--host", "127.0.0.1",
                    "--port", str(profile["gateway"]["port"])], gateway_env, session / "gateway.log",
        f"http://127.0.0.1:{profile['gateway']['port']}/health/liveliness", STARTUP_TIMEOUT_S["gateway"])
    status_env = {k: v for k, v in base.items()
                  if k not in {"HF_TOKEN", "LITELLM_MASTER_KEY", "LLM_BACKEND_KEY", "VLLM_API_KEY"}}
    status_env["LLM_SETUP_SESSION_ID"] = session_id
    services["status"] = Service(
        "status", [sys.executable, "-m", "llm_setup.status_server", str(session / "profile.yaml"),
                   "--database", str(session / "status.sqlite3"), "--session", str(session)],
        status_env, session / "status.log", f"http://127.0.0.1:{profile['status']['port']}/healthz",
        STARTUP_TIMEOUT_S["status"])
    return services


def wait_ready(services: list[Service], say: Say, interval: float = 15.0, poll: float = 2.0) -> None:
    """Wait until every service answers its health check, printing what each is doing.

    A line per ``interval`` names each service's phase (read from its log); a service that exits or
    exceeds its timeout raises with the end of its log and a hint.
    """
    waiting = list(services)
    last_report = time.monotonic()
    while waiting:
        for service in list(waiting):
            code = service.exited()
            if code is not None:
                raise RuntimeError(f"{service.name} exited with status {code} before it became healthy"
                                   f"{_failure_hint(service.log)}\n--- last lines of {service.log} ---\n"
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


def recorded_alive(session_dir: Path, session_id: str) -> list[str]:
    """Services of a recorded session whose process is still running (and carries its marker)."""
    state_file = session_dir / "processes.json"
    if not state_file.exists():
        return []
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [name for name, data in state.get("services", {}).items()
            if process_matches(int(data["pid"]), session_id)]


def clear_stale(root: Path, say: Say) -> None:
    """Forget a recorded session none of whose processes is alive (an allocation that ended, a
    terminal that closed). A session with live processes still blocks a new start."""
    current = root / "current"
    if not current.exists():
        return
    session_id = current.read_text(encoding="utf-8").strip()
    if not _valid_session_id(session_id):
        raise RuntimeError("runtime/current holds an invalid session id; remove it after checking runtime/")
    alive = recorded_alive(root / session_id, session_id)
    if alive:
        raise RuntimeError(f"session {session_id} is still running ({', '.join(alive)}); "
                           "run `llm-setup stop` first, or `scancel` its Slurm job")
    current.unlink()
    record_event(root / session_id, "session", "cleared", "no recorded process was alive at the next start")
    say(f"previous session {session_id} ended without `stop` (no process alive); cleared")


def record_event(session: Path, service: str, event: str, detail: str = "") -> dict[str, str]:
    row = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "service": service, "event": event,
           "detail": detail}
    try:
        with (session / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        pass
    return row


def start(profile_path: str | Path, profile: dict[str, Any], env: dict[str, str], say: Say = print) -> Session:
    """Launch the three models in parallel, then the gateway and the status API, with progress."""
    key = env.get("LITELLM_MASTER_KEY", "")
    if not key or key == "replace-with-a-local-secret":
        raise RuntimeError("set a non-placeholder LITELLM_MASTER_KEY before start")
    root = Path("runtime")
    root.mkdir(exist_ok=True)
    clear_stale(root, say)
    ports = {item["name"]: item["port"] for item in model_instances(profile)}
    ports.update(gateway=profile["gateway"]["port"], status=profile["status"]["port"])
    busy = [f"{name} :{port}" for name, port in ports.items() if port_in_use(port)]
    if busy:
        raise RuntimeError(f"ports already in use: {', '.join(busy)}; another server (or a session that was "
                           "not stopped) holds them. `llm-setup doctor` shows more")
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path = root / session_id
    path.mkdir()
    (path / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    (path / "litellm-config.yaml").write_text(yaml.safe_dump(render_litellm(profile), sort_keys=False),
                                              encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({"sessionId": session_id,
        "profile": str(profile_path), "services": list(profile["models"]), "host": socket.gethostname(),
        "slurmJobId": os.environ.get("SLURM_JOB_ID"), "environmentVariables": [
            "HF_TOKEN", "LITELLM_MASTER_KEY", "LLM_BACKEND_KEY", "LLM_SETUP_SESSION_ID"]}, indent=2),
        encoding="utf-8")
    session = Session(session_id, path, build_services(profile, env, path, session_id, key))
    say(f"session {session_id} on {socket.gethostname()}; logs in {path}")
    record_event(path, "session", "starting", socket.gethostname())
    try:
        models = [service for name, service in session.services.items() if name not in {"gateway", "status"}]
        for service in models:
            service.launch()
        session.save()
        say("models: " + ", ".join(f"{item['name']} on GPU {item['gpu']}" for item in model_instances(profile))
            + " (loading in parallel; usually 5-20 min)")
        wait_ready(models, say)
        for name in ("gateway", "status"):
            session.services[name].launch()
            session.save()
            wait_ready([session.services[name]], say)
        (root / "current").write_text(session_id, encoding="utf-8")
        record_event(path, "session", "ready")
        say(f"ready: gateway http://127.0.0.1:{profile['gateway']['port']}/v1 · "
            f"status http://127.0.0.1:{profile['status']['port']}/v1/status")
        return session
    except BaseException as exc:
        terminate(session, say=lambda _: None)
        record_event(path, "session", "start failed", str(exc).splitlines()[0][:300])
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"startup failed in session {session_id}: {exc}\nlogs: {path}") from exc


def terminate(session: Session, say: Say = print, grace_s: float = 30.0) -> list[str]:
    """Stop the session's own processes: SIGTERM, then SIGKILL after ``grace_s``."""
    stopped = []
    for service in reversed(list(session.services.values())):
        process = service.process
        if process is not None and process.poll() is None and process_matches(process.pid, session.id):
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
            if process_matches(process.pid, session.id):
                say(f"{service.name} did not stop in {int(grace_s)} s; killing it")
                signal_owned(process.pid, signal.SIGKILL)
        service.state = "stopped"
    return stopped


def signal_owned(pid: int, number: int) -> None:
    """Signal a session process and, when it leads its own process group (start_new_session), the
    whole group: vLLM starts engine workers that must stop with it."""
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, number)
        else:
            os.kill(pid, number)
    except ProcessLookupError:
        pass


def _valid_session_id(session_id: str) -> bool:
    return bool(session_id) and all(char in "0123456789T-Zabcdef-" for char in session_id)


def _save_records(session: Path, session_id: str, records: dict[str, Any]) -> None:
    (session / "processes.json").write_text(json.dumps({"sessionId": session_id, "services": records}, indent=2))


def stop() -> list[str]:
    root = Path("runtime")
    current = root / "current"
    if not current.exists():
        return []
    session_id = current.read_text(encoding="utf-8").strip()
    if not _valid_session_id(session_id):
        return ["invalid runtime/current session identifier; no processes were signalled"]
    session = root / session_id
    state_file = session / "processes.json"
    if not state_file.exists():
        return [f"missing process state for {session_id}"]
    state = json.loads(state_file.read_text(encoding="utf-8"))
    # Unlink first: a foreground supervisor sees the session end and does not restart what stops.
    current.unlink()
    record_event(session, "session", "stop requested", "llm-setup stop")
    stopped = []
    for service, data in reversed(list(state["services"].items())):
        pid = int(data["pid"])
        if process_matches(pid, session_id, root):
            signal_owned(pid, signal.SIGTERM)
            stopped.append(service)
    return stopped
