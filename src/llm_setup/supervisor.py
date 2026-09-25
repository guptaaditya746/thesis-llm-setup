"""Session process startup and cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from .models import model_targets
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
    for alias, target in model_targets(profile).items():
        task = "embedding" if target["role"] == "embed" else "chat"
        limits = profile["models"][target["role"]]
        models.append({"model_name": alias, "litellm_params": {
            "model": f"openai/{alias}",
            "api_base": target["backend"] + "/v1", "api_key": "os.environ/LLM_BACKEND_KEY",
            "drop_params": True,
            "max_parallel_requests": limits["max_num_active_seqs"] + limits["max_num_queued_reqs"],
        }, "model_info": {"mode": task}})
    return {"model_list": models,
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            "litellm_settings": {"request_timeout": 120},
            "router_settings": {"routing_strategy": "simple-shuffle", "num_retries": 1,
                                "retry_policy": RETRY_POLICY, "fallbacks": []}}


def _command(role: str, item: dict[str, Any], session: Path) -> list[str]:
    command = ["vllm", "serve", item["model"], "--host", "127.0.0.1", "--port", str(item["port"]),
               "--served-model-name", item["alias"], "--tensor-parallel-size", "1",
               "--gpu-memory-utilization", str(item["gpu_memory_utilization"]),
               "--max-model-len", str(item["max_model_len"]),
               "--max-num-seqs", str(item["max_num_active_seqs"])]
    if role == "embed":
        command += ["--runner", "pooling"]
    kv_cache_dtype = item.get("kv_cache_dtype", "auto")
    if role != "embed" and kv_cache_dtype != "auto":
        command += ["--kv-cache-dtype", kv_cache_dtype]
    # Tool calling (the harness agent loop sends tool_choice="auto") needs the
    # model's parser; without it vLLM answers HTTP 400 for tool requests.
    if role != "embed" and item.get("tool_call_parser"):
        command += ["--enable-auto-tool-choice", "--tool-call-parser", item["tool_call_parser"]]
    return command


def _vllm_environment(base: dict[str, str], item: dict[str, Any], session_id: str) -> dict[str, str]:
    child_env = base.copy()
    child_env.pop("LITELLM_MASTER_KEY", None)
    child_env.pop("LLM_BACKEND_KEY", None)
    child_env.pop("VLLM_API_KEY", None)
    child_env["HF_TOKEN"] = base.get("HF_TOKEN", "")
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


def _wait_health(url: str, process: subprocess.Popen[Any], timeout: int = 1800,
                 log_path: Path | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            hint = _failure_hint(log_path)
            raise RuntimeError(
                f"process exited with status {process.returncode} before {url} became healthy{hint}"
            )
        try:
            if httpx.get(url, timeout=2).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for {url}")


def start(profile_path: str | Path, profile: dict[str, Any], env: dict[str, str]) -> Path:
    key = env.get("LITELLM_MASTER_KEY", "")
    if not key or key == "replace-with-a-local-secret":
        raise RuntimeError("set a non-placeholder LITELLM_MASTER_KEY before start")
    root = Path("runtime")
    root.mkdir(exist_ok=True)
    if (root / "current").exists():
        active = (root / "current").read_text(encoding="utf-8").strip()
        raise RuntimeError(f"session {active} is recorded as current; run stop or verify before starting another")
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    session = root / session_id
    session.mkdir()
    (session / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    litellm = render_litellm(profile)
    litellm_path = session / "litellm-config.yaml"
    litellm_path.write_text(yaml.safe_dump(litellm, sort_keys=False), encoding="utf-8")
    records: dict[str, Any] = {}
    (session / "metadata.json").write_text(json.dumps({"sessionId": session_id,
        "profile": str(profile_path), "services": list(profile["models"]), "environmentVariables": [
            "HF_TOKEN", "LITELLM_MASTER_KEY", "LLM_BACKEND_KEY", "LLM_SETUP_SESSION_ID"]}, indent=2),
        encoding="utf-8")
    try:
        for role in ("embed", "heavy", "lite"):
            item = profile["models"][role]
            command = _command(role, item, session)
            child_env = _vllm_environment({**os.environ, **env}, item, session_id)
            log_path = session / f"{role}.log"
            log = log_path.open("ab")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=child_env,
                                       start_new_session=True)
            records[role] = {"pid": process.pid, "command": command, "log": str(session / f"{role}.log")}
            _save_records(session, session_id, records)
            _wait_health(f"http://127.0.0.1:{item['port']}/health", process, log_path=log_path)
        litellm_command = ["litellm", "--config", str(litellm_path), "--host", "127.0.0.1",
                           "--port", str(profile["gateway"]["port"])]
        gateway_env = os.environ.copy()
        gateway_env.pop("HF_TOKEN", None)
        gateway_env["LLM_SETUP_SESSION_ID"] = session_id
        gateway_env["LLM_BACKEND_KEY"] = key
        gateway_env["LITELLM_MASTER_KEY"] = key
        child = subprocess.Popen(litellm_command, stdout=(session / "gateway.log").open("ab"),
                                 stderr=subprocess.STDOUT, env=gateway_env, start_new_session=True)
        records["gateway"] = {"pid": child.pid, "command": litellm_command, "log": str(session / "gateway.log")}
        _save_records(session, session_id, records)
        _wait_health(f"http://127.0.0.1:{profile['gateway']['port']}/health/liveliness", child, 120)
        status_command = [sys.executable, "-m", "llm_setup.status_server", str(session / "profile.yaml"),
                          "--database", str(session / "status.sqlite3"), "--session", str(session)]
        status_env = os.environ.copy()
        status_env.pop("HF_TOKEN", None)
        status_env.pop("LITELLM_MASTER_KEY", None)
        status_env.pop("LLM_BACKEND_KEY", None)
        status_env.pop("VLLM_API_KEY", None)
        status_env["LLM_SETUP_SESSION_ID"] = session_id
        status = subprocess.Popen(status_command, stdout=(session / "status.log").open("ab"),
                                  stderr=subprocess.STDOUT, env=status_env, start_new_session=True)
        records["status"] = {"pid": status.pid, "command": status_command, "log": str(session / "status.log")}
        _save_records(session, session_id, records)
        _wait_health(f"http://127.0.0.1:{profile['status']['port']}/healthz", status, 60)
        (root / "current").write_text(session_id, encoding="utf-8")
        return session
    except Exception as exc:
        _terminate_records(session_id, records, root)
        raise RuntimeError(f"startup failed in session {session_id}: {exc}; logs: {session}") from exc


def _save_records(session: Path, session_id: str, records: dict[str, Any]) -> None:
    (session / "processes.json").write_text(json.dumps({"sessionId": session_id, "services": records}, indent=2))


def _terminate_records(session_id: str, records: dict[str, Any], root: Path) -> None:
    for data in reversed(list(records.values())):
        pid = data["pid"]
        if process_matches(pid, session_id, root):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def stop() -> list[str]:
    root = Path("runtime")
    current = root / "current"
    if not current.exists():
        return []
    session_id = current.read_text(encoding="utf-8").strip()
    if not session_id or any(char not in "0123456789T-Zabcdef-" for char in session_id):
        return ["invalid runtime/current session identifier; no processes were signalled"]
    session = root / session_id
    state_file = session / "processes.json"
    if not state_file.exists():
        return [f"missing process state for {session_id}"]
    state = json.loads(state_file.read_text(encoding="utf-8"))
    stopped = []
    for service, data in reversed(list(state["services"].items())):
        pid = int(data["pid"])
        if process_matches(pid, session_id, root):
            os.kill(pid, signal.SIGTERM)
            stopped.append(service)
    current.unlink()
    return stopped
