"""``llm-setup doctor`` and ``llm-setup smoke``: every check in order, and the first problem in plain words.

Run them on the compute node (the services listen on 127.0.0.1 only):
``srun --jobid <jobid> --overlap --pty bash -l``. Each doctor line is PASS, WARN, FAIL or INFO; the
exit status is 1 when anything failed.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .process import physical_gpu
from .profile import Profile, load_profile, visible_gpu_count
from .runtime import Runtime, alive_services, read_events
from .supervisor import port_in_use

Row = tuple[str, str, str]                     # level, check, detail
Say = Callable[[str], None]


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _get(url: str, **kwargs: Any) -> httpx.Response | None:
    try:
        return httpx.get(url, timeout=3, **kwargs)
    except httpx.HTTPError:
        return None


def hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _snapshots(model_id: str, cache: Path) -> Path:
    return cache / ("models--" + model_id.replace("/", "--")) / "snapshots"


def weights_cached(model_id: str, cache: Path) -> bool:
    snapshots = _snapshots(model_id, cache)
    return snapshots.is_dir() and any(snapshots.iterdir())


def _gliner_weights(model_id: str, cache: Path) -> list[Row]:
    """GLiNER may load offline: its weights and its backbone's tokenizer (named in gliner_config.json)
    must both be in the cache."""
    if not weights_cached(model_id, cache):
        return [("FAIL", "weights gliner", f"{model_id} is not in {cache} (docs/OPERATIONS.md, GLiNER)")]
    rows: list[Row] = [("PASS", "weights gliner", model_id)]
    backbone = None
    for config in sorted(_snapshots(model_id, cache).glob("*/gliner_config.json")):
        try:
            backbone = json.loads(config.read_text(encoding="utf-8")).get("model_name")
        except (OSError, ValueError):
            continue
        if backbone:
            break
    if backbone and not weights_cached(backbone, cache):
        rows.append(("WARN", "weights gliner backbone",
                     f"{backbone} (its tokenizer) is not in {cache}; GLiNER fails to load offline without it"))
    elif backbone:
        rows.append(("PASS", "weights gliner backbone", backbone))
    return rows


def _visible_gpus(output: str | None) -> dict[int, tuple[int, int]]:
    found = {}
    for line in (output or "").strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            found[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return found


def checks(profile_path: str, *, gpu_count: int | None = None, run: Callable[[list[str]], str | None] = _run,
           get: Callable[..., httpx.Response | None] = _get, root: Path = Path("runtime")) -> list[Row]:
    rows: list[Row] = []
    job = os.environ.get("SLURM_JOB_ID")
    rows.append(("INFO", "host", socket.gethostname() + (f", Slurm job {job}" if job else ", not in a Slurm job")))
    if job:
        left = (run(["squeue", "-h", "-j", job, "-o", "%L"]) or "").strip()
        if left:
            short = left.count(":") < 2 and "-" not in left           # under an hour: "MM:SS"
            rows.append(("WARN" if short else "PASS", "time left in the allocation", left))

    key = os.environ.get("LITELLM_MASTER_KEY", "")
    rows.append(("FAIL", "LITELLM_MASTER_KEY", "not set; `set -a; source .env; set +a`") if not key else
                ("FAIL", "LITELLM_MASTER_KEY", "still the placeholder from .env.example")
                if key == "replace-with-a-local-secret" else ("PASS", "LITELLM_MASTER_KEY", "set"))

    output = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"])
    found = _visible_gpus(output)
    # Size `gpus: all` as start does: from the allocation (CUDA_VISIBLE_DEVICES), not the whole node.
    if gpu_count is None:
        gpu_count = visible_gpu_count() if "CUDA_VISIBLE_DEVICES" in os.environ else (len(found) or None)
    try:
        profile = load_profile(profile_path, gpu_count)
        rows.append(("PASS", "profile", f"{profile.name} ({profile_path}): " + ", ".join(
            f"{item.name} GPU {item.gpu} :{item.port}" for item in profile.instances)))
    except Exception as exc:  # noqa: BLE001 - every problem is reported, not raised
        rows.append(("FAIL", "profile", str(exc)))
        return rows

    runtime = Runtime.default() if root == Path("runtime") else Runtime(root)
    session_id = runtime.current_id()
    alive = alive_services(runtime.session_path(session_id), session_id) if session_id else []
    if not session_id:
        rows.append(("INFO", "session", "none running"))
    elif alive:
        rows.append(("PASS", "session", f"{session_id}: {', '.join(alive)} alive"))
    else:
        rows.append(("WARN", "session", (f"{session_id} is recorded but none of its processes is alive "
                                         "(the allocation ended or it crashed); the next start clears it")))
    running = bool(alive)

    if output is None:
        rows.append(("FAIL", "GPUs", "nvidia-smi unavailable: not on a GPU node, or no GPUs in this allocation"))
    else:
        for name, gpu in profile.gpus():
            # Allocation index -> the device nvidia-smi numbers (they differ without device cgroups).
            try:
                device = physical_gpu(dict(os.environ), gpu)
            except RuntimeError as exc:
                rows.append(("FAIL", f"GPU {gpu} ({name})", str(exc)))
                continue
            index = int(device) if device.isdigit() else None
            if index is None:
                rows.append(("INFO", f"GPU {gpu} ({name})", f"device {device}"))
            elif index not in found:
                rows.append(("FAIL", f"GPU {gpu} ({name})", f"device {index} not visible; nvidia-smi shows {sorted(found)}"))
            elif not running and found[index][0] > 2048:
                rows.append(("FAIL", f"GPU {gpu} ({name})", (f"{found[index][0]} MiB already in use on device {index} "
                                                             "with no session running: a leftover process")))
            else:
                rows.append(("PASS", f"GPU {gpu} ({name})", f"device {index}: {found[index][0]}/{found[index][1]} MiB used"))

    cache = hf_cache()
    for role, model in profile.models.items():
        rows.append(("PASS", f"weights {role}", model.model) if weights_cached(model.model, cache) else
                    ("WARN", f"weights {role}", f"{model.model} is not in {cache}; start will download it"))
    if profile.gliner:
        rows.extend(_gliner_weights(profile.gliner.model, cache))

    if not running:
        for name, port in profile.ports().items():
            if port_in_use(port):
                rows.append(("FAIL", f"port {port} ({name})",
                             f"in use with no session running: another server holds it (`ss -ltnp | grep :{port}`)"))
        return rows

    for name, url in [(item.name, item.url) for item in profile.instances] + (
            [("gliner", f"http://127.0.0.1:{profile.gliner.port}")] if profile.gliner else []):
        response = get(f"{url}/health")
        rows.append(("PASS", f"{name} health", "ok") if response is not None and response.is_success else
                    ("FAIL", f"{name} health", f"no answer; `llm-setup logs {name}`"))
    gateway = f"http://127.0.0.1:{profile.gateway_port}"
    response = get(f"{gateway}/health/liveliness")
    rows.append(("PASS", "gateway", "alive") if response is not None and response.is_success else
                ("FAIL", "gateway", "no answer; `llm-setup logs gateway`"))
    response = get(f"{gateway}/v1/models", headers={"Authorization": f"Bearer {key}"})
    if response is None:
        rows.append(("FAIL", "gateway models", "no answer"))
    elif response.status_code in {401, 403}:
        rows.append(("FAIL", "gateway key", ("rejected: the LITELLM_MASTER_KEY in this shell differs from the "
                                             "gateway's (a client's API key must equal it too)")))
    else:
        try:
            aliases = {entry.get("id") for entry in response.json().get("data", [])}
        except ValueError:
            aliases = set()
        missing = {model.alias for model in profile.models.values()} - aliases
        rows.append(("FAIL", "gateway models", f"missing {', '.join(sorted(missing))}") if missing else
                    ("PASS", "gateway models", ", ".join(sorted(aliases))))
    response = get(f"http://127.0.0.1:{profile.status_port}/v1/status")
    try:
        overall = response.json().get("overall") if response is not None and response.is_success else None
    except ValueError:
        overall = None
    rows.append(("PASS" if overall in {"healthy", "busy", "recovering"} else "WARN" if overall else "FAIL",
                 "status API", overall or "no answer"))
    trouble = [row for row in read_events(runtime.session_path(session_id), 5)
               if row.get("event") in {"exited", "exited again", "gave up restarting", "start failed"}]
    if trouble:
        rows.append(("WARN", "recent events", "; ".join(f"{row['time']} {row['service']} {row['event']}"
                                                        for row in trouble)))
    return rows


def report(rows: list[Row], say: Say = print) -> int:
    for level, name, detail in rows:
        say(f"{level:<4}  {name}: {detail}")
    first = next((row for row in rows if row[0] == "FAIL"), None)
    if first:
        say(f"\nFirst problem: {first[1]}: {first[2]}")
        return 1
    warned = sum(row[0] == "WARN" for row in rows)
    say("\nNo failures" + (f", {warned} warning(s)" if warned else "") + ".")
    return 0


# Smoke test ----------------------------------------------------------------------------------------

TOOL_REQUEST = {
    "messages": [{"role": "user", "content": "What is the weather in Berlin? Use the tool."}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}],
    "tool_choice": "auto",
    "max_tokens": 64,
    "temperature": 0,
}


def tool_call_result(response: httpx.Response) -> tuple[bool, str]:
    """OK only when the reply carries a parsed get_weather call with JSON arguments. A missing or wrong
    parser leaves ``<tool_call>`` text in the content instead, which an agent loop cannot use."""
    if not response.is_success:
        return False, f"HTTP {response.status_code}: {response.text[:160]}"
    try:
        message = ((response.json().get("choices") or [{}])[0].get("message")) or {}
        calls = message.get("tool_calls") or []
        function = calls[0].get("function") or {} if calls else {}
        arguments = json.loads(function.get("arguments") or "null") if calls else None
    except (ValueError, AttributeError, IndexError, TypeError) as exc:
        return False, f"HTTP {response.status_code}, unreadable tool call: {exc}"
    if function.get("name") == "get_weather" and isinstance(arguments, dict):
        return True, f"HTTP {response.status_code}, tool call get_weather({json.dumps(arguments)})"
    content = str(message.get("content") or "")[:120]
    return False, f"HTTP {response.status_code}, no parsed tool call (check tool_call_parser); content: {content!r}"


def smoke(profile: Profile, key: str, say: Say = print) -> int:
    """One request per alias through the gateway, plus one tool call per model with a parser.
    Returns the number of failures."""
    if not key:
        say("LITELLM_MASTER_KEY is not set")
        return 1
    base = f"http://127.0.0.1:{profile.gateway_port}/v1"
    failures = 0
    with httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"}) as client:
        for model in profile.models.values():
            requests = [("embeddings", {"input": ["smoke test"]}, None)] if model.embed else [
                ("chat/completions", {"messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 8}, None)]
            if model.tool_call_parser:
                requests.append(("chat/completions", TOOL_REQUEST, tool_call_result))
            for endpoint, body, judge in requests:
                started = time.monotonic()
                try:
                    response = client.post(f"{base}/{endpoint}", json={"model": model.alias, **body})
                    ok, detail = judge(response) if judge else (response.is_success, f"HTTP {response.status_code}")
                except httpx.HTTPError as exc:
                    ok, detail = False, str(exc)
                what = f"tool calling ({model.tool_call_parser})" if judge else endpoint
                say(f"{model.alias} {what} {time.monotonic() - started:.2f}s {'OK' if ok else 'FAILED'} {detail}")
                failures += not ok
    return failures
