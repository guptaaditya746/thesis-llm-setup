"""``llm-setup doctor``: every check in order, and the first problem in plain words.

Run it on the compute node (the services listen on 127.0.0.1 only). Each line is PASS, WARN, FAIL or
INFO; the exit status is 1 when anything failed.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .models import model_instances
from .profile import load_profile
from .supervisor import port_in_use, recorded_alive

Row = tuple[str, str, str]                     # level, check, detail


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


def weights_cached(model_id: str, cache: Path) -> bool:
    snapshots = cache / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def checks(profile_path: str, *, run: Callable[[list[str]], str | None] = _run,
           get: Callable[..., httpx.Response | None] = _get, root: Path = Path("runtime")) -> list[Row]:
    rows: list[Row] = []
    job = os.environ.get("SLURM_JOB_ID")
    rows.append(("INFO", "host", f"{socket.gethostname()}" + (f", Slurm job {job}" if job else ", not in a Slurm job")))
    if job:
        left = (run(["squeue", "-h", "-j", job, "-o", "%L"]) or "").strip()
        if left:
            short = left.count(":") < 2 and "-" not in left           # under an hour: "MM:SS"
            rows.append(("WARN" if short else "PASS", "time left in the allocation", left))

    key = os.environ.get("LITELLM_MASTER_KEY", "")
    rows.append(("FAIL", "LITELLM_MASTER_KEY", "not set; `set -a; source .env; set +a`") if not key else
                ("FAIL", "LITELLM_MASTER_KEY", "still the placeholder from .env.example")
                if key == "replace-with-a-local-secret" else ("PASS", "LITELLM_MASTER_KEY", "set"))

    try:
        profile = load_profile(profile_path)
        rows.append(("PASS", "profile", f"{profile['name']} ({profile_path})"))
    except Exception as exc:  # noqa: BLE001 - every problem is reported, not raised
        rows.append(("FAIL", "profile", str(exc)))
        return rows

    # The session: is one recorded, and is anything of it alive?
    current = root / "current"
    session_id = current.read_text(encoding="utf-8").strip() if current.exists() else ""
    alive = recorded_alive(root / session_id, session_id) if session_id else []
    if not session_id:
        rows.append(("INFO", "session", "none running"))
    elif alive:
        rows.append(("PASS", "session", f"{session_id}: {', '.join(alive)} alive"))
    else:
        rows.append(("WARN", "session", (f"{session_id} is recorded but none of its processes is alive "
                                         "(the allocation ended or it crashed); the next start clears it")))
    running = bool(alive)

    gpus = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"])
    if gpus is None:
        rows.append(("FAIL", "GPUs", "nvidia-smi unavailable: not on a GPU node, or no GPUs in this allocation"))
    else:
        found = {}
        for line in gpus.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                found[int(parts[0])] = (int(parts[1]), int(parts[2]))
        for item in model_instances(profile):
            role, gpu = item["name"], item["gpu"]
            if gpu not in found:
                rows.append(("FAIL", f"GPU {gpu} ({role})", f"not visible; the allocation shows {sorted(found)}"))
            elif not running and found[gpu][0] > 2048:
                rows.append(("FAIL", f"GPU {gpu} ({role})", (f"{found[gpu][0]} MiB already in use with no session "
                                                             "running: a leftover process (nvidia-smi lists it)")))
            else:
                rows.append(("PASS", f"GPU {gpu} ({role})", f"{found[gpu][0]}/{found[gpu][1]} MiB used"))

    cache = hf_cache()
    for role, item in profile["models"].items():
        if weights_cached(item["model"], cache):
            rows.append(("PASS", f"weights {role}", item["model"]))
        else:
            rows.append(("WARN", f"weights {role}", f"{item['model']} is not in {cache}; start will download it"))

    ports = {item["name"]: item["port"] for item in model_instances(profile)}
    ports.update(gateway=profile["gateway"]["port"], status=profile["status"]["port"])
    if not running:
        for name, port in ports.items():
            if port_in_use(port):
                rows.append(("FAIL", f"port {port} ({name})", ("in use with no session running: another server "
                                                               f"holds it (`ss -ltnp | grep :{port}`)")))
        return rows

    for item in model_instances(profile):
        role = item["name"]
        response = get(f"http://127.0.0.1:{item['port']}/health")
        rows.append(("PASS", f"{role} health", "ok") if response is not None and response.is_success else
                    ("FAIL", f"{role} health", "no answer; `llm-setup logs --service " + role + "`"))
    gateway = profile["gateway"]["port"]
    response = get(f"http://127.0.0.1:{gateway}/health/liveliness")
    rows.append(("PASS", "gateway", "alive") if response is not None and response.is_success else
                ("FAIL", "gateway", "no answer; `llm-setup logs --service gateway`"))
    response = get(f"http://127.0.0.1:{gateway}/v1/models", headers={"Authorization": f"Bearer {key}"})
    if response is None:
        rows.append(("FAIL", "gateway models", "no answer"))
    elif response.status_code in {401, 403}:
        rows.append(("FAIL", "gateway key", ("rejected: the LITELLM_MASTER_KEY in this shell differs from the "
                                             "gateway's (the harness's VLLM_API_KEY must equal it too)")))
    else:
        try:
            aliases = {entry.get("id") for entry in response.json().get("data", [])}
        except ValueError:
            aliases = set()
        missing = {item["alias"] for item in profile["models"].values()} - aliases
        rows.append(("FAIL", "gateway models", f"missing {', '.join(sorted(missing))}") if missing else
                    ("PASS", "gateway models", ", ".join(sorted(aliases))))
    response = get(f"http://127.0.0.1:{profile['status']['port']}/v1/status")
    try:
        overall = response.json().get("overall") if response is not None and response.is_success else None
    except ValueError:
        overall = None
    rows.append(("PASS" if overall in {"healthy", "busy", "recovering"} else "WARN" if overall else "FAIL",
                 "status API", overall or "no answer"))
    events = root / session_id / "events.jsonl"
    if events.exists():
        recent = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()[-5:] if line.strip()]
        restarts = [row for row in recent if row.get("event") in {"exited", "exited again", "gave up restarting"}]
        if restarts:
            rows.append(("WARN", "recent events", "; ".join(f"{row['time']} {row['service']} {row['event']}"
                                                            for row in restarts)))
    return rows


def report(rows: list[Row], say: Callable[[str], None] = print) -> int:
    for level, name, detail in rows:
        say(f"{level:<4}  {name}: {detail}")
    first = next((row for row in rows if row[0] == "FAIL"), None)
    if first:
        say(f"\nFirst problem: {first[1]}: {first[2]}")
        return 1
    warned = [row for row in rows if row[0] == "WARN"]
    say("\nNo failures" + (f", {len(warned)} warning(s)" if warned else "") + ".")
    return 0
