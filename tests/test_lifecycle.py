"""Startup progress, stale sessions, the watchdog and doctor, with stand-in servers (no GPU)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from llm_setup.doctor import checks, report
from llm_setup.logs import current_phase
from llm_setup.runtime import Runtime, record_event
from llm_setup.supervisor import Service, Session, clear_stale, stop, wait_ready
from llm_setup.watchdog import Watchdog

PROFILES = Path(__file__).resolve().parents[1] / "profiles"

SERVER = """
import http.server, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *args):
        pass
print("Loading safetensors checkpoint shards:  50% Completed | 2/4", flush=True)
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def stand_in(tmp_path: Path, name: str, session_id: str, script: str = SERVER) -> Service:
    port = free_port()
    code = tmp_path / f"{name}.py"
    code.write_text(script)
    env = {**os.environ, "LLM_SETUP_SESSION_ID": session_id}
    kind = name if name in {"gateway", "status", "gliner"} else "model"
    return Service(name, kind, [sys.executable, str(code), str(port)], env, tmp_path / f"{name}.log",
                   f"http://127.0.0.1:{port}/health", timeout=20)


def test_phases_name_what_a_model_is_doing(tmp_path) -> None:
    log = tmp_path / "heavy.log"
    log.write_text("INFO non-default args: {...}\nLoading safetensors checkpoint shards:  25% Completed\r"
                   "Loading safetensors checkpoint shards:  63% Completed\n")
    assert current_phase(log) == "loading weights 63%"
    log.write_text("Loading weights took 41.2 seconds\nCapturing CUDA graphs (mixed prefill-decode): 40%|")
    assert current_phase(log) == "capturing CUDA graphs"
    log.write_text("something new\n")
    assert current_phase(log) == "something new"
    assert current_phase(tmp_path / "missing.log") == "no output yet"


def test_wait_ready_reports_progress_and_explains_an_exit(tmp_path) -> None:
    lines: list[str] = []
    good = stand_in(tmp_path, "lite", "s1")
    good.launch()
    try:
        wait_ready([good], lines.append, interval=0, poll=0.1)
    finally:
        good.process.terminate()
    assert any(line.startswith("lite: ready after") for line in lines)
    bad = stand_in(tmp_path, "heavy", "s1", "print('RuntimeError: CUDA out of memory'); raise SystemExit(3)")
    bad.launch()
    with pytest.raises(RuntimeError, match="heavy exited with status 3") as failure:
        wait_ready([bad], lines.append, poll=0.1)
    assert "CUDA out of memory" in str(failure.value)


def test_a_session_without_live_processes_is_cleared_and_a_live_one_blocks(tmp_path) -> None:
    root = tmp_path / "runtime"
    old = root / "20260101T000000Z-aaaaaaaa"
    old.mkdir(parents=True)
    (old / "processes.json").write_text(json.dumps({"sessionId": old.name, "services": {"heavy": {"pid": 999999}}}))
    (root / "current").write_text(old.name)
    said: list[str] = []
    clear_stale(Runtime(root), said.append)
    assert not (root / "current").exists() and "ended without `stop`" in said[0]
    assert json.loads((old / "events.jsonl").read_text().splitlines()[-1])["event"] == "cleared"
    live = subprocess.Popen(["sleep", "30"], env={**os.environ, "LLM_SETUP_SESSION_ID": old.name})
    try:
        time.sleep(0.2)
        (old / "processes.json").write_text(json.dumps({"sessionId": old.name, "services": {"lite": {"pid": live.pid}}}))
        (root / "current").write_text(old.name)
        with pytest.raises(RuntimeError, match="still running \\(lite\\)"):
            clear_stale(Runtime(root), said.append)
    finally:
        live.terminate()
        live.wait()


def test_watchdog_restarts_a_crashed_service_and_stops_everything_on_request(tmp_path) -> None:
    root = tmp_path / "runtime"
    session_id = "20260101T000000Z-bbbbbbbb"
    path = root / session_id
    path.mkdir(parents=True)
    (root / "current").write_text(session_id)
    gateway = stand_in(tmp_path, "gateway", session_id)
    session = Session(session_id, path, {"gateway": gateway})
    gateway.launch()
    wait_ready([gateway], lambda _: None, poll=0.1)
    session.save()
    said: list[str] = []
    watchdog = Watchdog(session, said.append, "http://127.0.0.1:9/v1/status", check_s=0.1, summary_s=3600,
                        runtime=Runtime(root), backoff_s=(0.1, 0.1))
    runner = threading.Thread(target=lambda: setattr(watchdog, "code", watchdog.run()))
    runner.start()
    first = gateway.process.pid
    gateway.process.send_signal(signal.SIGKILL)
    deadline = time.monotonic() + 20
    while not any("healthy again" in line for line in said) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert gateway.process.pid != first and gateway.state == "ready"
    watchdog.stop.set()
    runner.join(timeout=40)
    assert watchdog.code == 0 and gateway.process.poll() is not None
    assert not (root / "current").exists()
    events = [json.loads(line)["event"] for line in (path / "events.jsonl").read_text().splitlines()]
    assert events[:3] == ["exited", "restarting", "restarted"]
    assert events[-2:] == ["stopping", "stopped"]


def test_stop_from_another_shell_ends_the_watchdog_without_restarts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "runtime"
    session_id = "20260101T000000Z-cccccccc"
    path = root / session_id
    path.mkdir(parents=True)
    (root / "current").write_text(session_id)
    status = stand_in(tmp_path, "status", session_id)
    session = Session(session_id, path, {"status": status})
    status.launch()
    session.save()
    watchdog = Watchdog(session, lambda _: None, "http://127.0.0.1:9/v1/status", check_s=0.1,
                        runtime=Runtime(root), backoff_s=(0.1, 0.1))
    assert stop(Runtime(root)) == ["status"]
    assert watchdog.run() == 0
    assert "restarted" not in (path / "events.jsonl").read_text()


def test_doctor_names_the_first_problem(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_MASTER_KEY", "real-key")
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    (tmp_path / "hub" / "models--org--embed" / "snapshots" / "abc").mkdir(parents=True)

    def run(command: list[str]) -> str | None:
        return "0, 30000, 40960\n1, 10, 40960\n2, 10, 40960\n" if command[0] == "nvidia-smi" else None

    rows = checks(str(PROFILES / "heavy-lite-embed.yaml"), run=run, get=lambda *a, **k: None, root=tmp_path / "runtime")
    by_name = {name: (level, detail) for level, name, detail in rows}
    assert by_name["GPU 0 (heavy)"][0] == "FAIL" and "leftover process" in by_name["GPU 0 (heavy)"][1]
    assert by_name["GPU 1 (lite)"][0] == "PASS"
    assert by_name["weights embed"][0] == "PASS" and by_name["weights heavy"][0] == "WARN"
    out: list[str] = []
    assert report(rows, out.append) == 1
    assert out[-1].startswith("\nFirst problem: GPU 0 (heavy)")


def test_events_are_served_by_the_status_api(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from llm_setup.profile import load_profile
    from llm_setup.status import History, create_app

    record_event(tmp_path, "heavy", "exited", "status 1")
    app = create_app(load_profile(PROFILES / "heavy-lite-embed.yaml"), History(tmp_path / "s.sqlite3"), tmp_path)
    assert TestClient(app).get("/v1/events").json()["events"][0]["event"] == "exited"


FAKE_BINARY = """#!{python}
import http.server, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{{"data": []}}')
    def log_message(self, *args):
        pass
print("Loading weights took 1.0 seconds", flush=True)
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def test_start_launches_models_in_parallel_then_gateway_and_status(tmp_path, monkeypatch) -> None:
    """The whole start sequence with stand-ins for `vllm` and `litellm` on PATH."""
    from llm_setup.profile import load_profile, parse_profile, read_profile
    from llm_setup.supervisor import start, terminate

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("vllm", "litellm"):
        path = bin_dir / name
        path.write_text(FAKE_BINARY.format(python=sys.executable))
        path.chmod(0o755)
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    data = read_profile(PROFILES / "heavy-lite-embed.yaml")
    for item in data["models"].values():
        item["port"] = free_port()
    data["gateway_port"], data["status_port"] = free_port(), free_port()
    profile = parse_profile(data)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    said: list[str] = []
    session = start(profile, {"LITELLM_MASTER_KEY": "k", "PATH": os.environ["PATH"]},
                    say=said.append)
    try:
        assert (tmp_path / "runtime" / "current").read_text() == session.id
        assert all(item.state == "ready" for item in session.services.values())
        ready = [line.split(":")[0] for line in said if ": ready after" in line]
        assert set(ready[:3]) == {"embed", "heavy", "lite"} and ready[3:] == ["gateway", "status"]
        assert yaml.safe_load((session.path / "litellm-config.yaml").read_text())["model_list"]
        assert load_profile(session.path / "profile.yaml").instances == profile.instances
        with pytest.raises(RuntimeError, match="still running"):
            start(profile, {"LITELLM_MASTER_KEY": "k"}, say=said.append)
    finally:
        terminate(session, say=said.append, grace_s=5)
    # Once its processes are gone, the next start clears the old session first, then finds its ports free.
    assert not any(item.process.poll() is None for item in session.services.values())


def test_stop_keeps_the_session_current_when_its_process_record_is_unreadable(tmp_path) -> None:
    root = tmp_path / "runtime"
    session_id = "20260101T000000Z-dddddddd"
    (root / session_id).mkdir(parents=True)
    (root / "current").write_text(session_id)
    with pytest.raises(RuntimeError, match="nothing was signalled"):
        stop(Runtime(root))
    assert (root / "current").read_text() == session_id


def test_doctor_maps_allocation_gpus_to_node_devices(tmp_path, monkeypatch) -> None:
    """Without device cgroups the job sees the whole node: doctor must size and check only its own GPUs."""
    monkeypatch.setenv("LITELLM_MASTER_KEY", "real-key")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    node = "".join(f"{i}, {30000 if i < 4 else 10}, 40960\n" for i in range(8))   # other jobs on 0-3
    rows = checks(str(PROFILES / "orch.yaml"), run=lambda c: node if c[0] == "nvidia-smi" else None,
                  get=lambda *a, **k: None, root=tmp_path / "runtime")
    gpu_rows = {name: (level, detail) for level, name, detail in rows if name.startswith("GPU")}
    assert gpu_rows == {"GPU 0 (orch)": ("PASS", "device 4: 10/40960 MiB used"),
                        "GPU 1 (orch-2)": ("PASS", "device 5: 10/40960 MiB used")}


def test_logs_refuses_a_path_as_service_name(tmp_path, monkeypatch, capsys) -> None:
    from llm_setup.cli import main

    session = tmp_path / "runtime" / "20260101T000000Z-eeeeeeee"
    session.mkdir(parents=True)
    (session / "orch.log").write_text("hello\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as code:
        main(["logs", "orch"])
    assert code.value.code == 0 and "hello" in capsys.readouterr().out
    with pytest.raises(SystemExit) as code:
        main(["logs", "../../etc/x"])
    assert code.value.code == 1 and "not a service name" in capsys.readouterr().err
