"""The optional GLiNER service: profile section, server contract, launch and doctor (no GPU, no torch)."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_setup.profile import (
    GLINER_DEFAULTS,
    Gliner,
    ProfileError,
    load_profile,
    parse_profile,
    read_profile,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"


def orch_data(gpus: int = 3, **gliner) -> dict:
    data = read_profile(PROFILES / "orch.yaml")
    data["models"]["orch"]["gpus"] = list(range(gpus))
    if gliner:
        data["gliner"] = gliner
    return data


def orch_profile(gpus: int = 3, **gliner):
    return parse_profile(orch_data(gpus, **gliner))


class FakeModel:
    """Stands in for gliner.GLiNER: finds each requested label's word in the text."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], float]] = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls.append((text, list(labels), threshold))
        found = []
        for label, word, score in (("dataset", "SQuAD", 0.9), ("evaluation metric", "accuracy", 0.4),
                                   ("method", "RAG", 0.8)):
            start = text.find(word)
            if label in labels and start >= 0 and score >= threshold:
                found.append({"start": start, "end": start + len(word), "text": word, "label": label,
                              "score": score})
        return found


def client(model=None):
    from llm_setup.gliner_server import create_app

    model = model or FakeModel()
    return TestClient(create_app(model, "urchade/gliner_medium-v2.1", revision="abc123")), model


# Profile -------------------------------------------------------------------------------------------

def test_gliner_is_off_unless_a_profile_adds_it():
    assert load_profile(PROFILES / "orch.yaml", 3).gliner is None
    assert load_profile(PROFILES / "heavy-lite-embed.yaml", 3).gliner is None
    assert load_profile(PROFILES / "orch-gliner.yaml", 1).gliner is not None
    assert orch_profile(enabled=False).gliner is None
    assert load_profile(PROFILES / "orch-gliner.yaml", 3).extras() == ["gliner"]


def test_an_empty_section_gets_the_defaults_and_pinned_revision():
    gliner = parse_profile({**orch_data(), "gliner": {}}).gliner
    assert gliner == Gliner(gpu=0, port=8090, model=GLINER_DEFAULTS["model"], revision=GLINER_DEFAULTS["revision"],
                            max_chars=1500)
    # Another model without a revision is not given the default model's revision.
    assert orch_profile(model="org/other").gliner.revision is None


def test_gliner_section_is_validated():
    for section, message in [
        ({"enabled": "yes"}, "gliner.enabled"),
        ({"device": "cuda"}, "unsupported gliner fields"),
        ({"port": 8001}, "port 8001"),             # orch's port
        ({"port": 4000}, "port 4000"),             # the gateway's
        ({"gpu": 5}, "serves no model"),           # it shares a model's GPU
        ({"gpu": -1}, "gliner.gpu"),
        ({"model": ""}, "gliner.model"),
        ({"revision": " "}, "gliner.revision"),
        ({"max_chars": 0}, "gliner.max_chars"),
    ]:
        with pytest.raises(ProfileError, match=message):
            orch_profile(**section)
    with pytest.raises(ProfileError, match="gliner must be a mapping"):
        parse_profile({**orch_data(), "gliner": True})
    data = orch_data()
    data["models"]["gliner"] = data["models"].pop("orch")
    with pytest.raises(ProfileError, match="reserved service name"):
        parse_profile(data)
    assert orch_profile(gpu=2, port=8091, max_chars=3000).gliner.gpu == 2


def test_cli_lists_the_extras_a_profile_needs():
    run = [sys.executable, "-m", "llm_setup.cli", "profile", "extras", "--profile"]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("CUDA_VISIBLE_DEVICES", None)          # extras must not need the GPUs
    out = [subprocess.run([*run, str(PROFILES / name)], capture_output=True, text=True, env=env, check=True)
           .stdout.strip() for name in ("orch-gliner.yaml", "orch.yaml")]
    assert out == ["gliner", ""]


# Server --------------------------------------------------------------------------------------------

def test_predict_returns_the_client_contract():
    http, model = client()
    reply = http.post("/predict", json={"text": "RAG on SQuAD, accuracy 71%",
                                        "labels": ["dataset", "method", "evaluation metric"], "threshold": 0.5})
    assert reply.status_code == 200
    assert reply.json() == {"entities": [
        {"start": 0, "end": 3, "text": "RAG", "label": "method", "score": 0.8},
        {"start": 7, "end": 12, "text": "SQuAD", "label": "dataset", "score": 0.9},
    ]}
    assert model.calls == [("RAG on SQuAD, accuracy 71%", ["dataset", "method", "evaluation metric"], 0.5)]


def test_predict_sorts_and_types_what_the_model_returns():
    class Loose:
        def predict_entities(self, text, labels, threshold=0.5):
            return [{"start": 7, "end": 12, "text": "SQuAD", "label": "dataset", "score": 0.9, "extra": 1},
                    {"start": 0.0, "end": 3.0, "text": "RAG", "label": "method", "score": 1}]

    entities = client(Loose())[0].post("/predict", json={"text": "RAG on SQuAD", "labels": ["x"],
                                                          "threshold": 0.5}).json()["entities"]
    assert entities == [{"start": 0, "end": 3, "text": "RAG", "label": "method", "score": 1.0},
                        {"start": 7, "end": 12, "text": "SQuAD", "label": "dataset", "score": 0.9}]
    assert all(type(e["start"]) is int and type(e["score"]) is float for e in entities)


def test_predict_checks_its_input_and_skips_blank_text():
    http, model = client()
    assert http.post("/predict", json={"text": "x", "labels": [], "threshold": 0.5}).status_code == 422
    assert http.post("/predict", json={"text": "x", "labels": ["a"], "threshold": 1.5}).status_code == 422
    assert http.post("/predict", json={"labels": ["a"], "threshold": 0.5}).status_code == 422
    assert http.post("/predict", json={"text": "  ", "labels": ["a"], "threshold": 0.5}).json() == {"entities": []}
    assert model.calls == []
    # The threshold has a default, as GLiNER's own.
    assert http.post("/predict", json={"text": "SQuAD", "labels": ["dataset"]}).status_code == 200
    assert model.calls[-1][2] == 0.5


def test_predict_refuses_text_longer_than_max_chars():
    http, model = client()
    assert http.get("/info").json()["max_chars"] == 1500
    body = {"labels": ["dataset"], "threshold": 0.5}
    assert http.post("/predict", json={**body, "text": "SQuAD " + "x" * 1494}).status_code == 200   # 1500
    reply = http.post("/predict", json={**body, "text": "x" * 1501})
    assert reply.status_code == 413 and "at most 1500" in reply.json()["detail"]
    assert len(model.calls) == 1
    from llm_setup.gliner_server import create_app

    small = TestClient(create_app(FakeModel(), "m", max_chars=10))
    assert small.post("/predict", json={**body, "text": "SQuAD SQuAD"}).status_code == 413
    assert small.get("/info").json()["max_chars"] == 10


def test_info_names_the_served_model_and_health_answers():
    http, _ = client()
    info = http.get("/info").json()
    assert info["model"] == "urchade/gliner_medium-v2.1" and info["revision"] == "abc123"
    assert http.get("/health").status_code == 200


def test_server_module_does_not_import_torch_or_gliner():
    code = "import sys, llm_setup.gliner_server; print('torch' in sys.modules, 'gliner' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}).stdout
    assert out.strip() == "False False"


def test_load_model_is_pinned_offline_and_on_the_device(monkeypatch):
    from llm_setup import gliner_server

    seen = {}

    class Loaded:
        def to(self, device):
            seen["to"] = device
            return self

        def eval(self):
            seen["eval"] = True
            return self

    class GLiNER:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen["from_pretrained"] = (model_id, kwargs)
            return Loaded()

    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=GLiNER))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = gliner_server.load_model("urchade/gliner_medium-v2.1", "abc123", "cuda")
    assert isinstance(model, Loaded)
    model_id, kwargs = seen["from_pretrained"]
    assert model_id == "urchade/gliner_medium-v2.1"
    assert kwargs["revision"] == "abc123" and kwargs["local_files_only"] is True
    assert seen["to"] == "cuda" and seen["eval"]
    monkeypatch.delenv("HF_HUB_OFFLINE")
    gliner_server.load_model("m", None, "cpu")
    assert seen["from_pretrained"][1]["local_files_only"] is False


# Launch --------------------------------------------------------------------------------------------

def test_gliner_service_is_built_only_when_enabled(tmp_path):
    from llm_setup.supervisor import build_services

    env = {"LITELLM_MASTER_KEY": "k", "HF_TOKEN": "hf"}
    plain = build_services(orch_profile(), env, tmp_path, "s1", "k")
    assert list(plain) == ["orch", "orch-2", "orch-3", "gateway", "status"]
    services = build_services(orch_profile(gpu=2), env, tmp_path, "s1", "k")
    assert list(services) == ["orch", "orch-2", "orch-3", "gliner", "gateway", "status"]
    gliner = services["gliner"]
    assert gliner.command == [sys.executable, "-m", "llm_setup.gliner_server", "--model",
                              "urchade/gliner_medium-v2.1", "--port", "8090", "--max-chars", "1500", "--revision",
                              "40ec419335d09393f298636f471328b722c6da9e"]
    assert gliner.health == "http://127.0.0.1:8090/health"
    assert gliner.env["CUDA_VISIBLE_DEVICES"] == "2" and "HF_HUB_OFFLINE" not in gliner.env
    assert gliner.env["LLM_SETUP_SESSION_ID"] == "s1"
    assert "LITELLM_MASTER_KEY" not in gliner.env and "LLM_BACKEND_KEY" not in gliner.env
    assert gliner.log == tmp_path / "gliner.log"
    # The vLLM processes are launched exactly as without GLiNER.
    assert all(services[name].command == plain[name].command and services[name].env == plain[name].env
               for name in ("orch", "orch-2", "orch-3"))


def test_start_launches_gliner_after_the_models_and_before_the_gateway(tmp_path, monkeypatch):
    from llm_setup import supervisor

    order: list[list[str]] = []
    monkeypatch.setattr(supervisor.Service, "launch", lambda self: None)
    monkeypatch.setattr(supervisor, "wait_ready", lambda services, say: order.append([s.name for s in services]))
    monkeypatch.setattr(supervisor, "port_in_use", lambda port: False)
    monkeypatch.chdir(tmp_path)
    said: list[str] = []
    supervisor.start(orch_profile(gpu=0), {"LITELLM_MASTER_KEY": "k"},
                     say=said.append)
    assert order == [["orch", "orch-2", "orch-3"], ["gliner"], ["gateway"], ["status"]]
    assert any("gliner" in line and "GPU 0" in line for line in said)
    order.clear()
    supervisor.start(orch_profile(), {"LITELLM_MASTER_KEY": "k"}, say=said.append)
    assert order == [["orch", "orch-2", "orch-3"], ["gateway"], ["status"]]


def test_a_gliner_start_failure_leaves_the_stack_running(tmp_path, monkeypatch):
    import json

    from llm_setup import supervisor
    from llm_setup.watchdog import Watchdog

    launched: list[str] = []

    def wait_ready(services, say):
        if services[0].name == "gliner":
            raise RuntimeError("gliner exited with status 1 before it became healthy\n--- last lines ---")

    monkeypatch.setattr(supervisor.Service, "launch", lambda self: launched.append(self.name))
    monkeypatch.setattr(supervisor, "wait_ready", wait_ready)
    monkeypatch.setattr(supervisor, "port_in_use", lambda port: False)
    monkeypatch.chdir(tmp_path)
    said: list[str] = []
    session = supervisor.start(orch_profile(gpu=0), {"LITELLM_MASTER_KEY": "k"}, say=said.append)
    assert launched == ["orch", "orch-2", "orch-3", "gliner", "gateway", "status"]
    assert session.services["gliner"].state == "failed"
    assert (tmp_path / "runtime" / "current").read_text() == session.id
    assert any(line.startswith("gliner: start failed, continuing without it") for line in said)
    events = [json.loads(line) for line in (session.path / "events.jsonl").read_text().splitlines()]
    assert {"service": "gliner", "event": "start failed"}.items() <= next(
        e for e in events if e["service"] == "gliner").items()
    assert events[-1]["event"] == "ready"
    # The watchdog does not restart it.
    Watchdog(session, said.append, "http://127.0.0.1:9/v1/status").check(session.services["gliner"])
    assert launched.count("gliner") == 1


def test_start_refuses_a_busy_gliner_port(tmp_path, monkeypatch):
    from llm_setup import supervisor

    monkeypatch.setattr(supervisor, "port_in_use", lambda port: port == 8090)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="gliner :8090"):
        supervisor.start(orch_profile(gpu=0), {"LITELLM_MASTER_KEY": "k"}, say=lambda _: None)


# Doctor --------------------------------------------------------------------------------------------

def test_doctor_checks_gliner_weights_backbone_and_health(tmp_path, monkeypatch):
    import json

    import yaml

    from llm_setup.doctor import checks

    monkeypatch.setenv("LITELLM_MASTER_KEY", "real-key")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    snapshot = tmp_path / "hub" / "models--urchade--gliner_medium-v2.1" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "gliner_config.json").write_text(json.dumps({"model_name": "microsoft/deberta-v3-base"}))
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(orch_data(gpu=0)))

    def run(command):
        return "0, 10, 40960\n1, 10, 40960\n2, 10, 40960\n" if command[0] == "nvidia-smi" else None

    rows = {name: (level, detail) for level, name, detail in
            checks(str(path), run=run, get=lambda *a, **k: None, root=tmp_path / "runtime")}
    assert rows["profile"][0] == "PASS" and "orch-3 GPU 2 :8003" in rows["profile"][1]
    assert rows["weights gliner"][0] == "PASS"
    assert rows["weights gliner backbone"][0] == "WARN"
    assert "microsoft/deberta-v3-base" in rows["weights gliner backbone"][1]
    assert rows["GPU 0 (gliner)"][0] == "PASS"
    plain = {name for _, name, _ in checks(str(PROFILES / "orch.yaml"), run=run,
                                           get=lambda *a, **k: None, root=tmp_path / "runtime")}
    assert not any("gliner" in name for name in plain)

    # With a session running, its health is checked like a model's.
    runtime = tmp_path / "runtime"
    monkeypatch.setattr("llm_setup.doctor.alive_services", lambda *a: ["gliner"])
    runtime.mkdir()
    (runtime / "current").write_text("20261025T000000Z-abcdef12")
    rows = {name: level for level, name, _ in checks(str(path), run=run, get=lambda *a, **k: None, root=runtime)}
    assert rows["gliner health"] == "FAIL"
    (runtime / "20261025T000000Z-abcdef12").mkdir()
    from llm_setup.runtime import record_event

    record_event(runtime / "20261025T000000Z-abcdef12", "gliner", "start failed", "exited with status 1")
    rows = {name: (level, detail) for level, name, detail in
            checks(str(path), run=run, get=lambda *a, **k: None, root=runtime)}
    assert rows["recent events"][0] == "WARN" and "gliner start failed" in rows["recent events"][1]


def test_the_batch_job_installs_the_extras_the_profile_needs():
    script = (ROOT / "slurm" / "job.sh").read_text(encoding="utf-8")
    assert "profile extras --profile" in script
    assert "uv sync --python 3.13 --extra serve\n" in script     # the unchanged first install
