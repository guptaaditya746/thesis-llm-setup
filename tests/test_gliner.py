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
    ProfileError,
    gliner_settings,
    load_profile,
    validate_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def agent_profile(**gliner):
    data = load_profile(ROOT / "profiles" / "agent-3x40.yaml")
    if gliner:
        data["gliner"] = gliner
    return data


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

def test_existing_profiles_have_gliner_off_and_load_unchanged(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    for name in ("a100-3x40", "agent-3x40", "example"):
        data = load_profile(ROOT / "profiles" / f"{name}.yaml")
        assert "gliner" not in data          # the copied session profile stays as it was
        assert gliner_settings(data) is None


def test_enabled_gliner_gets_the_defaults_and_pinned_revision(monkeypatch):
    data = agent_profile(enabled=True)
    validate_profile(data)
    settings = gliner_settings(data)
    assert settings == {**GLINER_DEFAULTS, "enabled": True}
    assert settings["gpu"] == 0 and settings["port"] == 8090
    assert settings["model"] == "urchade/gliner_medium-v2.1"
    assert settings["revision"] == "40ec419335d09393f298636f471328b722c6da9e"
    # Another model without a revision is not given the default model's revision.
    assert gliner_settings(agent_profile(enabled=True, model="org/other"))["revision"] is None
    assert gliner_settings(agent_profile(enabled=False, gpu=1)) is None


def test_gliner_section_is_validated():
    for section, message in [
        ({"enabled": "yes"}, "gliner.enabled"),
        ({"enabled": True, "device": "cuda"}, "unsupported gliner fields"),
        ({"enabled": True, "port": 8001}, "port"),           # orch's port
        ({"enabled": True, "port": 4000}, "port"),           # the gateway's
        ({"enabled": True, "gpu": 5}, "GPU"),                # no model there: it shares a model's GPU
        ({"enabled": True, "gpu": -1}, "gliner.gpu"),
        ({"enabled": True, "model": ""}, "gliner.model"),
        ({"enabled": True, "revision": " "}, "gliner.revision"),
    ]:
        with pytest.raises(ProfileError, match=message):
            validate_profile(agent_profile(**section))
    data = agent_profile()
    data["models"]["gliner"] = data["models"].pop("sub")
    with pytest.raises(ProfileError, match="reserved service name"):
        validate_profile(data)
    with pytest.raises(ProfileError, match="gliner must be a mapping"):
        validate_profile({**agent_profile(), "gliner": True})
    validate_profile(agent_profile(enabled=True, gpu=2, port=8091))


def test_cli_lists_the_extras_a_profile_needs(monkeypatch, tmp_path):
    import yaml

    data = agent_profile(enabled=True)
    path = tmp_path / "gliner.yaml"
    path.write_text(yaml.safe_dump(data))
    run = [sys.executable, "-m", "llm_setup.cli", "profile", "extras", "--profile"]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    assert subprocess.run([*run, str(path)], capture_output=True, text=True, env=env,
                          check=True).stdout.strip() == "gliner"
    assert subprocess.run([*run, str(ROOT / "profiles" / "agent-3x40.yaml")], capture_output=True, text=True,
                          env=env, check=True).stdout.strip() == ""


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
    plain = build_services(agent_profile(), env, tmp_path, "s1", "k")
    assert list(plain) == ["orch", "sub", "sub-2", "gateway", "status"]
    services = build_services(agent_profile(enabled=True, gpu=2), env, tmp_path, "s1", "k")
    assert list(services) == ["orch", "sub", "sub-2", "gliner", "gateway", "status"]
    gliner = services["gliner"]
    assert gliner.command == [sys.executable, "-m", "llm_setup.gliner_server", "--model",
                              "urchade/gliner_medium-v2.1", "--port", "8090", "--revision",
                              "40ec419335d09393f298636f471328b722c6da9e"]
    assert gliner.health == "http://127.0.0.1:8090/health"
    assert gliner.env["CUDA_VISIBLE_DEVICES"] == "2" and gliner.env["HF_HUB_OFFLINE"] == "1"
    assert gliner.env["LLM_SETUP_SESSION_ID"] == "s1"
    assert "LITELLM_MASTER_KEY" not in gliner.env and "LLM_BACKEND_KEY" not in gliner.env
    assert gliner.log == tmp_path / "gliner.log"
    # The vLLM processes are launched exactly as without GLiNER.
    assert all(services[name].command == plain[name].command and services[name].env == plain[name].env
               for name in ("orch", "sub", "sub-2"))


def test_start_launches_gliner_after_the_models_and_before_the_gateway(tmp_path, monkeypatch):
    from llm_setup import supervisor

    order: list[list[str]] = []
    monkeypatch.setattr(supervisor.Service, "launch", lambda self: None)
    monkeypatch.setattr(supervisor, "wait_ready", lambda services, say: order.append([s.name for s in services]))
    monkeypatch.setattr(supervisor, "port_in_use", lambda port: False)
    monkeypatch.chdir(tmp_path)
    said: list[str] = []
    supervisor.start("p.yaml", agent_profile(enabled=True), {"LITELLM_MASTER_KEY": "k"}, say=said.append)
    assert order == [["orch", "sub", "sub-2"], ["gliner"], ["gateway"], ["status"]]
    assert any("gliner" in line and "GPU 0" in line for line in said)
    order.clear()
    supervisor.start("p.yaml", agent_profile(), {"LITELLM_MASTER_KEY": "k"}, say=said.append)
    assert order == [["orch", "sub", "sub-2"], ["gateway"], ["status"]]


def test_start_refuses_a_busy_gliner_port(tmp_path, monkeypatch):
    from llm_setup import supervisor

    monkeypatch.setattr(supervisor, "port_in_use", lambda port: port == 8090)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="gliner :8090"):
        supervisor.start("p.yaml", agent_profile(enabled=True), {"LITELLM_MASTER_KEY": "k"}, say=lambda _: None)


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
    path.write_text(yaml.safe_dump(agent_profile(enabled=True)))

    def run(command):
        return "0, 10, 40960\n1, 10, 40960\n2, 10, 40960\n" if command[0] == "nvidia-smi" else None

    rows = {name: (level, detail) for level, name, detail in
            checks(str(path), run=run, get=lambda *a, **k: None, root=tmp_path / "runtime")}
    assert rows["weights gliner"][0] == "PASS"
    assert rows["weights gliner backbone"][0] == "WARN"
    assert "microsoft/deberta-v3-base" in rows["weights gliner backbone"][1]
    assert rows["GPU 0 (gliner)"][0] == "PASS"
    plain = {name for _, name, _ in checks(str(ROOT / "profiles" / "agent-3x40.yaml"), run=run,
                                           get=lambda *a, **k: None, root=tmp_path / "runtime")}
    assert not any("gliner" in name for name in plain)

    # With a session running, its health is checked like a model's.
    runtime = tmp_path / "runtime"
    monkeypatch.setattr("llm_setup.doctor.recorded_alive", lambda *a: ["gliner"])
    runtime.mkdir()
    (runtime / "current").write_text("20261025T000000Z-abcdef12")
    rows = {name: level for level, name, _ in checks(str(path), run=run, get=lambda *a, **k: None, root=runtime)}
    assert rows["gliner health"] == "FAIL"


def test_sbatch_installs_the_extras_the_profile_enables():
    script = (ROOT / "slurm" / "llm-stack.sbatch").read_text(encoding="utf-8")
    assert "profile extras --profile" in script
    assert 'uv sync --python 3.13 --extra serve\n' in script     # the unchanged first install
