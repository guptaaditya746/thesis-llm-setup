import os
from pathlib import Path

import pytest
import yaml

from llm_setup.profile import ProfileError, load_profile, validate_profile


def profile():
    os.environ["EMBED_MODEL_ID"] = "org/qwen-embedder"
    return load_profile("profiles/a100-3x40.yaml")


def test_profile_loads_and_substitutes_embedding_model():
    data = profile()
    assert data["models"]["embed"]["model"] == "org/qwen-embedder"


def test_duplicate_gpu_or_port_is_rejected():
    data = profile()
    data["models"]["lite"]["gpu"] = 0
    with pytest.raises(ProfileError, match="GPU"):
        validate_profile(data)
    data = profile()
    data["models"]["lite"]["port"] = 8002
    with pytest.raises(ProfileError, match="port"):
        validate_profile(data)


def test_queue_configuration_must_be_finite_and_profile_fields_supported():
    data = profile()
    data["models"]["heavy"]["max_num_queued_reqs"] = 0
    with pytest.raises(ProfileError, match="positive integer"):
        validate_profile(data)
    data["models"]["heavy"]["max_num_queued_reqs"] = 7
    validate_profile(data)
    data["models"]["heavy"]["max_num_queued_request"] = 7
    with pytest.raises(ProfileError, match="unsupported fields"):
        validate_profile(data)


def test_profile_yaml_is_valid():
    yaml.safe_load(Path("profiles/a100-3x40.yaml").read_text(encoding="utf-8"))


def test_vllm_help_check_allows_slow_shared_filesystem_startup(monkeypatch):
    from types import SimpleNamespace

    from llm_setup.profile import validate_vllm_options

    monkeypatch.setattr("llm_setup.profile.shutil.which", lambda _name: "/env/bin/vllm")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        flags = "--host --port --gpu-memory-utilization --max-model-len --max-num-seqs " \
                "--served-model-name --runner {auto,draft,generate,pooling} --tensor-parallel-size"
        return SimpleNamespace(stdout=flags, stderr="", returncode=0)

    monkeypatch.setattr("llm_setup.profile.subprocess.run", run)
    validate_vllm_options()
    assert calls == [(["/env/bin/vllm", "serve", "--help=all"], 600)]


def test_vllm_help_timeout_has_actionable_error(monkeypatch):
    import subprocess

    from llm_setup.profile import validate_vllm_options

    monkeypatch.setattr("llm_setup.profile.shutil.which", lambda _name: "/env/bin/vllm")
    monkeypatch.setattr("llm_setup.profile.subprocess.run",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            subprocess.TimeoutExpired("vllm serve --help", 600)))
    with pytest.raises(ProfileError, match="no services were started"):
        validate_vllm_options()
