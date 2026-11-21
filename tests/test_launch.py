"""What gets launched: vLLM commands and environments, the gateway config, the installed-vLLM checks."""

import os
import socket
import subprocess
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from llm_setup import vllm
from llm_setup.doctor import tool_call_result
from llm_setup.gateway import render_litellm
from llm_setup.process import child_environment, owned, physical_gpu
from llm_setup.profile import ProfileError, load_profile
from llm_setup.supervisor import build_services

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
HELP = ("--host --port --gpu-memory-utilization --max-model-len --max-num-seqs --served-model-name "
        "--runner {auto,draft,generate,pooling} --tensor-parallel-size")


def three() -> object:
    return load_profile(PROFILES / "heavy-lite-embed.yaml", 3)


def instance(profile, name):
    return next(item for item in profile.instances if item.name == name)


def fake_help(monkeypatch, text):
    monkeypatch.setattr("llm_setup.vllm.shutil.which", lambda _name: "/env/bin/vllm")
    monkeypatch.setattr("llm_setup.vllm.subprocess.run",
                        lambda *_a, **_k: SimpleNamespace(stdout=text, stderr="", returncode=0))


# Gateway -------------------------------------------------------------------------------------------

def test_litellm_config_has_one_alias_per_model_and_no_fallbacks():
    profile = three()
    config = render_litellm(profile)
    assert [item["model_name"] for item in config["model_list"]] == ["heavy-model", "lite-model", "qwen-embed"]
    assert config["router_settings"]["fallbacks"] == []
    assert config["router_settings"]["routing_strategy"] == "least-busy"
    policy = config["router_settings"]["retry_policy"]
    assert policy["TimeoutErrorRetries"] == 0 and policy["RateLimitErrorRetries"] == 0
    assert config["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"
    for model, deployment in zip(profile.models.values(), config["model_list"]):
        assert deployment["litellm_params"]["max_parallel_requests"] == (
            model.max_num_active_seqs + model.max_num_queued_reqs)
    assert config["model_list"][2]["model_info"]["mode"] == "embedding"


def test_replicas_share_the_alias_and_get_pinned_aliases(monkeypatch):
    deployments = render_litellm(load_profile(PROFILES / "orch.yaml", 3))["model_list"]
    assert [(item["model_name"], item["litellm_params"]["api_base"]) for item in deployments] == [
        ("orch-model", "http://127.0.0.1:8001/v1"), ("orch-model-1", "http://127.0.0.1:8001/v1"),
        ("orch-model", "http://127.0.0.1:8002/v1"), ("orch-model-2", "http://127.0.0.1:8002/v1"),
        ("orch-model", "http://127.0.0.1:8003/v1"), ("orch-model-3", "http://127.0.0.1:8003/v1"),
    ]
    assert {item["litellm_params"]["model"] for item in deployments} == {"openai/orch-model"}
    assert [item["model_name"] for item in render_litellm(load_profile(PROFILES / "orch.yaml", 1))["model_list"]] \
        == ["orch-model"]
    monkeypatch.setenv("LITELLM_ROUTING_STRATEGY", "simple-shuffle")
    assert render_litellm(load_profile(PROFILES / "orch.yaml", 1))["router_settings"]["routing_strategy"] == \
        "simple-shuffle"


# vLLM command and environment ----------------------------------------------------------------------

def test_vllm_commands_carry_the_profile_settings():
    profile = three()
    embed = vllm.command(instance(profile, "embed"))
    assert embed[embed.index("--runner") + 1] == "pooling"
    assert "--kv-cache-dtype" not in embed and "--tool-call-parser" not in embed
    lite = vllm.command(instance(profile, "lite"))
    assert "--enable-auto-tool-choice" in lite and lite[lite.index("--tool-call-parser") + 1] == "granite4"
    heavy = instance(profile, "heavy")
    assert "--kv-cache-dtype" not in vllm.command(heavy)
    fp8 = replace(heavy, model=replace(heavy.model, kv_cache_dtype="fp8", revision="abc123"))
    command = vllm.command(fp8)
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
    assert command[command.index("--revision") + 1] == "abc123"
    orch = vllm.command(load_profile(PROFILES / "orch.yaml", 3).instances[2])
    assert orch[orch.index("--port") + 1] == "8003"
    assert orch[-5:] == ["--reasoning-parser", "qwen3", "--generation-config", "vllm", "--enable-prefix-caching"]


def test_vllm_environment_keeps_cluster_settings_and_drops_gateway_keys():
    cache = {name: f"/cluster/{name}" for name in ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "XDG_CACHE_HOME")}
    base = {**cache, "LITELLM_MASTER_KEY": "secret", "HF_TOKEN": "hf", "HF_HUB_OFFLINE": "1"}
    env = vllm.environment(base, instance(three(), "embed"), "s1")
    assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["HF_TOKEN"] == "hf" and env["HF_HUB_OFFLINE"] == "1" and env["LLM_SETUP_SESSION_ID"] == "s1"
    assert "LITELLM_MASTER_KEY" not in env and all(env[name] == value for name, value in cache.items())
    assert vllm.environment({"VLLM_USE_FLASHINFER_SAMPLER": "1"}, instance(three(), "heavy"), "s1")[
        "VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_allocation_gpu_indexes_map_to_the_devices_slurm_granted():
    assert physical_gpu({"CUDA_VISIBLE_DEVICES": "4,5,6"}, 1) == "5"
    assert physical_gpu({}, 1) == "1"
    assert child_environment({"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}, "s", gpu=1)["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    # An index beyond the allocation is refused, never mapped onto another job's device.
    with pytest.raises(RuntimeError, match="not in this allocation"):
        physical_gpu({"CUDA_VISIBLE_DEVICES": "4,5"}, 2)


def test_extra_args_cannot_override_what_the_launcher_sets():
    from llm_setup.profile import parse_profile, read_profile, resolve_gpus

    for args in (["--host", "0.0.0.0"], ["--port=9000"], ["--served-model-name", "x"]):
        data = resolve_gpus(read_profile(PROFILES / "orch.yaml"), 1)
        data["models"]["orch"]["extra_args"] = args
        with pytest.raises(ProfileError, match="the launcher sets it"):
            parse_profile(data)


def test_parallel_replicas_get_their_own_rendezvous_ports(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        base = probe.getsockname()[1]
    environment = {"VLLM_PORT": str(base)}
    services = build_services(load_profile(PROFILES / "orch.yaml", 3), environment, tmp_path, "s", "key")
    ports = [int(services[name].env["VLLM_PORT"]) for name in ("orch", "orch-2", "orch-3")]
    assert ports == [base, base + 4, base + 8]
    with ExitStack() as stack:
        for port in ports:
            listener = stack.enter_context(socket.socket())
            listener.bind(("127.0.0.1", port))
    assert environment == {"VLLM_PORT": str(base)}


def test_services_keep_secrets_where_they_belong(tmp_path):
    services = build_services(three(), {"LITELLM_MASTER_KEY": "k", "HF_TOKEN": "hf"}, tmp_path, "s1", "k")
    assert list(services) == ["heavy", "lite", "embed", "gateway", "status"]
    assert services["gateway"].env["LLM_BACKEND_KEY"] == "k" and "HF_TOKEN" not in services["gateway"].env
    assert "LITELLM_MASTER_KEY" not in services["status"].env and "HF_TOKEN" not in services["status"].env
    assert {item.kind for item in services.values()} == {"model", "gateway", "status"}


def test_pid_ownership_requires_the_exact_session_marker():
    process = subprocess.Popen(["sleep", "5"], env={**os.environ, "LLM_SETUP_SESSION_ID": "owned-session"})
    try:
        deadline = time.monotonic() + 2
        while not owned(process.pid, "owned-session") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owned(process.pid, "owned-session") and not owned(process.pid, "another-session")
    finally:
        process.terminate()
        process.wait(timeout=5)


# Installed vLLM ------------------------------------------------------------------------------------

def test_vllm_help_must_list_every_flag_the_profile_uses(monkeypatch):
    orch = load_profile(PROFILES / "orch.yaml", 1)
    fake_help(monkeypatch, HELP + " --reasoning-parser --generation-config --enable-prefix-caching")
    vllm.check_installed(orch)
    fake_help(monkeypatch, HELP + " --reasoning-parser --enable-prefix-caching")
    with pytest.raises(ProfileError, match="--generation-config"):
        vllm.check_installed(orch)


def test_tool_call_parsers_and_kv_flags_are_checked(monkeypatch):
    profile = three()
    tools = " --kv-cache-dtype --enable-auto-tool-choice --tool-call-parser {granite,granite4,hermes}"
    fake_help(monkeypatch, HELP + tools)
    vllm.check_installed(profile)
    fake_help(monkeypatch, HELP + tools.replace("granite4,", ""))
    with pytest.raises(ProfileError, match="granite4"):
        vllm.check_installed(profile)
    fake_help(monkeypatch, HELP)
    with pytest.raises(ProfileError, match="--tool-call-parser"):
        vllm.check_installed(profile)


def test_vllm_help_timeout_is_explained(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("llm_setup.vllm.shutil.which", lambda _name: "/env/bin/vllm")
    monkeypatch.setattr("llm_setup.vllm.subprocess.run", run)
    with pytest.raises(ProfileError, match="no services were started"):
        vllm.check_installed(three())
    assert calls == [600]


def test_fp8_kv_cache_is_refused_on_ampere_without_a_compiler(monkeypatch):
    profile = three()
    fp8 = replace(profile, models={**profile.models, "heavy": replace(profile.models["heavy"], kv_cache_dtype="fp8")})
    monkeypatch.setattr(vllm.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(vllm.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout="8.0\n8.0\n", returncode=0))
    with pytest.raises(ProfileError, match="kv_cache_dtype: auto"):
        vllm.check_fp8_kv_cache(fp8)
    vllm.check_fp8_kv_cache(profile)
    monkeypatch.setattr(vllm.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout="9.0\n", returncode=0))
    vllm.check_fp8_kv_cache(fp8)


def test_failure_hints_name_the_missing_compiler(tmp_path):
    log = tmp_path / "heavy.log"
    log.write_text("RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist")
    assert "VLLM_USE_FLASHINFER_SAMPLER=1" in vllm.failure_hint(log)
    log.write_text("Using FLASHINFER backend.\nRuntimeError: Could not find nvcc")
    assert "kv_cache_dtype: auto" in vllm.failure_hint(log)
    log.write_text("all fine")
    assert vllm.failure_hint(log) == "" and vllm.failure_hint(tmp_path / "missing.log") == ""


def test_tool_call_smoke_needs_a_parsed_call():
    def reply(message):
        return httpx.Response(200, json={"choices": [{"message": message}]})

    ok, detail = tool_call_result(reply({"tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'}}]}))
    assert ok and "Berlin" in detail
    ok, detail = tool_call_result(reply({"content": '<tool_call>{"name": "get_weather"}</tool_call>'}))
    assert not ok and "no parsed tool call" in detail
    assert not tool_call_result(reply(None))[0]
    assert not tool_call_result(httpx.Response(400, text="auto tool choice requires"))[0]
