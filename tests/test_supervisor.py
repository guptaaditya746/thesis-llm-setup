import os
import subprocess
import time
from pathlib import Path

from llm_setup.process import process_matches
from llm_setup.profile import load_profile
from llm_setup.supervisor import _command, _failure_hint, _vllm_environment, render_litellm


def test_litellm_config_has_aliases_without_cross_role_fallback(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    data = load_profile("profiles/a100-3x40.yaml")
    config = render_litellm(data)
    assert {item["model_name"] for item in config["model_list"]} == {
        "heavy-model", "lite-model", "qwen-embed"
    }
    assert config["router_settings"]["fallbacks"] == []
    # A saturated backend is not retried by the gateway (that only adds load).
    policy = config["router_settings"]["retry_policy"]
    assert policy["TimeoutErrorRetries"] == 0 and policy["RateLimitErrorRetries"] == 0
    assert config["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"
    for role, deployment in zip(("heavy", "lite", "embed"), config["model_list"]):
        assert deployment["litellm_params"]["model"] == f"openai/{deployment['model_name']}"
        limits = data["models"][role]
        assert deployment["litellm_params"]["max_parallel_requests"] == (
            limits["max_num_active_seqs"] + limits["max_num_queued_reqs"]
        )


def test_pid_ownership_requires_exact_session_marker():
    environment = os.environ.copy()
    environment["LLM_SETUP_SESSION_ID"] = "owned-session"
    process = subprocess.Popen(["sleep", "5"], env=environment)
    try:
        deadline = time.monotonic() + 2
        while not process_matches(process.pid, "owned-session") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert process_matches(process.pid, "owned-session")
        assert not process_matches(process.pid, "another-session")
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_embedding_launch_uses_pooling_runner(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    data = load_profile("profiles/a100-3x40.yaml")
    command = _command("embed", data["models"]["embed"], Path("runtime/test"))
    assert command[command.index("--runner") + 1] == "pooling"
    assert "--task" not in command


def test_vllm_environment_disables_flashinfer_sampler_without_toolkit():
    env = _vllm_environment({"LITELLM_MASTER_KEY": "secret", "HF_TOKEN": "hf-secret"},
                            {"gpu": 2}, "session-1")
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["HF_TOKEN"] == "hf-secret"
    assert "LITELLM_MASTER_KEY" not in env


def test_vllm_environment_respects_explicit_flashinfer_opt_in():
    env = _vllm_environment({"VLLM_USE_FLASHINFER_SAMPLER": "1"}, {"gpu": 0}, "session-1")
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_startup_failure_names_missing_cuda_compiler(tmp_path):
    log = tmp_path / "heavy.log"
    log.write_text("RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist")
    assert "VLLM_USE_FLASHINFER_SAMPLER=1" in _failure_hint(log)


def test_kv_cache_dtype_is_passed_to_chat_backends_only(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    data = load_profile("profiles/a100-3x40.yaml")
    heavy = _command("heavy", data["models"]["heavy"], Path("runtime/test"))
    assert heavy[heavy.index("--kv-cache-dtype") + 1] == "fp8"
    lite = _command("lite", {**data["models"]["lite"], "kv_cache_dtype": "auto"}, Path("runtime/test"))
    assert "--kv-cache-dtype" not in lite
    embed = _command("embed", data["models"]["embed"], Path("runtime/test"))
    assert "--kv-cache-dtype" not in embed


def test_tool_call_parser_enables_auto_tool_choice(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    data = load_profile("profiles/a100-3x40.yaml")
    lite = _command("lite", data["models"]["lite"], Path("runtime/test"))
    assert "--enable-auto-tool-choice" in lite
    assert lite[lite.index("--tool-call-parser") + 1] == "granite4"
    heavy = _command("heavy", data["models"]["heavy"], Path("runtime/test"))
    assert heavy[heavy.index("--tool-call-parser") + 1] == "hermes"
    embed = _command("embed", data["models"]["embed"], Path("runtime/test"))
    assert "--tool-call-parser" not in embed
    plain = _command("lite", {k: v for k, v in data["models"]["lite"].items() if k != "tool_call_parser"},
                     Path("runtime/test"))
    assert "--enable-auto-tool-choice" not in plain
