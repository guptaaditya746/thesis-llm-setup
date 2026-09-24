from llm_setup.process import process_matches
from llm_setup.profile import load_profile
from llm_setup.supervisor import render_litellm


def test_litellm_config_has_aliases_without_cross_role_fallback(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    data = load_profile("profiles/a100-3x40.yaml")
    config = render_litellm(data)
    assert {item["model_name"] for item in config["model_list"]} == {
        "heavy-model", "lite-model", "qwen-embed"
    }
    assert config["router_settings"]["fallbacks"] == []
    assert config["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"
    for role, deployment in zip(("heavy", "lite", "embed"), config["model_list"]):
        assert deployment["litellm_params"]["model"] == f"openai/{deployment['model_name']}"
        limits = data["models"][role]
        assert deployment["litellm_params"]["max_parallel_requests"] == (
            limits["max_num_active_seqs"] + limits["max_num_queued_reqs"]
        )


def test_pid_ownership_requires_exact_session_marker():
    import os
    import subprocess
    import time

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
