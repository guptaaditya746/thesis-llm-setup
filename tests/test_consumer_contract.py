from llm_setup.profile import load_profile
from llm_setup.status import StatusMonitor


class EmptyHistory:
    def add(self, alias, data, now=None):
        pass


def test_consumer_status_json_schema(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/embed")
    profile = load_profile("profiles/a100-3x40.yaml")
    monitor = StatusMonitor(profile, EmptyHistory())
    monitor.last = {alias: {"state": "unknown", "backend": f"http://127.0.0.1:{port}"}
                    for alias, port in (("heavy-model", 8002), ("lite-model", 8003), ("qwen-embed", 8001))}
    result = monitor.document()
    assert set(result) == {"overall", "updatedAt", "models"}
    assert set(result["models"]) == {"heavy-model", "lite-model", "qwen-embed"}
    assert result["models"]["heavy-model"]["backend"] == "http://127.0.0.1:8002"
