from pathlib import Path

from llm_setup.profile import load_profile, parse_profile, read_profile
from llm_setup.status import History, StatusMonitor, classify

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def test_status_classification_and_recovery():
    assert classify("unknown", True, {}) == "healthy"
    assert classify("healthy", True, {"waiting": 2}) == "busy"
    assert classify("healthy", True, {"waiting": 5}) == "saturated"
    assert classify("healthy", False, {}) == "unavailable"
    assert classify("unavailable", True, {}, last_bad_at=90, now=100) == "recovering"
    assert classify("unavailable", True, {}, last_bad_at=1, now=100) == "healthy"


def test_sqlite_retention_prunes_old_samples(tmp_path):
    history = History(tmp_path / "history.sqlite3", retention_seconds=10)
    history.add("heavy-model", {"state": "old"}, now=1)
    history.add("heavy-model", {"state": "new"}, now=20)
    assert history.recent("heavy-model", seconds=86400, now=20) == [{"state": "new"}]


def test_status_polls_fake_http_backends(tmp_path):
    import asyncio
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"ok" if self.path == "/health" else b"vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(3)]
    threads = [Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        data = read_profile(PROFILES / "heavy-lite-embed.yaml")
        for item, server in zip(data["models"].values(), servers):
            item["port"] = server.server_address[1]
        profile = parse_profile(data)
        monitor = StatusMonitor(profile, History(tmp_path / "status.sqlite3"))
        report = asyncio.run(monitor.poll_once())
        assert report["overall"] == "healthy"
        assert all(item["running"] == 1 for item in report["models"].values())
        assert set(report["models"]) == {"heavy-model", "lite-model", "qwen-embed"}
        assert set(report) == {"overall", "updatedAt", "models"}
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def test_replicas_are_reported_by_alias_and_port():
    monitor = StatusMonitor(load_profile(PROFILES / "orch.yaml", 3), History(":memory:"))
    assert monitor.targets == {f"orch-model@{port}": f"http://127.0.0.1:{port}" for port in (8001, 8002, 8003)}
