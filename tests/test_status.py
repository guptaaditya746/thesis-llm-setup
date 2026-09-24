from llm_setup.history import History
from llm_setup.profile import load_profile
from llm_setup.status import StatusMonitor, classify


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
        profile = load_profile("profiles/a100-3x40.yaml")
        for item, server in zip(profile["models"].values(), servers):
            item["port"] = server.server_address[1]
        monitor = StatusMonitor(profile, History(tmp_path / "status.sqlite3"))
        report = asyncio.run(monitor.poll_once())
        assert report["overall"] == "healthy"
        assert all(item["running"] == 1 for item in report["models"].values())
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
