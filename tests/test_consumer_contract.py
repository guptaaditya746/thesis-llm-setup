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


def test_gateway_alias_probe_authenticates_to_litellm():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from llm_setup.cli import _gateway_aliases

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer secret":
                self.send_response(401)
                self.end_headers()
                return
            body = b'{"data":[{"id":"heavy-model"},{"id":"lite-model"},{"id":"qwen-embed"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _gateway_aliases(server.server_address[1], "secret") == {
            "heavy-model", "lite-model", "qwen-embed"
        }
        assert _gateway_aliases(server.server_address[1], "") == set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
