"""Parallel replicas must not inherit the same torch rendezvous listener port."""
import socket
from contextlib import ExitStack

from llm_setup.profile import load_profile
from llm_setup.supervisor import build_services


def test_parallel_replicas_can_bind_their_distributed_ports(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        base = probe.getsockname()[1]
    profile = load_profile("profiles/orch-3x40.yaml")
    environment = {"VLLM_PORT": str(base)}
    services = build_services(profile, environment, tmp_path, "port-test", "test-key")
    replicas = [services[name] for name in ("orch", "orch-2", "orch-3")]
    ports = [int(service.env["VLLM_PORT"]) for service in replicas]
    assert len(set(ports)) == 3, f"replicas share a distributed listener port: {ports}"
    assert ports == [base, base + 4, base + 8]
    with ExitStack() as stack:
        for port in ports:
            listener = stack.enter_context(socket.socket())
            listener.bind(("127.0.0.1", port))
            listener.listen()
    assert environment == {"VLLM_PORT": str(base)}
