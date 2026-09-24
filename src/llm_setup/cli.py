"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import yaml

from .history import History
from .models import model_targets
from .profile import ProfileError, load_profile, validate_vllm_options
from .status import StatusMonitor
from .supervisor import render_litellm, start, stop


def _active_session() -> Path | None:
    current = Path("runtime/current")
    if not current.exists():
        return None
    session_id = current.read_text(encoding="utf-8").strip()
    if not session_id or any(char not in "0123456789T-Zabcdef-" for char in session_id):
        raise RuntimeError("runtime/current contains an invalid session identifier")
    return Path("runtime") / session_id


def _verify(profile_path: str) -> int:
    profile = load_profile(profile_path)
    try:
        validate_vllm_options()
        vllm = "available"
    except ProfileError as exc:
        vllm = str(exc)
    generated = yaml.safe_load(yaml.safe_dump(render_litellm(profile)))
    alias_targets = {item["model_name"]: item["litellm_params"]["api_base"]
                     for item in generated["model_list"]}
    expected_targets = {alias: target["backend"] + "/v1" for alias, target in model_targets(profile).items()}
    config_ok = alias_targets == expected_targets and not generated["router_settings"]["fallbacks"]
    aliases = set(alias_targets)
    print(f"Profile: valid ({profile['name']})")
    print(f"LiteLLM configuration: {'valid' if config_ok else 'invalid'}; aliases: "
          f"{', '.join(sorted(aliases))}; cross-role fallbacks: none")
    print(f"vLLM: {vllm}")
    session = _active_session()
    if session and (session / "processes.json").exists():
        state = json.loads((session / "processes.json").read_text())
        for name, data in state["services"].items():
            from .process import process_matches
            owned = process_matches(int(data["pid"]), state["sessionId"])
            print(f"Process {name}: PID {data['pid']} {'owned/alive' if owned else 'stale or not owned'}")
    else:
        print("Process state: no active session")
    all_ok = vllm == "available" and config_ok
    for alias, target in model_targets(profile).items():
        for endpoint in ("/health", "/metrics"):
            try:
                response = httpx.get(target["backend"] + endpoint, timeout=2)
                print(f"{alias} {endpoint}: HTTP {response.status_code}")
                all_ok = all_ok and response.is_success
            except httpx.HTTPError:
                print(f"{alias} {endpoint}: unreachable")
                all_ok = False
    try:
        response = httpx.get(f"http://127.0.0.1:{profile['gateway']['port']}/v1/models", timeout=3)
        found = {entry.get("id") for entry in response.json().get("data", [])} if response.is_success else set()
        for alias in aliases:
            okay = alias in found
            print(f"LiteLLM alias {alias}: {'reachable' if okay else 'missing'}")
            all_ok = all_ok and okay
    except (httpx.HTTPError, ValueError):
        print("LiteLLM model aliases: gateway unreachable")
    return 0 if all_ok else 1


def _smoke(profile_path: str) -> int:
    profile = load_profile(profile_path)
    key = os.getenv("LITELLM_MASTER_KEY", "")
    if not key:
        print("LITELLM_MASTER_KEY is not set")
        return 1
    base = f"http://127.0.0.1:{profile['gateway']['port']}/v1/"
    failures = 0
    with httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"}) as client:
        for alias, request in (
            ("heavy-model", {"messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 8}),
            ("lite-model", {"messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 8}),
            ("qwen-embed", {"input": ["smoke test"]}),
        ):
            started = time.monotonic()
            try:
                endpoint = "embeddings" if alias == "qwen-embed" else "chat/completions"
                response = client.post(urljoin(base, endpoint), json={"model": alias, **request})
                success = response.is_success
                backend = model_targets(profile)[alias]["backend"]
                detail = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                success, backend, detail = False, model_targets(profile)[alias]["backend"], str(exc)
            elapsed = time.monotonic() - started
            print(f"{alias} backend={backend} elapsed={elapsed:.2f}s {'OK' if success else 'FAILED'} {detail}")
            failures += not success
    return int(bool(failures))


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-setup")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("profile", help="profile operations")
    validate_sub = validate.add_subparsers(dest="profile_command", required=True)
    profile_validate = validate_sub.add_parser("validate")
    profile_validate.add_argument("--profile", required=True)
    for command in ("start", "verify", "smoke"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--profile", required=True)
    sub.add_parser("status")
    sub.add_parser("stop")
    logs = sub.add_parser("logs")
    logs.add_argument("--service", required=True)
    args = parser.parse_args()
    try:
        if args.command == "profile":
            profile = load_profile(args.profile)
            print(f"Profile valid: {profile['name']}")
        elif args.command == "start":
            profile = load_profile(args.profile, check_vllm=True)
            session = start(args.profile, profile, os.environ)
            print(f"Started session {session}")
        elif args.command == "verify":
            sys.exit(_verify(args.profile))
        elif args.command == "smoke":
            sys.exit(_smoke(args.profile))
        elif args.command == "stop":
            print("Stopped: " + (", ".join(stop()) or "no active owned processes"))
        elif args.command == "logs":
            if args.service not in {"heavy", "lite", "embed", "gateway", "status"}:
                raise RuntimeError("service must be one of heavy, lite, embed, gateway, status")
            session = _active_session()
            if not session:
                raise RuntimeError("no active session")
            log = session / f"{args.service}.log"
            if not log.exists():
                raise RuntimeError(f"no log for service {args.service}")
            print(log.read_text(errors="replace")[-20000:])
        elif args.command == "status":
            session = _active_session()
            if not session:
                raise RuntimeError("no active session")
            profile = load_profile(session / "profile.yaml")
            monitor = StatusMonitor(profile, History(session / "status.sqlite3"))
            import asyncio
            report = asyncio.run(monitor.poll_once())
            state_path = session / "processes.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"services": {}}
            services = state.get("services", {})
            print("ALIAS        PID    STATE        RUNNING WAITING KV CACHE BACKEND")
            for alias, item in report["models"].items():
                role = {"heavy-model": "heavy", "lite-model": "lite", "qwen-embed": "embed"}[alias]
                entry = services.get(role, {})
                pid = entry.get("pid", "-")
                if isinstance(pid, int):
                    from .process import process_matches
                    if not process_matches(pid, state.get("sessionId", "")):
                        pid = "stale"
                print(f"{alias:<12} {pid!s:<6} {item['state']:<12} {item.get('running', '-'):>7} "
                      f"{item.get('waiting', '-'):>7} "
                      f"{item.get('kvCache', '-'):>8} {item['backend']}")
    except (ProfileError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
