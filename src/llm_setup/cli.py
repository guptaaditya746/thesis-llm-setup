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
from .models import model_instances, model_targets
from .profile import ProfileError, load_profile, validate_vllm_options
from .status import StatusMonitor
from .supervisor import clock, render_litellm, start, stop


def _active_session() -> Path | None:
    current = Path("runtime/current")
    if not current.exists():
        return None
    session_id = current.read_text(encoding="utf-8").strip()
    if not session_id or any(char not in "0123456789T-Zabcdef-" for char in session_id):
        raise RuntimeError("runtime/current contains an invalid session identifier")
    return Path("runtime") / session_id


def _latest_session() -> Path | None:
    """The current session, or else the newest one (so events of a stopped session stay readable)."""
    current = _active_session()
    if current:
        return current
    folders = sorted(path for path in Path("runtime").glob("2*") if path.is_dir())
    return folders[-1] if folders else None


def _gateway_aliases(port: int, key: str) -> set[str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = httpx.get(f"http://127.0.0.1:{port}/v1/models", headers=headers, timeout=3)
    if not response.is_success:
        return set()
    return {entry.get("id") for entry in response.json().get("data", [])}


def _verify(profile_path: str) -> int:
    profile = load_profile(profile_path)
    try:
        validate_vllm_options(profile)
        vllm = "available"
    except ProfileError as exc:
        vllm = str(exc)
    generated = yaml.safe_load(yaml.safe_dump(render_litellm(profile)))
    alias_targets = {(item["model_name"], item["litellm_params"]["api_base"])
                     for item in generated["model_list"]}
    expected_targets = {(item["alias"], f"http://127.0.0.1:{item['port']}/v1")
                        for item in model_instances(profile)}
    config_ok = alias_targets == expected_targets and not generated["router_settings"]["fallbacks"]
    aliases = {alias for alias, _ in alias_targets}
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
    for target in model_instances(profile):
        alias = target["alias"]
        for endpoint in ("/health", "/metrics"):
            try:
                response = httpx.get(f"http://127.0.0.1:{target['port']}" + endpoint, timeout=2)
                print(f"{alias} {endpoint}: HTTP {response.status_code}")
                all_ok = all_ok and response.is_success
            except httpx.HTTPError:
                print(f"{alias} {endpoint}: unreachable")
                all_ok = False
    try:
        found = _gateway_aliases(profile["gateway"]["port"], os.getenv("LITELLM_MASTER_KEY", ""))
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
        for alias, target in model_targets(profile).items():
            request = ({"input": ["smoke test"]} if target.get("task") == "embed" else
                       {"messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 8})
            started = time.monotonic()
            try:
                endpoint = "embeddings" if target.get("task") == "embed" else "chat/completions"
                response = client.post(urljoin(base, endpoint), json={"model": alias, **request})
                success = response.is_success
                backend = target["backend"]
                detail = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                success, backend, detail = False, target["backend"], str(exc)
            elapsed = time.monotonic() - started
            print(f"{alias} backend={backend} elapsed={elapsed:.2f}s {'OK' if success else 'FAILED'} {detail}")
            failures += not success
        failures += _smoke_tool_calls(client, base, profile)
    return int(bool(failures))


TOOL_SMOKE = {
    "messages": [{"role": "user", "content": "What is the weather in Berlin? Use the tool."}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}],
    "tool_choice": "auto",
    "max_tokens": 64,
    "temperature": 0,
}


def _tool_call_result(response: httpx.Response) -> tuple[bool, str]:
    """OK only when the reply carries a parsed get_weather call with JSON arguments.

    A missing or wrong parser leaves ``<tool_call>`` text in the content instead,
    which the harness agent loop cannot use.
    """
    if not response.is_success:
        return False, f"HTTP {response.status_code}: {response.text[:160]}"
    try:
        message = ((response.json().get("choices") or [{}])[0].get("message")) or {}
        calls = message.get("tool_calls") or []
        function = calls[0].get("function") or {} if calls else {}
        arguments = json.loads(function.get("arguments") or "null") if calls else None
    except (ValueError, AttributeError, IndexError, TypeError) as exc:
        return False, f"HTTP {response.status_code}, unreadable tool call: {exc}"
    if function.get("name") == "get_weather" and isinstance(arguments, dict):
        return True, f"HTTP {response.status_code}, tool call get_weather({json.dumps(arguments)})"
    content = str(message.get("content") or "")[:120]
    return False, (f"HTTP {response.status_code}, no parsed tool call (check tool_call_parser); "
                   f"content: {content!r}")


def _smoke_tool_calls(client: httpx.Client, base: str, profile: dict) -> int:
    """One tool-calling request per role that serves a tool-call parser (the agent loop needs it)."""
    failures = 0
    for item in profile["models"].values():
        if item.get("task") == "embed" or not item.get("tool_call_parser"):
            continue
        alias = item["alias"]
        started = time.monotonic()
        try:
            response = client.post(urljoin(base, "chat/completions"), json={"model": alias, **TOOL_SMOKE})
            success, detail = _tool_call_result(response)
        except httpx.HTTPError as exc:
            success, detail = False, str(exc)
        print(f"{alias} tool calling ({item['tool_call_parser']}) elapsed={time.monotonic() - started:.2f}s "
              f"{'OK' if success else 'FAILED'} {detail}")
        failures += not success
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-setup")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("profile", help="profile operations")
    validate_sub = validate.add_subparsers(dest="profile_command", required=True)
    profile_validate = validate_sub.add_parser("validate")
    profile_validate.add_argument("--profile", required=True)
    for command in ("start", "verify", "smoke", "doctor"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--profile", required=True)
        if command == "start":
            cmd.add_argument("--foreground", action="store_true",
                             help="stay running: restart crashed services, stop everything on SIGTERM/Ctrl-C "
                                  "(used by the Slurm batch job)")
        if command == "doctor":
            cmd.add_argument("--smoke", action="store_true", help="also send one request per model")
    sub.add_parser("status")
    sub.add_parser("stop")
    logs = sub.add_parser("logs")
    logs.add_argument("--service", required=True)
    events = sub.add_parser("events", help="what the supervisor saw: starts, exits, restarts, stops")
    events.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    try:
        if args.command == "profile":
            profile = load_profile(args.profile)
            print(f"Profile valid: {profile['name']}")
        elif args.command == "start":
            started = time.monotonic()

            def say(text: str) -> None:
                print(f"[{clock(time.monotonic() - started)}] {text}", flush=True)

            say("checking the profile and vLLM options")
            profile = load_profile(args.profile, check_vllm=True)
            session = start(args.profile, profile, dict(os.environ), say=say)
            if not args.foreground:
                print(f"Started session {session.path}")
                return
            from .watchdog import Watchdog

            watchdog = Watchdog(session, say, f"http://127.0.0.1:{profile['status']['port']}/v1/status")
            watchdog.install_signals()
            sys.exit(watchdog.run())
        elif args.command == "doctor":
            from .doctor import checks, report

            code = report(checks(args.profile))
            if args.smoke:
                code = max(code, _smoke(args.profile))
            sys.exit(code)
        elif args.command == "events":
            session = _latest_session()
            if not session or not (session / "events.jsonl").exists():
                raise RuntimeError("no events recorded yet")
            lines = (session / "events.jsonl").read_text(encoding="utf-8").splitlines()[-args.count:]
            print(f"session {session.name}")
            for line in lines:
                row = json.loads(line)
                print(f"{row['time']}  {row['service']:<8} {row['event']}"
                      + (f" ({row['detail']})" if row.get("detail") else ""))
        elif args.command == "verify":
            sys.exit(_verify(args.profile))
        elif args.command == "smoke":
            sys.exit(_smoke(args.profile))
        elif args.command == "stop":
            print("Stopped: " + (", ".join(stop()) or "no active owned processes"))
        elif args.command == "logs":
            session = _active_session()
            if not session:
                raise RuntimeError("no active session")
            state_path = session / "processes.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"services": {}}
            if args.service not in state.get("services", {}):
                raise RuntimeError(f"unknown service {args.service!r}; check the active session's services")
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
            instances = model_instances(profile)
            alias_counts = {item["alias"]: sum(other["alias"] == item["alias"] for other in instances)
                            for item in instances}
            service_by_status_key = {
                (item["alias"] if alias_counts[item["alias"]] == 1
                 else f"{item['alias']}@{item['port']}"): item["name"] for item in instances
            }
            print("ALIAS             PID    STATE        RUNNING WAITING KV CACHE BACKEND")
            for alias, item in report["models"].items():
                entry = services.get(service_by_status_key[alias], {})
                pid = entry.get("pid", "-")
                if isinstance(pid, int):
                    from .process import process_matches
                    if not process_matches(pid, state.get("sessionId", "")):
                        pid = "stale"
                print(f"{alias:<12} {pid!s:<6} {item['state']:<12} {item.get('running', '-'):>7} "
                      f"{item.get('waiting', '-'):>7} "
                      f"{item.get('kvCache', '-'):>8} {item['backend']}")
    except KeyboardInterrupt:
        print("interrupted; processes of a session that did not finish starting were stopped", file=sys.stderr)
        sys.exit(130)
    except (ProfileError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
