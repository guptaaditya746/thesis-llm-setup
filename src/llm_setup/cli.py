"""``llm-setup``: start, watch and stop a serving session.

    llm-setup start [--profile P] [--gpus N] [--foreground]
    llm-setup stop | status | events | logs SERVICE
    llm-setup doctor [--profile P] [--gpus N] [--smoke]
    llm-setup smoke [--profile P]
    llm-setup profile show|extras [--profile P] [--gpus N]

The profile defaults to $LLM_SETUP_PROFILE, else profiles/orch-gliner.yaml. Commands that read a
running session (status, smoke, logs, events) use the profile that session runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

from .profile import ProfileError, load_profile, parse_profile, read_profile, visible_gpu_count
from .runtime import Runtime, load_processes, read_events
from .supervisor import clock, start, stop

DEFAULT_PROFILE = "profiles/orch-gliner.yaml"


def _profile_arguments(command: argparse.ArgumentParser, gpus: bool = True) -> None:
    command.add_argument("--profile", default=os.environ.get("LLM_SETUP_PROFILE", DEFAULT_PROFILE),
                         help="profile file (default: $LLM_SETUP_PROFILE, else %(default)s)")
    if gpus:
        command.add_argument("--gpus", type=int, help="GPUs for `gpus: all` (default: the allocation's)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-setup", description="Serve LLMs on a Slurm GPU allocation.")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("start", help="start a session (in the foreground: supervise it until stopped)")
    _profile_arguments(command)
    command.add_argument("--foreground", action="store_true",
                         help="stay running: restart crashed services, stop everything on SIGTERM/Ctrl-C")
    sub.add_parser("stop", help="stop the running session")
    sub.add_parser("status", help="each model's state, load and KV cache")
    command = sub.add_parser("events", help="what the supervisor saw: starts, exits, restarts, stops")
    command.add_argument("--count", type=int, default=30)
    command = sub.add_parser("logs", help="the end of one service's log (running or last session)")
    command.add_argument("service", help="a model instance (e.g. orch, orch-2), gliner, gateway or status")
    command.add_argument("--lines", type=int, default=200)
    command = sub.add_parser("doctor", help="every check in order; names the first problem")
    _profile_arguments(command)
    command.add_argument("--smoke", action="store_true", help="also send one request per model")
    command = sub.add_parser("smoke", help="one request per model through the gateway")
    command.add_argument("--profile", help="default: the running session's profile")
    command = sub.add_parser("profile", help="show a resolved profile, or the extras it needs")
    command.add_argument("action", choices=("show", "extras"))
    _profile_arguments(command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        sys.exit(COMMANDS[args.command](args) or 0)
    except KeyboardInterrupt:
        print("interrupted; processes of a session that did not finish starting were stopped", file=sys.stderr)
        sys.exit(130)
    except (ProfileError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _start(args: argparse.Namespace) -> int:
    from . import vllm

    began = time.monotonic()

    def say(text: str) -> None:
        print(f"[{clock(time.monotonic() - began)}] {text}", flush=True)

    say(f"checking {args.profile} and the installed vLLM")
    visible = visible_gpu_count()
    if args.gpus is not None and visible is not None and args.gpus > visible:
        raise RuntimeError(f"--gpus {args.gpus}, but this allocation has {visible} GPU(s)")
    profile = load_profile(args.profile, args.gpus if args.gpus is not None else visible)
    vllm.check_installed(profile)
    vllm.check_fp8_kv_cache(profile)
    session = start(profile, dict(os.environ), say=say)
    if not args.foreground:
        print(f"started session {session.path}; `llm-setup stop` stops it")
        return 0
    from .watchdog import Watchdog

    watchdog = Watchdog(session, say, f"http://127.0.0.1:{profile.status_port}/v1/status")
    watchdog.install_signals()
    return watchdog.run()


def _stop(_args: argparse.Namespace) -> int:
    print("stopped: " + (", ".join(stop()) or "no running session"))
    return 0


def _running_session() -> tuple[str, Path]:
    runtime = Runtime()
    session_id = runtime.current_id()
    if not session_id:
        raise RuntimeError("no running session")
    return session_id, runtime.root / session_id


def _status(_args: argparse.Namespace) -> int:
    from .process import owned
    from .status import History, StatusMonitor

    session_id, path = _running_session()
    profile = load_profile(path / "profile.yaml")
    report = asyncio.run(StatusMonitor(profile, History(path / "status.sqlite3")).poll_once())
    processes = load_processes(path)

    def pid(name: str) -> str:
        value = processes.get(name, {}).get("pid")
        return "-" if value is None else str(value) if owned(int(value), session_id) else "stale"

    keys = profile.status_keys()
    print(f"overall {report['overall']} · session {session_id}")
    print(f"{'SERVICE':<12} {'PID':<7} {'STATE':<12} {'RUNNING':>7} {'WAITING':>7} {'KV':>6}  ENDPOINT")
    for key, item in report["models"].items():
        instance = keys[key]
        kv = f"{item['kvCache']:.0%}" if isinstance(item.get("kvCache"), float) else "-"
        print(f"{instance.name:<12} {pid(instance.name):<7} {item['state']:<12} {item.get('running', '-'):>7} "
              f"{item.get('waiting', '-'):>7} {kv:>6}  {key} {item['backend']}")
    if profile.gliner:
        import httpx

        url = f"http://127.0.0.1:{profile.gliner.port}"
        try:
            state = "ready" if httpx.get(f"{url}/health", timeout=2).is_success else "unhealthy"
        except httpx.HTTPError:
            state = "down"
        print(f"{'gliner':<12} {pid('gliner'):<7} {state:<12} {'-':>7} {'-':>7} {'-':>6}  {url}")
    return 0


def _events(args: argparse.Namespace) -> int:
    session = Runtime().latest()
    rows = read_events(session, args.count)
    if not rows:
        raise RuntimeError("no events recorded yet")
    print(f"session {session.name}")
    for row in rows:
        print(f"{row['time']}  {row['service']:<8} {row['event']}" + (f" ({row['detail']})" if row.get("detail") else ""))
    return 0


def _logs(args: argparse.Namespace) -> int:
    session = Runtime().latest()
    if session is None:
        raise RuntimeError("no session recorded in runtime/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.service):
        raise RuntimeError(f"not a service name: {args.service!r}")
    log = session / f"{args.service}.log"
    if not log.exists():
        names = sorted(path.stem for path in session.glob("*.log"))
        raise RuntimeError(f"no log for {args.service!r} in {session}; services: {', '.join(names) or 'none'}")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-args.lines:]))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from .doctor import checks, report, smoke

    code = report(checks(args.profile, gpu_count=args.gpus))
    if args.smoke:
        code = max(code, int(bool(smoke(load_profile(args.profile, args.gpus), os.getenv("LITELLM_MASTER_KEY", "")))))
    return code


def _smoke(args: argparse.Namespace) -> int:
    from .doctor import smoke

    profile = load_profile(args.profile if args.profile else _running_session()[1] / "profile.yaml")
    return int(bool(smoke(profile, os.getenv("LITELLM_MASTER_KEY", ""))))


def _profile(args: argparse.Namespace) -> int:
    if args.action == "extras":            # no GPU needed: the batch job asks before it starts anything
        data = read_profile(args.profile)
        gliner = data.get("gliner")
        print("gliner" if isinstance(gliner, dict) and gliner.get("enabled", True) is True else "")
        return 0
    profile = load_profile(args.profile, args.gpus)
    print(f"profile {profile.name} ({args.profile})")
    for instance in profile.instances:
        print(f"  {instance.name:<12} {instance.model.alias:<14} GPU {instance.gpu}  :{instance.port}  "
              f"{instance.model.model}")
    if profile.gliner:
        print(f"  {'gliner':<12} {'-':<14} GPU {profile.gliner.gpu}  :{profile.gliner.port}  {profile.gliner.model}")
    print(f"  gateway :{profile.gateway_port} · status :{profile.status_port}")
    parse_profile(profile.data)            # the resolved copy a session stores validates on its own
    return 0


COMMANDS = {"start": _start, "stop": _stop, "status": _status, "events": _events, "logs": _logs,
            "doctor": _doctor, "smoke": _smoke, "profile": _profile}


if __name__ == "__main__":
    main()
