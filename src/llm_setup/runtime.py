"""The ``runtime/`` folder: one directory per session and a ``current`` file naming the running one.

    runtime/
      current                         the running session's id (absent when none runs)
      slurm-<jobid>.out               the batch job's output
      <session-id>/
        profile.yaml                  the resolved profile the session runs
        litellm-config.yaml           the generated gateway configuration
        metadata.json                 host, Slurm job, profile source
        processes.json                pid, command and log of each service
        events.jsonl                  starts, exits, restarts, stops
        status.sqlite3                one day of status samples
        <service>.log                 each service's output
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .process import owned

SESSION_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")


class Runtime:
    def __init__(self, root: str | Path = "runtime") -> None:
        self.root = Path(root)

    @property
    def current_file(self) -> Path:
        return self.root / "current"

    def current_id(self) -> str | None:
        try:
            session_id = self.current_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not SESSION_ID.fullmatch(session_id):
            raise RuntimeError(f"{self.current_file} holds an invalid session id; check {self.root}/ and remove it")
        return session_id

    def current(self) -> Path | None:
        session_id = self.current_id()
        return self.root / session_id if session_id else None

    def latest(self) -> Path | None:
        """The running session, or else the newest one, so a stopped or failed session stays readable."""
        current = self.current()
        if current:
            return current
        sessions = sorted(path for path in self.root.glob("*") if path.is_dir() and SESSION_ID.fullmatch(path.name))
        return sessions[-1] if sessions else None

    def new_session(self) -> tuple[str, Path]:
        session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        path = self.root / session_id
        path.mkdir(parents=True)
        return session_id, path

    def mark_current(self, session_id: str) -> None:
        self.current_file.write_text(session_id, encoding="utf-8")

    def clear_current(self, only: str | None = None) -> None:
        """Forget the running session; with ``only``, only when it is that session."""
        try:
            if only is None or self.current_file.read_text(encoding="utf-8").strip() == only:
                self.current_file.unlink()
        except FileNotFoundError:
            pass


# Inside a session folder ---------------------------------------------------------------------------

def record_event(session: Path, service: str, event: str, detail: str = "") -> dict[str, str]:
    row = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "service": service, "event": event,
           "detail": detail}
    try:
        with (session / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        pass
    return row


def read_events(session: Path | None, count: int = 50) -> list[dict[str, Any]]:
    if session is None or not (session / "events.jsonl").exists():
        return []
    rows = []
    for line in (session / "events.jsonl").read_text(encoding="utf-8").splitlines()[-count:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def save_processes(session: Path, session_id: str, records: dict[str, Any]) -> None:
    (session / "processes.json").write_text(json.dumps({"sessionId": session_id, "services": records}, indent=2),
                                            encoding="utf-8")


def load_processes(session: Path) -> dict[str, dict[str, Any]]:
    """Each recorded service's pid, command and log; empty when nothing was recorded."""
    try:
        return json.loads((session / "processes.json").read_text(encoding="utf-8")).get("services", {})
    except (OSError, ValueError):
        return {}


def alive_services(session: Path, session_id: str) -> list[str]:
    """Recorded services whose process still runs and carries the session marker."""
    return [name for name, data in load_processes(session).items() if owned(int(data["pid"]), session_id)]
