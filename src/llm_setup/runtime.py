"""The ``runtime/`` folder: one directory per session and a ``current`` file naming the running one.

With ``LLM_SETUP_LOCAL_ROOT`` set (the Slurm job sets it), a session writes this layout under
``$LLM_SETUP_LOCAL_ROOT/runtime`` on node-local disk, and ``llm_setup.mirror`` copies it to
``runtime/`` on the shared filesystem every minute. The services and the supervisor then never write to
the shared filesystem: when it hangs (Ceph), only the mirror waits, and the GPU services stay
stoppable. Commands on another machine read the mirror, up to a minute old.

    runtime/
      current                         the running session's id (absent when none runs)
      slurm-<jobid>.out               the batch job's output
      <session-id>/
        profile.yaml                  the resolved profile the session runs
        litellm-config.yaml           the generated gateway configuration
        metadata.json                 host, Slurm job, profile source, node-local folder
        processes.json                pid, command and log of each service
        events.jsonl                  starts, exits, restarts, stops
        status.sqlite3                one day of status samples
        <service>.log                 each service's output
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .process import owned

SESSION_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")


class Runtime:
    """``root``: ``runtime/`` on the shared filesystem. ``work``: where a session on this machine writes
    (node-local scratch; ``root`` itself when no local root is set)."""

    def __init__(self, root: str | Path = "runtime", work: str | Path | None = None) -> None:
        self.root = Path(root)
        self.work = Path(work) if work is not None else self.root

    @classmethod
    def default(cls) -> Runtime:
        """``runtime/``, working under ``$LLM_SETUP_LOCAL_ROOT/runtime`` when that is set."""
        local = os.environ.get("LLM_SETUP_LOCAL_ROOT", "").strip()
        return cls("runtime", Path(local) / "runtime" if local else None)

    @property
    def mirrored(self) -> bool:
        return self.work.resolve() != self.root.resolve()

    @property
    def current_file(self) -> Path:
        return self.work / "current"

    def _read_current(self, file: Path) -> str | None:
        try:
            session_id = file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not SESSION_ID.fullmatch(session_id):
            raise RuntimeError(f"{file} holds an invalid session id; check {file.parent}/ and remove it")
        return session_id

    def current_id(self, shared: bool = True) -> str | None:
        """The running session: this machine's, else (``shared``) the one the mirror names in ``root``."""
        session_id = self._read_current(self.current_file)
        if session_id is None and shared and self.mirrored:
            session_id = self._read_current(self.root / "current")
        return session_id

    def session_path(self, session_id: str) -> Path:
        """The session folder: the live node-local one when this machine has it, else the mirror."""
        local = self.work / session_id
        return local if local.is_dir() else self.root / session_id

    def is_local(self, session_id: str) -> bool:
        return (self.work / session_id).is_dir()

    def current(self) -> Path | None:
        session_id = self.current_id()
        return self.session_path(session_id) if session_id else None

    def latest(self) -> Path | None:
        """The running session, or else the newest one, so a stopped or failed session stays readable."""
        current = self.current()
        if current:
            return current
        names = {path.name for base in {self.work, self.root} if base.is_dir() for path in base.glob("*")
                 if path.is_dir() and SESSION_ID.fullmatch(path.name)}
        return self.session_path(max(names)) if names else None

    def new_session(self) -> tuple[str, Path]:
        session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        path = self.work / session_id
        path.mkdir(parents=True)
        return session_id, path

    def mark_current(self, session_id: str) -> None:
        self.current_file.write_text(session_id, encoding="utf-8")

    def clear_current(self, only: str | None = None, shared: bool = False) -> None:
        """Forget the running session; with ``only``, only when it is that session. ``shared`` also clears
        the mirror's ``root/current`` (a running session leaves that to its mirror)."""
        files = [self.current_file] + ([self.root / "current"] if shared and self.mirrored else [])
        for file in files:
            try:
                if only is None or file.read_text(encoding="utf-8").strip() == only:
                    file.unlink()
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


def read_metadata(session: Path) -> dict[str, Any]:
    """metadata.json: host, Slurm job, profile and, for a mirrored session, its node-local folder."""
    try:
        data = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


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
