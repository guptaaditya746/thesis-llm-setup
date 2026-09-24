"""Exact PID ownership checks for repository-managed child processes."""

from __future__ import annotations

import os
import signal
from pathlib import Path


def process_matches(pid: int, session_id: str, runtime_root: str | Path = "runtime") -> bool:
    """Return true only when /proc identifies this process as owned by this session."""
    marker = f"LLM_SETUP_SESSION_ID={session_id}".encode()
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return marker in environment


def stop_owned(pid: int, session_id: str, runtime_root: str | Path = "runtime") -> bool:
    if not process_matches(pid, session_id, runtime_root):
        return False
    os.kill(pid, signal.SIGTERM)
    return True
