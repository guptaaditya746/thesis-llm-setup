"""Copy a running session's node-local folder to ``runtime/`` on the shared filesystem.

    python -m llm_setup.mirror SOURCE TARGET [--every 60]

SOURCE is ``$LLM_SETUP_LOCAL_ROOT/runtime/<session>``, TARGET ``runtime/<session>``. This is the only
process of a session that writes to the shared filesystem, and it runs on its own, so a hung
filesystem (Ceph) stops the copying but not the GPU services or the supervisor that must stop them.

Every ``--every`` seconds it appends what each log gained and replaces every other file that changed
(``status.sqlite3`` through SQLite's backup, so the copy is consistent; SQLite's side files are
skipped). A failed copy is logged and retried on the next pass. Once the session is ready it names it
in ``runtime/current`` (again whenever that file goes missing). It passes on a stop asked for on
another machine: ``llm-setup stop`` there writes ``TARGET/stop-requested``, and the mirror stops the
session here. It exits after a last copy on SIGTERM (the supervisor's stop), or once the session is
over (no longer current, or its supervisor gone before it was ready) and none of its processes lives.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sqlite3
import sys
import threading
import traceback
from pathlib import Path

APPEND = {".log", ".jsonl"}
SKIP_SUFFIXES = ("-journal", "-wal", "-shm", ".part")
STOP_REQUEST = "stop-requested"
CHECK_S = 10.0


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _copy_whole(item: Path, dest: Path) -> None:
    partial = dest.with_name(f".{dest.name}.part")
    if item.suffix == ".sqlite3":
        partial.unlink(missing_ok=True)
        source, copy = sqlite3.connect(item), sqlite3.connect(partial)
        try:
            source.backup(copy)
        finally:
            source.close()
            copy.close()
    else:
        shutil.copyfile(item, partial)
    os.replace(partial, dest)


def _append_new(item: Path, dest: Path, size: int, copied: int | None) -> None:
    """Append the bytes past ``copied``, the length the last copy left. When the copy is not that long
    (written to by someone else, or lost) or the source shrank, copy it whole instead."""
    if copied is None or size < copied or not dest.exists() or dest.stat().st_size != copied:
        _copy_whole(item, dest)
        return
    with item.open("rb") as source, dest.open("ab") as copy:
        source.seek(copied)
        copy.write(source.read(size - copied))


def sync(source: Path, target: Path, seen: dict[str, tuple[int, int]]) -> None:
    """Bring ``target`` up to date with ``source``. ``seen``: each file's (size, mtime) at its last
    copy. A file that fails is logged and left for the next pass."""
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.name.startswith(".") or item.name.endswith(SKIP_SUFFIXES):
            continue
        try:
            if not item.is_file():
                continue
            stat = item.stat()
            mark = (stat.st_size, stat.st_mtime_ns)
            dest = target / item.name
            if seen.get(item.name) == mark and dest.exists():
                continue
            if item.suffix in APPEND:
                before = seen.get(item.name)
                _append_new(item, dest, stat.st_size, before[0] if before else None)
            else:
                _copy_whole(item, dest)
            seen[item.name] = mark
        except FileNotFoundError:
            continue                       # gone between listing and copying
        except (OSError, sqlite3.Error) as exc:
            seen.pop(item.name, None)      # copy it whole next time
            _say(f"copying {item.name} failed: {exc}")


class Mirror:
    def __init__(self, source: Path, target: Path, every: float = 60.0, parent: int | None = None) -> None:
        self.source, self.target, self.every = source, target, every
        self.session_id = source.name
        self.local_current = source.parent / "current"
        self.shared_current = target.parent / "current"
        self.parent = parent if parent is not None else os.getppid()
        self.seen: dict[str, tuple[int, int]] = {}
        self.was_current = False
        self.named = False
        self.idle_checks = 0
        self.stop = threading.Event()

    def is_current(self) -> bool:
        try:
            return self.local_current.read_text(encoding="utf-8").strip() == self.session_id
        except FileNotFoundError:
            return False

    def name_session(self) -> None:
        """Name this session in ``runtime/current`` when it becomes ready, and again if the file is
        removed; a newer session that names itself there is left alone."""
        if not self.named or not self.shared_current.exists():
            self.shared_current.write_text(self.session_id, encoding="utf-8")
            self.named = True

    def once(self) -> None:
        sync(self.source, self.target, self.seen)

    def pass_on_stop_request(self) -> None:
        request = self.target / STOP_REQUEST
        if not request.exists():
            return
        from .runtime import Runtime
        from .supervisor import stop

        _say("stop requested on another machine; stopping the session")
        stop(Runtime(self.target.parent, self.source.parent), on_node=True)
        request.unlink(missing_ok=True)

    def anything_alive(self) -> bool:
        from .runtime import alive_services

        return bool(alive_services(self.source, self.session_id))

    def over(self, current: bool) -> bool:
        """The session ended (stopped, or its supervisor died during startup) and nothing of it runs,
        on two checks in a row, so the last log lines and events are written before the last copy."""
        ended = (self.was_current and not current) or (not self.was_current and os.getppid() != self.parent)
        self.idle_checks = self.idle_checks + 1 if ended and not self.anything_alive() else 0
        return self.idle_checks >= 2

    def finish(self) -> None:
        sync(self.source, self.target, self.seen)
        try:
            if self.shared_current.read_text(encoding="utf-8").strip() == self.session_id:
                self.shared_current.unlink()
        except FileNotFoundError:
            pass

    def run(self) -> None:
        waited = self.every                    # copy at once on start
        while not self.stop.is_set():
            try:
                current = self.is_current()
                if self.over(current):
                    break
                if current and not self.was_current:
                    self.was_current = True
                    waited = self.every        # name the session and copy now, not at the next tick
                if waited >= self.every:
                    waited = 0.0
                    self.once()
                if current:
                    self.name_session()
                    self.pass_on_stop_request()
            except Exception:                  # noqa: BLE001 - keep mirroring; the trace goes to mirror.log
                _say(traceback.format_exc())
            self.stop.wait(CHECK_S)
            waited += CHECK_S
        try:
            self.finish()
        except Exception:                      # noqa: BLE001
            _say(traceback.format_exc())


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm_setup.mirror")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--every", type=float, default=60.0)
    args = parser.parse_args()
    mirror = Mirror(args.source, args.target, args.every)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda *_: mirror.stop.set())
    mirror.run()


if __name__ == "__main__":
    main()
