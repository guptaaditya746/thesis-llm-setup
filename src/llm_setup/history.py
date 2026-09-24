"""Compact SQLite sample retention."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class History:
    def __init__(self, path: str | Path, retention_seconds: int = 86400):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_seconds
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS samples (ts REAL NOT NULL, alias TEXT NOT NULL, data TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, alias: str, data: dict[str, Any], now: float | None = None) -> None:
        timestamp = now if now is not None else time.time()
        with self._connect() as db:
            db.execute("INSERT INTO samples VALUES (?, ?, ?)", (timestamp, alias, json.dumps(data)))
            db.execute("DELETE FROM samples WHERE ts < ?", (timestamp - self.retention_seconds,))

    def recent(self, alias: str, seconds: int = 86400, now: float | None = None) -> list[dict[str, Any]]:
        cutoff_now = now if now is not None else time.time()
        with self._connect() as db:
            rows = db.execute("SELECT data FROM samples WHERE alias=? AND ts>=? ORDER BY ts",
                              (alias, cutoff_now - seconds)).fetchall()
        return [json.loads(row[0]) for row in rows]
