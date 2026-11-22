"""Tests run as on a machine without node-local scratch, whatever the developer's shell sourced."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_local_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_SETUP_LOCAL_ROOT", raising=False)
