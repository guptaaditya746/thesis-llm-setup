"""What a starting service is doing, read from the end of its log."""

from __future__ import annotations

import re
from pathlib import Path

# Checked against the newest log line first; the first pattern that matches a line names the phase.
# vLLM and LiteLLM wording; a line that matches nothing is shown as it is.
PHASES: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(pattern, re.IGNORECASE), name) for pattern, name in (
    (r"Application startup complete|Uvicorn running on|Starting vLLM API server", "starting the API"),
    (r"Graph capturing finished|CUDA graphs? captured", "CUDA graphs done"),
    (r"Capturing CUDA graph|Capturing cudagraph", "capturing CUDA graphs"),
    (r"torch\.compile|Compiling a graph|Dynamo bytecode", "compiling"),
    (r"KV cache|Memory profiling|determine_available_memory|# GPU blocks", "sizing the KV cache"),
    (r"Loading weights took|Model loading took", "weights loaded"),
    (r"Loading safetensors checkpoint shards:\s*(\d+)%", "loading weights {0}%"),
    (r"Loading safetensors|Loading weights|load_model", "loading weights"),
    (r"Downloading|Fetching \d+ files", "downloading weights"),
    (r"Initializing a V\d+ LLM engine|Initializing an LLM engine|non-default args", "initialising the engine"),
    (r"Traceback|Error:|CUDA out of memory|OutOfMemoryError", "error (see the log)"),
))


def log_lines(path: Path, limit_bytes: int = 64_000) -> list[str]:
    """The last lines of a log; progress bars rewrite one line with carriage returns."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]


def current_phase(path: Path) -> str:
    """A short name for what the service is doing now, or its newest log line."""
    lines = log_lines(path)
    for line in reversed(lines):
        for pattern, name in PHASES:
            match = pattern.search(line)
            if match:
                return name.format(*match.groups())
    return lines[-1][:100] if lines else "no output yet"


def tail(path: Path, count: int = 20) -> str:
    return "\n".join(log_lines(path)[-count:])
