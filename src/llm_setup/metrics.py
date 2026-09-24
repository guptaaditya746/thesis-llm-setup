"""Safe Prometheus text parser for the small vLLM metric subset we consume."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_metrics(text: str) -> dict[str, Any]:
    values: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for line in text.splitlines():
        match = _LINE.match(line.strip())
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = dict(_LABEL.findall(raw_labels or ""))
        values[name].append((labels, value))
    result: dict[str, Any] = {}
    for key, names in {
        "running": ("vllm:num_requests_running", "vllm:num_requests_running_total"),
        "waiting": ("vllm:num_requests_waiting",),
        "kvCache": ("vllm:kv_cache_usage_perc",),
    }.items():
        item = next((values[name] for name in names if name in values), [])
        if item:
            number = sum(v for _, v in item)
            if key == "kvCache":
                number = max(0.0, min(1.0, number / (100 if number > 1 else 1)))
            result[key] = int(number) if key in ("running", "waiting") else number
    result["queueP95Seconds"] = _histogram_quantile(values, "vllm:request_queue_time_seconds", 0.95)
    result["latencyP95Seconds"] = _histogram_quantile(values, "vllm:e2e_request_latency_seconds", 0.95)
    successes = values.get("vllm:request_success", [])
    if successes:
        result["requestSuccess"] = sum(v for _, v in successes)
    return result


def _histogram_quantile(values: dict[str, Any], name: str, quantile: float) -> float | None:
    buckets = sorted((float(labels["le"]), count) for labels, count in values.get(name + "_bucket", [])
                     if "le" in labels and labels["le"] != "+Inf")
    if not buckets:
        return None
    total = max((count for labels, count in values.get(name + "_bucket", [])
                 if labels.get("le") == "+Inf"), default=buckets[-1][1])
    target = total * quantile
    for bound, count in buckets:
        if count >= target:
            return bound if math.isfinite(bound) else None
    return buckets[-1][0]
