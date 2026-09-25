"""Profile loading, interpolation, and validation."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

ALIASES = {"heavy": "heavy-model", "lite": "lite-model", "embed": "qwen-embed"}
_ENV = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_DEFAULT_VLLM_HELP_TIMEOUT_SECONDS = 600
# KV-cache storage types accepted by vLLM's --kv-cache-dtype. "auto" keeps the
# model's own dtype; the FP8 variants roughly double the tokens that fit.
KV_CACHE_DTYPES = {"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}
# vLLM tool-call parser names look like "hermes", "granite4", "granite-20b-fc".
_PARSER_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class ProfileError(ValueError):
    """Invalid or unsupported serving profile."""


def load_profile(path: str | Path, *, check_vllm: bool = False) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")

    def substitute(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    data = yaml.safe_load(_ENV.sub(substitute, raw))
    if os.environ.get("EMBED_MODEL_ID"):
        data["models"]["embed"]["model"] = os.environ["EMBED_MODEL_ID"]
    validate_profile(data)
    if check_vllm:
        validate_vllm_options(data)
        check_fp8_kv_cache(data)
    return data


def validate_profile(data: Any) -> None:
    if not isinstance(data, dict) or set(data.get("models") or {}) != set(ALIASES):
        raise ProfileError("profile.models must define exactly heavy, lite, and embed")
    allowed = {"name", "bind_host", "gateway", "status", "models", "storage"}
    unknown = set(data) - allowed
    if unknown:
        raise ProfileError(f"unsupported profile fields: {', '.join(sorted(unknown))}")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ProfileError("profile.name is required")
    if not isinstance(data.get("storage"), dict) or not isinstance(data["storage"].get("database"), str):
        raise ProfileError("storage.database path is required")
    if set(data["storage"]) != {"database"}:
        raise ProfileError("unsupported storage fields")
    if data.get("bind_host") != "127.0.0.1":
        raise ProfileError("bind_host must be 127.0.0.1 so vLLM stays loopback-only")
    used_ports: set[int] = set()
    used_gpus: set[int] = set()
    for role, expected_alias in ALIASES.items():
        model = data["models"][role]
        if not isinstance(model, dict):
            raise ProfileError(f"models.{role} must be a mapping")
        supported_fields = {"alias", "model", "gpu", "port", "max_num_active_seqs",
                            "max_num_queued_reqs", "max_model_len", "gpu_memory_utilization"}
        if role == "embed":
            supported_fields.add("task")
        else:
            supported_fields.update({"kv_cache_dtype", "tool_call_parser"})
        unknown = set(model) - supported_fields
        if unknown:
            raise ProfileError(f"unsupported fields for models.{role}: {', '.join(sorted(unknown))}")
        if model.get("alias") != expected_alias:
            raise ProfileError(f"models.{role}.alias must be {expected_alias}")
        if not model.get("model"):
            raise ProfileError(f"models.{role}.model is required; set EMBED_MODEL_ID if needed")
        for key in ("gpu", "port", "max_num_active_seqs", "max_num_queued_reqs", "max_model_len"):
            minimum = 0 if key == "gpu" else 1
            if type(model.get(key)) is not int or model[key] < minimum:
                raise ProfileError(f"models.{role}.{key} must be a positive integer")
        if model["gpu"] in used_gpus:
            raise ProfileError(f"GPU {model['gpu']} is assigned to more than one role")
        if model["port"] in used_ports:
            raise ProfileError(f"port {model['port']} is assigned to more than one role")
        try:
            memory_fraction = float(model.get("gpu_memory_utilization", 0))
        except (TypeError, ValueError) as exc:
            raise ProfileError(f"models.{role}.gpu_memory_utilization must be numeric") from exc
        if not 0 < memory_fraction <= 1:
            raise ProfileError(f"models.{role}.gpu_memory_utilization must be in (0, 1]")
        if model.get("kv_cache_dtype", "auto") not in KV_CACHE_DTYPES:
            raise ProfileError(
                f"models.{role}.kv_cache_dtype must be one of: {', '.join(sorted(KV_CACHE_DTYPES))}"
            )
        parser = model.get("tool_call_parser")
        if parser is not None and (not isinstance(parser, str) or not _PARSER_NAME.fullmatch(parser)):
            raise ProfileError(f"models.{role}.tool_call_parser must be a vLLM parser name such as hermes")
        used_gpus.add(model["gpu"])
        used_ports.add(model["port"])
    if not isinstance(data.get("gateway"), dict) or not isinstance(data.get("status"), dict):
        raise ProfileError("gateway and status endpoint configuration is required")
    for name in ("gateway", "status"):
        if set(data[name]) != {"host", "port"}:
            raise ProfileError(f"{name} must define only host and port")
    endpoints = [data["gateway"], data["status"]]
    endpoints += [{"host": "127.0.0.1", "port": p} for p in used_ports]
    if any(e.get("host") != "127.0.0.1" for e in endpoints):
        raise ProfileError("all service hosts must be 127.0.0.1")
    ports = [e.get("port") for e in endpoints]
    if any(not isinstance(port, int) or not 1 <= port <= 65535 for port in ports):
        raise ProfileError("service ports must be between 1 and 65535")
    if len(set(ports)) != len(ports):
        raise ProfileError("service ports must be unique")
    if data["models"]["embed"].get("task") != "embed":
        raise ProfileError("models.embed.task must be embed")


def uses_kv_cache_dtype(profile: dict[str, Any] | None) -> bool:
    """True when any role asks for a non-default KV-cache dtype."""
    if not profile:
        return False
    return any(item.get("kv_cache_dtype", "auto") != "auto" for item in profile["models"].values())


def _gpu_compute_capabilities() -> list[float]:
    """Compute capability of each visible GPU from nvidia-smi; [] when unavailable."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, "--query-gpu=compute_cap", "--format=csv,noheader"],
                                capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    values = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return values


def check_fp8_kv_cache(profile: dict[str, Any] | None) -> None:
    """Refuse an FP8 KV cache that would make vLLM compile FlashInfer kernels at startup.

    Before Hopper (compute capability 9.0), vLLM serves an FP8 KV cache with the
    FlashInfer attention backend. Its kernels are JIT-compiled with nvcc unless
    the ``flashinfer_jit_cache`` package is installed; Slurm images usually have
    neither, and the backend then dies before /health answers.
    """
    if not uses_kv_cache_dtype(profile):
        return
    capabilities = _gpu_compute_capabilities()
    if not capabilities or min(capabilities) >= 9.0:
        return
    import importlib.util

    if importlib.util.find_spec("flashinfer_jit_cache") is not None or shutil.which("nvcc"):
        return
    raise ProfileError(
        f"kv_cache_dtype fp8 on GPUs with compute capability {min(capabilities):g} needs the FlashInfer "
        "attention backend, which JIT-compiles with nvcc; neither nvcc nor flashinfer-jit-cache is "
        "available. Set kv_cache_dtype: auto, or install flashinfer-jit-cache matching the installed "
        "flashinfer version. No services were started."
    )


def tool_call_parsers(profile: dict[str, Any] | None) -> set[str]:
    """Tool-call parsers the profile asks for (roles that serve tool calling)."""
    if not profile:
        return set()
    return {item["tool_call_parser"] for item in profile["models"].values() if item.get("tool_call_parser")}


def validate_vllm_options(profile: dict[str, Any] | None = None) -> set[str]:
    """Read installed CLI help and fail early if required launch flags are absent."""
    executable = shutil.which("vllm")
    if not executable:
        raise ProfileError("vLLM is not installed or vllm is not on PATH")
    timeout_value = os.environ.get(
        "LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS", str(_DEFAULT_VLLM_HELP_TIMEOUT_SECONDS)
    )
    try:
        timeout_seconds = float(timeout_value)
    except ValueError as exc:
        raise ProfileError("LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS must be a number") from exc
    if not 1 <= timeout_seconds <= 3600:
        raise ProfileError("LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS must be between 1 and 3600")
    try:
        result = subprocess.run([executable, "serve", "--help=all"], capture_output=True,
                                text=True, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ProfileError(
            f"timed out after {timeout_seconds:g}s running vllm serve --help=all; "
            "increase LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS if shared-filesystem imports are slow or blocked; "
            "no services were started"
        ) from exc
    help_text = result.stdout + result.stderr
    if result.returncode:
        raise ProfileError(f"could not inspect vllm serve --help: {help_text[-1000:]}")
    required = {"--host", "--port", "--gpu-memory-utilization", "--max-model-len", "--max-num-seqs",
                "--served-model-name", "--runner", "--tensor-parallel-size"}
    if uses_kv_cache_dtype(profile):
        required.add("--kv-cache-dtype")
    parsers = tool_call_parsers(profile)
    if parsers:
        required.update({"--enable-auto-tool-choice", "--tool-call-parser"})
    available = {flag for flag in required if flag in help_text}
    missing = sorted(required - available)
    if missing:
        raise ProfileError("installed vLLM does not support required options: " + ", ".join(missing))
    listed = re.search(r"--tool-call-parser\s+\{([^}]+)}", help_text)
    if parsers and listed:
        unknown = parsers - {name.strip() for name in listed.group(1).split(",")}
        if unknown:
            raise ProfileError("installed vLLM has no tool-call parser named: " + ", ".join(sorted(unknown)))
    runner = re.search(r"--runner\s+\{([^}]+)}", help_text)
    if not runner or "pooling" not in runner.group(1).split(","):
        raise ProfileError("installed vLLM does not support the pooling runner required for embeddings")
    return available
