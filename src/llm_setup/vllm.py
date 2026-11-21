"""vLLM: the launch command and environment of one model instance, and checks of the installed CLI."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

from .process import GATEWAY_SECRETS, child_environment
from .profile import Instance, Profile, ProfileError

REQUIRED_FLAGS = {"--host", "--port", "--gpu-memory-utilization", "--max-model-len", "--max-num-seqs",
                  "--served-model-name", "--runner", "--tensor-parallel-size"}
HELP_TIMEOUT_S = 600        # `vllm serve --help=all` imports torch; slow on a shared filesystem


def command(instance: Instance) -> list[str]:
    model = instance.model
    args = ["vllm", "serve", model.model, "--host", "127.0.0.1", "--port", str(instance.port),
            "--served-model-name", model.alias, "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(model.gpu_memory_utilization),
            "--max-model-len", str(model.max_model_len), "--max-num-seqs", str(model.max_num_active_seqs)]
    if model.embed:
        args += ["--runner", "pooling"]
    else:
        if model.kv_cache_dtype != "auto":
            args += ["--kv-cache-dtype", model.kv_cache_dtype]
        # Tool calling (tool_choice="auto") needs the model's parser, or vLLM answers HTTP 400.
        if model.tool_call_parser:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", model.tool_call_parser]
    if model.revision:
        args += ["--revision", model.revision]
    return args + list(model.extra_args)


def environment(base: dict[str, str], instance: Instance, session_id: str, number: int = 0) -> dict[str, str]:
    """A model process's environment: no gateway keys, the cluster's Hugging Face settings, one GPU."""
    env = child_environment(base, session_id, gpu=instance.gpu, drop=GATEWAY_SECRETS)
    # FlashInfer's sampler JIT-compiles with nvcc on the first request; Slurm images rarely have nvcc.
    # The native PyTorch sampler needs no compiler. Set VLLM_USE_FLASHINFER_SAMPLER=1 to opt back in.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if base.get("VLLM_PORT"):
        # Parallel instances need separate torch rendezvous ports, even within one Slurm job.
        env["VLLM_PORT"] = str(int(base["VLLM_PORT"]) + number * 4)
    return env


def check_installed(profile: Profile) -> None:
    """Fail before launch when the installed vLLM lacks a flag, parser or runner the profile needs."""
    executable = shutil.which("vllm")
    if not executable:
        raise ProfileError("vLLM is not installed or vllm is not on PATH")
    try:
        timeout = float(os.environ.get("LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS", HELP_TIMEOUT_S))
    except ValueError as exc:
        raise ProfileError("LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS must be a number") from exc
    if not 1 <= timeout <= 3600:
        raise ProfileError("LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS must be between 1 and 3600")
    try:
        result = subprocess.run([executable, "serve", "--help=all"], capture_output=True, text=True,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProfileError(f"timed out after {timeout:g}s running vllm serve --help=all; increase "
                           "LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS if shared-filesystem imports are slow; "
                           "no services were started") from exc
    help_text = result.stdout + result.stderr
    if result.returncode:
        raise ProfileError(f"could not inspect vllm serve --help: {help_text[-1000:]}")
    models = profile.models.values()
    parsers = {model.tool_call_parser for model in models if model.tool_call_parser}
    required = set(REQUIRED_FLAGS)
    required |= {arg.split("=", 1)[0] for model in models for arg in model.extra_args if arg.startswith("--")}
    if any(model.kv_cache_dtype != "auto" for model in models):
        required.add("--kv-cache-dtype")
    if parsers:
        required |= {"--enable-auto-tool-choice", "--tool-call-parser"}
    missing = sorted(flag for flag in required if flag not in help_text)
    if missing:
        raise ProfileError("installed vLLM does not support required options: " + ", ".join(missing))
    listed = re.search(r"--tool-call-parser\s+\{([^}]+)}", help_text)
    unknown = parsers - {name.strip() for name in listed.group(1).split(",")} if listed else set()
    if unknown:
        raise ProfileError("installed vLLM has no tool-call parser named: " + ", ".join(sorted(unknown)))
    runner = re.search(r"--runner\s+\{([^}]+)}", help_text)
    if any(model.embed for model in models) and (not runner or "pooling" not in runner.group(1).split(",")):
        raise ProfileError("installed vLLM does not support the pooling runner required for embeddings")


def _compute_capabilities() -> list[float]:
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


def check_fp8_kv_cache(profile: Profile) -> None:
    """Refuse an FP8 KV cache that would make vLLM compile FlashInfer kernels at startup.

    Before Hopper (compute capability 9.0), vLLM serves an FP8 KV cache with the FlashInfer attention
    backend, JIT-compiled with nvcc unless ``flashinfer_jit_cache`` is installed. Slurm images usually
    have neither, and the model then dies before /health answers.
    """
    if all(model.kv_cache_dtype == "auto" for model in profile.models.values()):
        return
    capabilities = _compute_capabilities()
    if not capabilities or min(capabilities) >= 9.0:
        return
    if importlib.util.find_spec("flashinfer_jit_cache") is not None or shutil.which("nvcc"):
        return
    raise ProfileError(
        f"kv_cache_dtype fp8 on GPUs with compute capability {min(capabilities):g} needs the FlashInfer "
        "attention backend, which JIT-compiles with nvcc; neither nvcc nor flashinfer-jit-cache is available. "
        "Set kv_cache_dtype: auto, or install flashinfer-jit-cache. No services were started.")


def failure_hint(log: Path) -> str:
    """A known cause of a failed start, read from the end of the log; empty when none is recognised."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")[-16000:]
    except OSError:
        return ""
    if "Could not find nvcc" not in text:
        return ""
    if any(marker in text for marker in ("FLASHINFER backend", "FlashInfer attention", "kv_cache_dtype=fp8",
                                         "kv_cache_dtype='fp8'", "--kv-cache-dtype fp8")):
        return ("; FlashInfer attention kernels could not be compiled because nvcc is unavailable. This happens "
                "with kv_cache_dtype fp8 on pre-Hopper GPUs: set kv_cache_dtype: auto in the profile, or install "
                "flashinfer-jit-cache")
    return ("; FlashInfer sampler JIT failed because nvcc is unavailable. Native PyTorch sampling is the default; "
            "set VLLM_USE_FLASHINFER_SAMPLER=1 only when a compatible CUDA toolkit is installed")
