"""Serving profiles: which models run on which GPUs and ports.

A profile file is read in three steps:

1. ``${NAME}`` and ``${NAME:-default}`` are replaced from the environment;
2. ``extends: other.yaml`` merges a base profile underneath this one (mappings merge, other values
   are replaced);
3. ``gpus: all`` becomes every GPU of the allocation that no other model claims, so one profile
   serves one replica on a 1-GPU job and three on a 3-GPU job.

The result is validated into a :class:`Profile`. Every service listens on 127.0.0.1 only.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HOST = "127.0.0.1"
RESERVED = {"gateway", "status", "gliner"}
KV_CACHE_DTYPES = {"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}
# The optional GLiNER service. The revision pins the default model; another model brings its own.
GLINER_DEFAULTS: dict[str, Any] = {"enabled": True, "gpu": 0, "port": 8090,
                                   "model": "urchade/gliner_medium-v2.1",
                                   "revision": "40ec419335d09393f298636f471328b722c6da9e",
                                   "max_chars": 1500}

_ENV = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_PARSER = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_TOP_FIELDS = {"name", "gateway_port", "status_port", "models", "gliner"}
_MODEL_REQUIRED = {"alias", "model", "gpus", "port", "max_num_active_seqs", "max_num_queued_reqs",
                   "max_model_len", "gpu_memory_utilization"}
_MODEL_OPTIONAL = {"revision", "task", "extra_args"}
_CHAT_ONLY = {"kv_cache_dtype", "tool_call_parser"}
# Set by the launcher; in extra_args they would override it (the last flag wins), e.g. --host 0.0.0.0.
_MANAGED_FLAGS = {"--host", "--port", "--served-model-name", "--tensor-parallel-size"}


class ProfileError(ValueError):
    """Invalid profile, or one this machine cannot serve."""


@dataclass(frozen=True)
class Model:
    """One role: a model served under one alias, one vLLM process per GPU in ``gpus``."""

    role: str
    alias: str
    model: str
    gpus: tuple[int, ...]
    port: int                       # the first replica's port; replica i listens on port + i
    max_num_active_seqs: int
    max_num_queued_reqs: int
    max_model_len: int
    gpu_memory_utilization: float
    revision: str | None = None
    task: str = "generate"          # or "embed"
    kv_cache_dtype: str = "auto"
    tool_call_parser: str | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def embed(self) -> bool:
        return self.task == "embed"

    @property
    def instances(self) -> list[Instance]:
        return [Instance(self, index, gpu, self.port + index) for index, gpu in enumerate(self.gpus)]


@dataclass(frozen=True)
class Instance:
    """One vLLM process: a replica of a model on one GPU and one port."""

    model: Model
    index: int
    gpu: int
    port: int

    @property
    def name(self) -> str:
        return self.model.role if self.index == 0 else f"{self.model.role}-{self.index + 1}"

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"


@dataclass(frozen=True)
class Gliner:
    gpu: int
    port: int
    model: str
    revision: str | None
    max_chars: int


@dataclass(frozen=True)
class Profile:
    name: str
    models: dict[str, Model]
    gateway_port: int = 4000
    status_port: int = 8010
    gliner: Gliner | None = None
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)   # resolved YAML

    @property
    def instances(self) -> list[Instance]:
        return [instance for model in self.models.values() for instance in model.instances]

    def gpus(self) -> list[tuple[str, int]]:
        """(service, GPU index) of every GPU service."""
        placed = [(instance.name, instance.gpu) for instance in self.instances]
        return placed + ([("gliner", self.gliner.gpu)] if self.gliner else [])

    def ports(self) -> dict[str, int]:
        """Every service's port, by service name."""
        ports = {instance.name: instance.port for instance in self.instances}
        if self.gliner:
            ports["gliner"] = self.gliner.port
        return {**ports, "gateway": self.gateway_port, "status": self.status_port}

    def status_keys(self) -> dict[str, Instance]:
        """The key of each instance in the status API: its alias, or alias@port for replicas."""
        return {(instance.model.alias if len(instance.model.gpus) == 1
                 else f"{instance.model.alias}@{instance.port}"): instance for instance in self.instances}

    def extras(self) -> list[str]:
        """Optional dependency extras the profile needs beyond ``serve``."""
        return ["gliner"] if self.gliner else []


# Loading -------------------------------------------------------------------------------------------

def load_profile(path: str | Path, gpu_count: int | None = None) -> Profile:
    """Read, resolve and validate a profile. ``gpu_count`` sizes ``gpus: all``; by default the
    allocation's visible GPUs are counted."""
    profile = parse_profile(resolve_gpus(read_profile(path), gpu_count), source=str(path))
    beyond = [f"{name} GPU {gpu}" for name, gpu in profile.gpus() if gpu_count is not None and gpu >= gpu_count]
    if beyond:
        raise ProfileError(f"the allocation has {gpu_count} GPU(s), but the profile places {', '.join(beyond)}")
    return profile


def read_profile(path: str | Path, _depth: int = 0) -> dict[str, Any]:
    """The profile as a mapping, with the environment substituted and ``extends`` merged in."""
    path = Path(path)

    def substitute(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1)) or match.group(2) or ""

    data = yaml.safe_load(_ENV.sub(substitute, path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: a profile is a YAML mapping")
    base = data.pop("extends", None)
    if base is None:
        return data
    if not isinstance(base, str) or _depth >= 4:
        raise ProfileError(f"{path}: extends must name one profile file (at most 4 levels)")
    return _merge(read_profile(path.parent / base, _depth + 1), data)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        both = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = _merge(merged[key], value) if both else value
    return merged


def visible_gpu_count() -> int | None:
    """GPUs this process may use: CUDA_VISIBLE_DEVICES (set by Slurm), else nvidia-smi; None if unknown."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is not None:
        return len([item for item in devices.split(",") if item.strip() and item != "NoDevFiles"])
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run([executable, "-L"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return sum(line.startswith("GPU ") for line in result.stdout.splitlines()) if result.returncode == 0 else None


def resolve_gpus(data: dict[str, Any], gpu_count: int | None = None) -> dict[str, Any]:
    """Replace ``gpus: all`` with the GPUs of the allocation that no other model claims."""
    models = data.get("models")
    if not isinstance(models, dict):
        return data
    wanting = [role for role, model in models.items() if isinstance(model, dict) and model.get("gpus") == "all"]
    if not wanting:
        return data
    if len(wanting) > 1:
        raise ProfileError(f"only one model can use gpus: all (found {', '.join(wanting)})")
    count = gpu_count if gpu_count is not None else visible_gpu_count()
    if count is None:
        raise ProfileError("gpus: all needs the allocation's GPUs: run inside the GPU job, or pass --gpus N")
    claimed = {gpu for model in models.values() if isinstance(model, dict) and isinstance(model.get("gpus"), list)
               for gpu in model["gpus"]}
    free = [gpu for gpu in range(count) if gpu not in claimed]
    if not free:
        raise ProfileError(f"models.{wanting[0]}.gpus: all, but no GPU of {count} is left for it")
    role = wanting[0]
    return {**data, "models": {**models, role: {**models[role], "gpus": free}}}


# Validation ----------------------------------------------------------------------------------------

def parse_profile(data: Any, source: str = "") -> Profile:
    if not isinstance(data, dict):
        raise ProfileError("a profile is a YAML mapping")
    unknown = set(data) - _TOP_FIELDS
    if unknown:
        raise ProfileError(f"unsupported profile fields: {', '.join(sorted(unknown))}")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ProfileError("profile.name is required")
    if not isinstance(data.get("models"), dict) or not data["models"]:
        raise ProfileError("profile.models must define at least one model")
    models = {role: _model(role, raw) for role, raw in data["models"].items()}
    gliner = _gliner(data.get("gliner"))
    profile = Profile(name=data["name"], models=models,
                      gateway_port=_integer(data.get("gateway_port", 4000), "gateway_port", 1),
                      status_port=_integer(data.get("status_port", 8010), "status_port", 1),
                      gliner=gliner, source=source, data=data)
    _check_placement(profile)
    return profile


def _check_placement(profile: Profile) -> None:
    """One process per GPU (GLiNER shares a model's), unique ports, aliases and service names."""
    owners: dict[int, str] = {}
    for instance in profile.instances:
        if instance.gpu in owners:
            raise ProfileError(f"GPU {instance.gpu} is assigned to both {owners[instance.gpu]} and {instance.name}")
        owners[instance.gpu] = instance.name
    if profile.gliner and profile.gliner.gpu not in owners:
        raise ProfileError(f"gliner.gpu: GPU {profile.gliner.gpu} serves no model; GLiNER shares a model's GPU")
    names = [instance.name for instance in profile.instances]
    duplicate = {name for name in names if names.count(name) > 1} | (set(names) & RESERVED)
    if duplicate:
        raise ProfileError(f"duplicate or reserved service name: {', '.join(sorted(duplicate))}")
    seen: dict[int, str] = {}
    for name, port in profile.ports().items():
        if not 1 <= port <= 65535:
            raise ProfileError(f"port {port} of {name} is outside 1-65535")
        if port in seen:
            raise ProfileError(f"port {port} is assigned to both {seen[port]} and {name}")
        seen[port] = name
    aliases = [model.alias for model in profile.models.values()]
    if len(set(aliases)) != len(aliases):
        raise ProfileError("each model needs its own alias")


def _model(role: Any, raw: Any) -> Model:
    if not isinstance(role, str) or not _NAME.fullmatch(role):
        raise ProfileError(f"invalid model role name: {role!r}")
    if role in RESERVED:
        raise ProfileError(f"models.{role} conflicts with a reserved service name")
    if not isinstance(raw, dict):
        raise ProfileError(f"models.{role} must be a mapping")
    task = raw.get("task", "generate")
    if task not in {"generate", "embed"}:
        raise ProfileError(f"models.{role}.task must be generate or embed")
    allowed = _MODEL_REQUIRED | _MODEL_OPTIONAL | (set() if task == "embed" else _CHAT_ONLY)
    unknown = set(raw) - allowed
    if unknown:
        raise ProfileError(f"unsupported fields for models.{role}: {', '.join(sorted(unknown))}")
    missing = _MODEL_REQUIRED - set(raw)
    if missing:
        raise ProfileError(f"models.{role} is missing: {', '.join(sorted(missing))}")
    where = f"models.{role}"
    gpus = raw["gpus"]
    if not isinstance(gpus, list) or not gpus:
        raise ProfileError(f"{where}.gpus must be a list of GPU indexes, or all")
    gpus = [_integer(gpu, f"{where}.gpus", 0) for gpu in gpus]
    fraction = raw["gpu_memory_utilization"]
    if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction <= 1:
        raise ProfileError(f"{where}.gpu_memory_utilization must be a number in (0, 1]")
    if raw.get("kv_cache_dtype", "auto") not in KV_CACHE_DTYPES:
        raise ProfileError(f"{where}.kv_cache_dtype must be one of: {', '.join(sorted(KV_CACHE_DTYPES))}")
    parser = raw.get("tool_call_parser")
    if parser is not None and (not isinstance(parser, str) or not _PARSER.fullmatch(parser)):
        raise ProfileError(f"{where}.tool_call_parser must be a vLLM parser name such as hermes")
    extra_args = raw.get("extra_args", [])
    if not isinstance(extra_args, list) or any(not isinstance(arg, str) or not arg for arg in extra_args):
        raise ProfileError(f"{where}.extra_args must be a list of vLLM arguments")
    managed = sorted({arg.split("=", 1)[0] for arg in extra_args} & _MANAGED_FLAGS)
    if managed:
        raise ProfileError(f"{where}.extra_args may not set {', '.join(managed)}: the launcher sets it")
    return Model(role=role, alias=_text(raw["alias"], f"{where}.alias"), model=_text(raw["model"], f"{where}.model"),
                 gpus=tuple(gpus), port=_integer(raw["port"], f"{where}.port", 1),
                 max_num_active_seqs=_integer(raw["max_num_active_seqs"], f"{where}.max_num_active_seqs", 1),
                 max_num_queued_reqs=_integer(raw["max_num_queued_reqs"], f"{where}.max_num_queued_reqs", 1),
                 max_model_len=_integer(raw["max_model_len"], f"{where}.max_model_len", 1),
                 gpu_memory_utilization=float(fraction),
                 revision=_optional_text(raw.get("revision"), f"{where}.revision"), task=task,
                 kv_cache_dtype=raw.get("kv_cache_dtype", "auto"), tool_call_parser=parser,
                 extra_args=tuple(extra_args))


def _gliner(section: Any) -> Gliner | None:
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ProfileError("gliner must be a mapping")
    unknown = set(section) - set(GLINER_DEFAULTS)
    if unknown:
        raise ProfileError(f"unsupported gliner fields: {', '.join(sorted(unknown))}")
    settings = {**GLINER_DEFAULTS, **section}
    if section.get("model", GLINER_DEFAULTS["model"]) != GLINER_DEFAULTS["model"] and "revision" not in section:
        settings["revision"] = None
    if type(settings["enabled"]) is not bool:
        raise ProfileError("gliner.enabled must be true or false")
    if not settings["enabled"]:
        return None
    return Gliner(gpu=_integer(settings["gpu"], "gliner.gpu", 0), port=_integer(settings["port"], "gliner.port", 1),
                  model=_text(settings["model"], "gliner.model"),
                  revision=_optional_text(settings["revision"], "gliner.revision"),
                  max_chars=_integer(settings["max_chars"], "gliner.max_chars", 1))


def _integer(value: Any, what: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ProfileError(f"{what} must be {'a positive' if minimum else 'a non-negative'} integer")
    return value


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{what} must be a non-empty string")
    return value


def _optional_text(value: Any, what: str) -> str | None:
    return None if value is None else _text(value, what)
