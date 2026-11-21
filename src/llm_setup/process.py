"""Child processes: their environment, the session marker that proves ownership, and signalling.

Every process a session starts carries ``LLM_SETUP_SESSION_ID=<session>`` in its environment. Only a
process whose /proc environment holds that exact marker is ever signalled, so a recycled PID or another
user's server is never touched.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER = "LLM_SETUP_SESSION_ID"
GATEWAY_SECRETS = ("LITELLM_MASTER_KEY", "LLM_BACKEND_KEY", "VLLM_API_KEY")


def owned(pid: int, session_id: str) -> bool:
    """True only when /proc shows this process carries this session's marker."""
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return f"{MARKER}={session_id}".encode() in environment


def signal_owned(pid: int, number: int) -> None:
    """Signal a session process and, when it leads its own process group, the whole group: vLLM starts
    engine workers that must stop with it."""
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, number)
        else:
            os.kill(pid, number)
    except ProcessLookupError:
        pass


def physical_gpu(base: dict[str, str], index: int) -> str:
    """The CUDA_VISIBLE_DEVICES entry for allocation GPU ``index``. Slurm may hand the job GPUs 4,5,6
    rather than 0,1,2; index 0 is then device 4, not another job's device 0. Outside an allocation
    (no CUDA_VISIBLE_DEVICES) the index is the device."""
    if "CUDA_VISIBLE_DEVICES" not in base:
        return str(index)
    visible = [item.strip() for item in base["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
    if index >= len(visible):
        raise RuntimeError(f"GPU {index} is not in this allocation (CUDA_VISIBLE_DEVICES={base['CUDA_VISIBLE_DEVICES']})")
    return visible[index]


def child_environment(base: dict[str, str], session_id: str, *, gpu: int | None = None,
                      drop: tuple[str, ...] = ()) -> dict[str, str]:
    """``base`` without the ``drop`` variables, with the session marker and, for a GPU service, one GPU."""
    env = {key: value for key, value in base.items() if key not in drop}
    env[MARKER] = session_id
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu(base, gpu)
    return env
