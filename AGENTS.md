# Repository guide

## Map

- `slurm/submit.sh`: login-node entry point (new job, or a step in an existing job with `--job`).
- `slurm/job.sh`: runs inside the allocation; installs on node-local scratch and runs the supervisor.
- `profiles/`: what runs where (`gpus: all`, `extends:`; see `docs/PROFILES.md`).
- `src/llm_setup/`: `profile.py` (loading, validation), `supervisor.py` (start/stop), `watchdog.py`
  (restarts), `vllm.py` and `gateway.py` (launch commands and the LiteLLM config), `runtime.py`
  (session folders), `process.py` (ownership marker, signals), `status.py`, `doctor.py`, `cli.py`.
- `tests/`: GPU-free tests. `runtime/`: session state and logs; ignored by git.

## Safe commands

- `uv run llm-setup profile show --profile profiles/orch.yaml --gpus 3`
- `slurm/submit.sh [--gpus N] [--profile FILE]`, `slurm/submit.sh --job <jobid>`
- `uv run llm-setup doctor`, `status`, `events`, `logs <service>`, `stop`
- `uv run --extra dev ruff check .` and `uv run --extra dev pytest -q`

## Prohibited actions

- Never edit the older `llm_setup` repository or the research harness from this project.
- Never use broad process matching or kill a PID unless it carries the session marker.
- Never bind a service to a non-loopback address, delete model caches, override the cluster's
  Hugging Face cache variables, or clean outside this repository's `runtime/`.
- Never commit `.env`, tokens, API keys, Hugging Face credentials, or secrets in logs.
- Never add fallbacks between different models in the gateway.
- Never stop another user's process that holds one of our ports; move our port instead.

## Which document to read

- Starting, stopping, the cluster cache, GLiNER, load tests: `docs/OPERATIONS.md`.
- Changing models, placement or capacity: `docs/PROFILES.md`.
- Failures, recovery and the incident log: `docs/DEBUGGING.md`.
