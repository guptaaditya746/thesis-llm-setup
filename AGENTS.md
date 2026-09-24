# Repository guide

## Map

- `profiles/`: deployment values and aliases.
- `src/llm_setup/`: validation, launch supervision, status, metrics, and CLI.
- `config/`: LiteLLM config template; runtime config is generated per session.
- `docs/`: operator and consumer guidance.
- `tests/`: GPU-free contract tests.
- `runtime/`: local session state, logs, and SQLite; ignored by git.

## Safe commands

- `uv run llm-setup profile validate --profile profiles/a100-3x40.yaml`
- `uv run llm-setup verify --profile profiles/a100-3x40.yaml`
- `uv run llm-setup start --profile profiles/a100-3x40.yaml`
- `uv run llm-setup status`, `uv run llm-setup logs --service heavy`, `uv run llm-setup stop`
- `uv run ruff check .` and `uv run pytest -q`

## Prohibited actions

- Never edit the older `llm_setup` repository or the research harness from this project.
- Never use broad process matching or kill a PID unless the recorded session marker matches.
- Never bind vLLM to a non-loopback address, delete model caches, or clean outside this repository's `runtime/`.
- Never commit `.env`, tokens, API keys, Hugging Face credentials, or secrets in logs.
- Never add cross-role fallback between Heavy, Lite, and Embed.

## Which document to read

- Starting or stopping a Slurm session: `docs/OPERATIONS.md`.
- Failures and recovery: `docs/DEBUGGING.md` and `docs/INCIDENTS.md`.
- Calling the service: `docs/CONSUMERS.md`.
- Changing model placement or capacity: `docs/MODEL_PROFILES.md`.
