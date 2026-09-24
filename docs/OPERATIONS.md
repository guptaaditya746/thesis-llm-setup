# Operations

Request one allocation:

```sh
srun --gres=gpu:ampere:3 --cpus-per-task=16 --mem=64g --time=48:00:00 --pty bash -l
```

Keep the Python environment, managed Python, and uv cache on node-local scratch. The example `.env` sets these paths using `SLURM_TMPDIR`, or `/tmp` when that variable is unavailable. After sourcing `.env`, install the local Python and serving dependencies:

```sh
mkdir -p "$LLM_SETUP_LOCAL_ROOT"
uv python install 3.13
uv sync --python 3.13 --extra serve
```

This avoids importing Torch and vLLM from the shared Ceph filesystem, where CLI startup can block on file reads. Model weights remain in the Hugging Face cache and are not copied by setup. Set `EMBED_MODEL_ID` only if selecting a different embedding model. Then start with `uv run llm-setup start --profile profiles/a100-3x40.yaml`. Startup order is Embed vLLM, Heavy vLLM, Lite vLLM, LiteLLM, then status API. Each backend must pass `/health` before its dependent service starts. All listeners bind to `127.0.0.1`.

Useful commands:

```sh
uv run llm-setup verify --profile profiles/a100-3x40.yaml
uv run llm-setup smoke --profile profiles/a100-3x40.yaml
uv run llm-setup status
uv run llm-setup logs --service heavy
uv run llm-setup stop
```

Session process records, copied profile, generated gateway configuration, logs, and SQLite history live in `runtime/<session-id>/`. Stop before leaving the allocation. An SSH tunnel from a trusted workstation can forward the local gateway and status ports with `ssh -L 4000:127.0.0.1:4000 -L 8010:127.0.0.1:8010 <host>`; model ports remain local to the allocation.
