# Operations

## Start as a batch job (recommended)

```sh
sbatch slurm/llm-stack.sbatch                  # resources in the script's #SBATCH lines, or on the command line
tail -f runtime/slurm-<jobid>.out              # progress, events, summaries
srun --jobid <jobid> --overlap --pty bash -l   # a shell in the same allocation
scancel <jobid>                                # clean stop
```

The job sources `.env`, installs the environment on node-local scratch, and runs
`llm-setup start --foreground`, which:

- starts Embed, Heavy and Lite **in parallel** (one GPU each), then LiteLLM, then the status API, each
  only after the previous ones answer their health check;
- prints every 15 s what each model is doing, read from its log (downloading, loading weights NN%,
  sizing the KV cache, compiling, capturing CUDA graphs), and `ready after M:SS` for each;
- on a failed start, prints the end of the failing service's log with a hint, and stops the rest;
- then watches: a service whose process exits is restarted (LiteLLM and the status API after 5 s,
  a model after 30 s, doubling each time, a model at most 3 times an hour); a live service that stops
  answering its health check for a minute is reported, and again when it answers;
- prints a summary every 5 minutes (each service, restarts, running/waiting per model);
- on SIGTERM (`scancel`, `--signal=B:TERM@180` before the time limit) or Ctrl-C stops every process of the
  session (SIGTERM to each process group, SIGKILL after 30 s) and clears `runtime/current`.

Every event goes to `runtime/<session>/events.jsonl`: `llm-setup events` prints them, and the status API
serves them at `/v1/events`.

## Interactive allocation

```sh
srun --gres=gpu:ampere:3 --cpus-per-task=16 --mem=64g --time=48:00:00 --pty bash -l
```

Closing this terminal ends the allocation and every model in it; prefer the batch job for long runs.

Keep the Python environment, managed Python, and uv cache on node-local scratch. The example `.env` sets these paths using `SLURM_TMPDIR`, or `/tmp` when that variable is unavailable. After sourcing `.env`, install the local Python and serving dependencies:

```sh
mkdir -p "$LLM_SETUP_LOCAL_ROOT"
uv python install 3.13
uv sync --python 3.13 --extra serve
```

This avoids importing Torch and vLLM from the shared Ceph filesystem, where CLI startup can block on file reads. Model weights remain in the Hugging Face cache and are not copied by setup. Set `EMBED_MODEL_ID` only if selecting a different embedding model. Then start with `uv run llm-setup start --profile profiles/a100-3x40.yaml`. Startup order is Embed vLLM, Heavy vLLM, Lite vLLM, LiteLLM, then status API. Each backend must pass `/health` before its dependent service starts. All listeners bind to `127.0.0.1`.

Useful commands:

```sh
uv run llm-setup doctor --profile profiles/a100-3x40.yaml   # every check in order; names the first problem
uv run llm-setup events                                      # starts, exits, restarts, stops
uv run llm-setup verify --profile profiles/a100-3x40.yaml
uv run llm-setup smoke --profile profiles/a100-3x40.yaml
uv run llm-setup status
uv run llm-setup logs --service heavy
uv run llm-setup stop
```

`start` refuses to run while a recorded session still has live processes, and clears a recorded
session none of whose processes is alive (an allocation that ended without `stop`). It also refuses when
a profile port is already in use. Session process records, copied profile, generated gateway configuration, logs, and SQLite history live in `runtime/<session-id>/`. Stop before leaving the allocation. An SSH tunnel from a trusted workstation can forward the local gateway and status ports with `ssh -L 4000:127.0.0.1:4000 -L 8010:127.0.0.1:8010 <host>`; model ports remain local to the allocation.

## Load test

Measure before changing concurrency limits, context or memory fractions. Run inside the allocation while nothing else uses the models, one role at a time, with prompts close to real sizes (the harness sends roughly 4k to 20k input tokens):

```sh
for c in 1 2 4 8; do
  vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8002 \
    --endpoint /v1/chat/completions \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --served-model-name heavy-model \
    --dataset-name random --random-input-len 12000 --random-output-len 800 \
    --num-prompts $((c*4)) --max-concurrency $c 2>&1 \
    | grep -E "Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Request throughput"
done
curl -s http://127.0.0.1:8010/v1/status | python3 -m json.tool
```

For Lite use port 8003, `ibm-granite/granite-4.1-8b` and `lite-model`. Record time to first token, time per output token and throughput per concurrency level, plus the `preemptions` and `kvCache` values from `/v1/status`. Set `max_num_active_seqs` at the last level where time to first token stays acceptable and preemptions stay at zero.
