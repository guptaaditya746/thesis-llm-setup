# Operations

## Starting

From the login node, in this repository:

```sh
squeue -u "$USER"                  # submit.sh also refuses a second llm-stack job
slurm/submit.sh [--gpus N] [--profile FILE] [--node NAME] [--time T]
tail -f runtime/slurm-<jobid>.out
```

`submit.sh` checks `.env` and the profile, then submits `slurm/job.sh` with
`--partition=gammaweb --gres=gpu:ampere:N` and the cluster container
(`pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime`, writable, root remapped). `--native` runs without
the container; `LLM_SETUP_PARTITION`, `LLM_SETUP_GPU_TYPE` and `LLM_SETUP_CONTAINER` change the
defaults; arguments after `--` go to `sbatch`. Stay on Ampere: the profiles are sized for A100 40 GB.

**Inside a job you already hold** (for example `srun --gres=gpu:ampere:3 ... --pty bash`):
`slurm/submit.sh --job <jobid>` runs the same `job.sh` as an extra step (`srun --overlap` with the
job's GPUs and `--mem=0`), detached from the terminal, logging to `runtime/slurm-<jobid>.out`.
Stop it with `scancel <jobid>.<step>` (`squeue -s -j <jobid>` lists steps) or by ending the job.

## What the job does

`slurm/job.sh` sources `.env`, installs Python 3.13 and the serving environment on node-local
scratch (`LLM_SETUP_LOCAL_ROOT`; importing torch and vLLM from the shared filesystem can stall),
adds the extras the profile needs (GLiNER), installs a C compiler for Triton in the container if
missing (skipped with the prebuilt image: run `slurm/build-image.sh`
once; it saves `runtime/images/pytorch-2.8.0-cuda12.9-gcc.sqsh`, which `submit.sh` then uses, so a slow
package mirror cannot stall the start), gives vLLM a per-job rendezvous port block, then runs `llm-setup start --foreground`, which:

- starts every model replica **in parallel**, then GLiNER, then LiteLLM, then the status API, each
  only after the previous ones answer their health check;
- prints every 15 s what each model is doing (downloading, loading weights NN%, sizing the KV cache,
  compiling, capturing CUDA graphs) and `ready after M:SS` for each;
- on a failed start prints the end of the failing service's log with a hint and stops the rest
  (a GLiNER failure is the exception: the stack runs without it);
- then watches: a crashed service is restarted (the gateway and status API after 5 s, a model
  after 30 s, doubling each time, a model at most 3 times an hour); a live service that stops
  answering its health check for a minute is reported, and again when it answers;
- prints a summary every 5 minutes;
- on SIGTERM (`scancel`, or `--signal=B:TERM@180` before the time limit) or Ctrl-C stops every
  process of the session (SIGTERM to each process group, SIGKILL after 30 s).

Every event goes to `events.jsonl` in the session folder (`llm-setup events`, `GET :8010/v1/events`).

The session folder lives on node-local disk (`$LLM_SETUP_LOCAL_ROOT/runtime/<session>`): the services
and the supervisor never write to Ceph. A separate mirror process (`llm_setup.mirror`) copies it to
`runtime/<session>` every minute and once more at stop, so on the login node logs and events are up
to a minute old. If Ceph hangs, only the mirror (and `cat` feeding the job log) wait; the GPU services
keep running and stop on `scancel`. A process stuck on a hung Ceph still keeps the job in CG until Ceph
recovers; that needs the admins. `llm-setup stop` from a shell that cannot see the node-local folder
(the login node, or a shell outside the job's container) writes `runtime/<session>/stop-requested`;
the mirror stops the session within a minute. `scancel` is still the usual stop.
Only processes carrying the session's marker are ever signalled.

## Model cache (Webis)

The cluster container supplies `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `XDG_CACHE_HOME`,
`HF_ASSETS_CACHE` and `HF_XET_CACHE`. Never set or replace them, copy weights to `/tmp`, or create a
personal cache. Caches under `/var/tmp` are node-local: check that the pinned revisions are on the
node before submitting, and choose it with `--node`. A short CPU-only check of a node:

```sh
srun --partition=gammaweb --nodelist=<node> --cpus-per-task=1 --mem=16G --time=00:15:00 \
  --container-image=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime \
  --container-writable --container-remap-root --pty bash -l
for v in HF_HUB_CACHE HF_DATASETS_CACHE XDG_CACHE_HOME HF_ASSETS_CACHE HF_XET_CACHE; do
  printf '%s=%s\n' "$v" "${!v-UNSET}"        # never print HF_TOKEN
done
```

If the variables are missing or point somewhere unexpected, stop. Look snapshots up with
`huggingface_hub.snapshot_download(..., revision=..., local_files_only=True)`; a snapshot directory
alone does not prove every file is there. Do not download the large Qwen model repeatedly.
`HF_HUB_OFFLINE` is inherited: set it to 1 only when the pinned revisions and GLiNER's backbone
tokenizer are already cached. The container needs more than 8 GB of job memory during its first
import (the job asks for 64 GB) and package repository access for the compiler bootstrap.

## GLiNER

An optional named-entity service for the thesis evidence engine (`src/llm_setup/gliner_server.py`),
on in `profiles/orch-gliner.yaml`. It starts after the models (vLLM sizes its memory from what is
free) and shares GPU 0; it needs an estimated 1.5-2 GB, and a model at
`gpu_memory_utilization: 0.92` leaves about 3 GB of an A100 40 GB. If GPU 0 has under ~1 GB free
after start, lower orch's fraction to 0.88 in `orch-gliner.yaml`. If it fails to start, the stack
runs without it (`gliner: start failed, continuing without it`); the watchdog restarts it like a
model once it has been ready.

API on `127.0.0.1:${GLINER_PORT:-8090}`, which the evidence engine reads from `GLINER_URL`:

- `GET /health`; `GET /info` -> `{"model", "revision", "device", "max_chars"}`
- `POST /predict` `{"text": str, "labels": [str], "threshold": 0.5}` ->
  `{"entities": [{"start", "end", "text", "label", "score"}]}`, character offsets into `text`.
  A text longer than `max_chars` gets HTTP 413: GLiNER medium reads about 384 words and would drop
  the rest silently, so callers send smaller chunks.

First run on a node: check the cache variables (above), make sure the pinned Qwen and GLiNER
snapshots and GLiNER's backbone tokenizer (`microsoft/deberta-v3-base`) are there, submit, then
check the replicas, gateway, `curl :8090/info` and `nvidia-smi` memory (`llm-setup doctor` does
all of it). A running job alone does not mean GLiNER is healthy.

## Load test

Measure before changing concurrency limits, context or memory fractions. Inside the allocation,
while nothing else uses the models, one replica at a time, with prompts close to real sizes:

```sh
for c in 1 2 4 8; do
  vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8001 \
    --endpoint /v1/chat/completions --model Qwen/Qwen3.8-27B-FP8 --served-model-name orch-model \
    --dataset-name random --random-input-len 12000 --random-output-len 800 \
    --num-prompts $((c*4)) --max-concurrency $c 2>&1 \
    | grep -E "Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Request throughput"
done
curl -s http://127.0.0.1:8010/v1/status | python3 -m json.tool
```

Record time to first token, time per output token and throughput per level, plus `preemptions`
and `kvCache`. Set `max_num_active_seqs` at the last level where time to first token stays
acceptable and preemptions stay at zero.

## Without Slurm's batch job

In an interactive allocation, after `set -a; source .env; set +a` and
`uv sync --python 3.13 --extra serve`: `uv run llm-setup start --profile <file>` starts the stack
and returns; `uv run llm-setup stop` stops it. Closing the terminal ends the allocation and every
model in it, so prefer `submit.sh` for anything long.
