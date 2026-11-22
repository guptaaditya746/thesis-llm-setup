#!/bin/bash
# The stack inside a Slurm allocation: install the environment on node-local scratch, then run
# `llm-setup start --foreground` until scancel or the time limit. Start it with slurm/submit.sh
# (a new job, or a step of a job you hold); plain `sbatch slurm/job.sh` uses the defaults below.
#SBATCH --job-name=llm-stack
#SBATCH --gres=gpu:ampere:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runtime/slurm-%j.out
# SIGTERM three minutes before the time limit, so the services stop cleanly instead of being killed.
#SBATCH --signal=B:TERM@180

set -euo pipefail
cd "${LLM_SETUP_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
PROFILE="${LLM_SETUP_PROFILE:-profiles/orch-gliner.yaml}"

if [ ! -f .env ]; then
  stamp "no .env in $PWD: copy .env.example to .env and set LITELLM_MASTER_KEY"; exit 1
fi
set -a; source .env; set +a

# vLLM picks a torch rendezvous port, closes it, then binds it again; another process can take it in
# between (EADDRINUSE). Give each job its own non-ephemeral 16-port block; instances step through it.
export VLLM_PORT="${VLLM_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 1250) * 16))}"

# Minimal runtime containers may lack the C/C++ toolchain Triton needs. slurm/submit.sh opts in for its
# writable Debian/Ubuntu container; native jobs are unchanged.
if [ "${LLM_SETUP_BOOTSTRAP_COMPILER:-0}" = 1 ] && { ! command -v cc >/dev/null || ! command -v c++ >/dev/null; }; then
  stamp "installing the compiler Triton needs in the writable container"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev
fi
mkdir -p "$LLM_SETUP_LOCAL_ROOT"

stamp "job ${SLURM_JOB_ID:-?} on $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-none visible}; profile $PROFILE"
stamp "a shell on this node:      srun --jobid ${SLURM_JOB_ID:-?} --overlap --pty bash -l"
stamp "ports from a workstation:  ssh -L 4000:127.0.0.1:4000 -L 8010:127.0.0.1:8010 $(hostname)"

stamp "installing Python and the serving environment on node-local scratch (a few minutes)"
uv python install 3.13
uv sync --python 3.13 --extra serve --no-editable
# A service the profile adds beyond the models (GLiNER) needs its extra too.
needed="$("$UV_PROJECT_ENVIRONMENT/bin/llm-setup" profile extras --profile "$PROFILE")"
extras=()
for extra in $needed; do
  extras+=(--extra "$extra")
done
if [ ${#extras[@]} -gt 0 ]; then
  stamp "the profile also needs: ${extras[*]}"
  uv sync --python 3.13 --extra serve --no-editable "${extras[@]}"
fi

# llm-setup replaces this shell, so Slurm's SIGTERM (scancel, the time limit) reaches it directly.
# Its session folder is node-local ($LLM_SETUP_LOCAL_ROOT/runtime, copied to runtime/ every minute), so
# the services never write to the shared filesystem. Its own output reaches the job log on that
# filesystem through `cat`, which ignores SIGTERM so the supervisor's last lines still arrive: if the
# filesystem hangs, cat waits and llm-setup keeps running and can still stop the GPU services. The
# package is installed non-editable, so a restarted service imports nothing from the shared filesystem.
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
stamp "starting the stack in the foreground"
exec "$UV_PROJECT_ENVIRONMENT/bin/llm-setup" start --profile "$PROFILE" --foreground > >(trap '' TERM INT HUP; exec cat) 2>&1
