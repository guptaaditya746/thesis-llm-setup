#!/bin/bash
# The model stack as a Slurm batch job: it keeps running when your terminal closes.
#
#   sbatch slurm/llm-stack.sbatch                     # from the repository root; prints the job id
#   tail -f runtime/slurm-<jobid>.out                 # startup progress, then restarts and summaries
#   srun --jobid <jobid> --overlap --pty bash -l      # a shell on the same node (doctor, harness, tunnels)
#   scancel <jobid>                                   # stops every service cleanly
#
# Change the resources below or on the command line (sbatch --time=24:00:00 ...). Another profile:
# LLM_SETUP_PROFILE=profiles/other.yaml sbatch slurm/llm-stack.sbatch
#SBATCH --job-name=llm-stack
#SBATCH --gres=gpu:ampere:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runtime/slurm-%j.out
# SIGTERM three minutes before the time limit, so the services stop cleanly instead of being killed.
#SBATCH --signal=B:TERM@180

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
PROFILE="${LLM_SETUP_PROFILE:-profiles/a100-3x40.yaml}"

# vLLM normally asks the kernel for an ephemeral rendezvous port, closes the
# probing socket, then binds that port in torch.distributed. Another process
# can claim it in that gap, making model startup fail with EADDRINUSE. Give
# each Slurm job a stable, non-ephemeral 16-port block instead. vLLM scans
# forward from VLLM_PORT if a port in the block is already occupied.
export VLLM_PORT="$((12000 + (SLURM_JOB_ID % 1250) * 16))"
stamp "vLLM distributed port block starts at $VLLM_PORT (job ${SLURM_JOB_ID:-?})"

if [ ! -f .env ]; then
  stamp "no .env in $PWD: copy .env.example to .env and set LITELLM_MASTER_KEY"; exit 1
fi
set -a; source .env; set +a
mkdir -p "$LLM_SETUP_LOCAL_ROOT"

stamp "job ${SLURM_JOB_ID:-?} on $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-none visible}"
stamp "a shell on this node:  srun --jobid ${SLURM_JOB_ID:-?} --overlap --pty bash -l"
stamp "ports from a workstation:  ssh -L 4000:127.0.0.1:4000 -L 8010:127.0.0.1:8010 $(hostname)"

stamp "installing Python and the serving environment on node-local scratch (a few minutes)"
uv python install 3.13
uv sync --python 3.13 --extra serve
# A service the profile enables beyond the models (GLiNER: gliner.enabled: true) needs its extra too.
extras="$("$UV_PROJECT_ENVIRONMENT/bin/llm-setup" profile extras --profile "$PROFILE")"
if [ -n "$extras" ]; then
  stamp "the profile enables: $extras; installing those extras"
  uv sync --python 3.13 --extra serve $(for extra in $extras; do printf -- '--extra %s ' "$extra"; done)
fi

# llm-setup replaces this shell, so Slurm's SIGTERM (scancel, the time limit) reaches it directly.
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
stamp "starting the stack in the foreground"
exec "$UV_PROJECT_ENVIRONMENT/bin/llm-setup" start --profile "$PROFILE" --foreground
