#!/bin/bash
# Start the stack inside a Slurm job you already hold (e.g. an interactive `srun ... --pty bash` with
# gpu:ampere:3), instead of submitting a new batch job. Run it on the login node from anywhere:
#
#   LLM_SETUP_PROFILE=profiles/orch-3x40-gliner.yaml slurm/start-in-job.sh <jobid>
#   tail -f runtime/slurm-<jobid>.out            # same progress lines as the batch job
#
# It runs slurm/llm-stack.sbatch as an extra job step (`srun --overlap`) in the Webis container, detached
# from this terminal. Stop it with `scancel <jobid>.<step>` (see `squeue -s -j <jobid>`) or end the job.
# Optional: LLM_SETUP_CONTAINER (default: the verified pytorch image; set to "none" on a native node),
# GLINER_PORT (when another process on the node holds 8090).
set -euo pipefail
job="${1:?usage: slurm/start-in-job.sh <jobid>}"
here="$(cd "$(dirname "$0")/.." && pwd)"   # absolute: the container does not start in this directory
image="${LLM_SETUP_CONTAINER:-pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime}"
log="$here/runtime/slurm-$job.out"
mkdir -p "$here/runtime"

container=()
if [ "$image" != none ]; then
  container=(--container-image="$image" --container-writable --container-remap-root)
  export LLM_SETUP_BOOTSTRAP_COMPILER="${LLM_SETUP_BOOTSTRAP_COMPILER:-1}"
fi

# The step needs the job's GPUs explicitly; --mem=0 shares the job's memory instead of asking for more.
SLURM_SUBMIT_DIR="$here" setsid nohup srun --jobid "$job" --overlap --gres=gpu:ampere:3 --mem=0 \
  "${container[@]}" bash -c "cd '$here' && exec bash '$here/slurm/llm-stack.sbatch'" \
  >"$log" 2>&1 </dev/null &
echo "started in job $job; log: $log"
echo "ready when the log shows 'ready after' for each model, then the gateway and status API"
