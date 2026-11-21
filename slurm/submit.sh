#!/bin/bash
# Start the LLM stack on Slurm, from the login node.
#
#   slurm/submit.sh                          # new job: 3 GPUs, profiles/orch-gliner.yaml
#   slurm/submit.sh --gpus 1 -p profiles/orch.yaml
#   slurm/submit.sh --job 12345              # as a step of a GPU job you already hold
#
# Options:
#   -p, --profile FILE  serving profile (default: $LLM_SETUP_PROFILE, else profiles/orch-gliner.yaml)
#   -g, --gpus N        GPUs; a `gpus: all` model runs one replica per GPU (default 3)
#   -t, --time T        time limit of a new job (default 48:00:00)
#   -n, --node NAME     run on this node (one that has the pinned weights cached)
#   -j, --job ID        start inside job ID (e.g. an interactive srun) instead of submitting one
#       --native        no container: the node has its own CUDA runtime and compiler
#       --force         submit even though you already have a running llm-stack job
#   -- ARGS             passed on to sbatch
#
# Cluster defaults, overridable in the environment:
#   LLM_SETUP_PARTITION=gammaweb  LLM_SETUP_GPU_TYPE=ampere
#   LLM_SETUP_CONTAINER=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"      # absolute: a container does not start in this directory

profile="${LLM_SETUP_PROFILE:-profiles/orch-gliner.yaml}" gpus=3 time=48:00:00 node="" job="" native=0 force=0
partition="${LLM_SETUP_PARTITION-gammaweb}" gpu_type="${LLM_SETUP_GPU_TYPE:-ampere}"
image="${LLM_SETUP_CONTAINER:-pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime}"
die() { echo "submit.sh: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--profile) profile="${2:?}"; shift 2 ;;
    -g|--gpus) gpus="${2:?}"; shift 2 ;;
    -t|--time) time="${2:?}"; shift 2 ;;
    -n|--node) node="${2:?}"; shift 2 ;;
    -j|--job) job="${2:?}"; shift 2 ;;
    --native) native=1; shift ;;
    --force) force=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    *) die "unknown option $1 (--help lists them)" ;;
  esac
done
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || die "--gpus takes a positive number, not $gpus"
# A profile given relative to the caller's directory keeps working after the cd below.
if [[ "$profile" != /* && -f "$profile" ]]; then
  profile="$(cd "$(dirname "$profile")" && pwd)/$(basename "$profile")"
fi
cd "$here"
[ -f "$profile" ] || die "no profile $profile"
[ -f .env ] || die "no .env: cp .env.example .env and set LITELLM_MASTER_KEY"
grep -Eq '^LITELLM_MASTER_KEY=.+' .env && ! grep -q '^LITELLM_MASTER_KEY=replace-with-a-local-secret' .env \
  || die "set LITELLM_MASTER_KEY in .env (not the placeholder)"
if command -v uv >/dev/null; then                 # fail here, not after the job waited in the queue
  uv run --quiet --no-dev llm-setup profile show --profile "$profile" --gpus "$gpus" || die "profile check failed"
fi

export LLM_SETUP_PROFILE="$profile" LLM_SETUP_DIR="$here"
container=()
if [ "$native" = 0 ]; then
  container=(--container-image="$image" --container-writable --container-remap-root)
  export LLM_SETUP_BOOTSTRAP_COMPILER="${LLM_SETUP_BOOTSTRAP_COMPILER:-1}"
fi
mkdir -p runtime

if [ -n "$job" ]; then
  [ $# -eq 0 ] || die "arguments after -- go to sbatch; --job starts a step and takes none"
  [ -n "$(squeue -h -j "$job" -t RUNNING -o %i 2>/dev/null)" ] || die "job $job is not running"
  log="$here/runtime/slurm-$job.out"
  # An overlapping step sees no GPU unless it asks for the job's; --mem=0 shares the job's memory.
  # The step must ask for no more GPUs than the job holds: pass --gpus 1 for a 1-GPU job.
  setsid nohup srun --jobid "$job" --overlap --gres="gpu:$gpu_type:$gpus" --mem=0 \
    ${container[@]+"${container[@]}"} bash -c "cd '$here' && exec bash slurm/job.sh" >"$log" 2>&1 </dev/null &
  step=$!
  sleep 5
  if ! kill -0 "$step" 2>/dev/null; then
    tail -n 20 "$log" >&2
    die "the step ended at once (wrong --gpus for this job? see $log)"
  fi
  echo "started in job $job (profile $profile, $gpus GPUs)"
  echo "stop it:  scancel $job.<step>   (squeue -s -j $job lists the steps)"
else
  running="$(squeue -h -u "${USER:-$(id -un)}" -n llm-stack -o %i 2>/dev/null || true)"
  [ -z "$running" ] || [ "$force" = 1 ] || die "you already run llm-stack job(s) $running; scancel it or pass --force"
  args=(--parsable --gres="gpu:$gpu_type:$gpus" --time="$time")
  [ -n "$partition" ] && args+=(--partition="$partition")
  [ -n "$node" ] && args+=(--nodelist="$node")
  job="$(sbatch "${args[@]}" ${container[@]+"${container[@]}"} "$@" slurm/job.sh)"
  job="${job%%;*}"
  log="$here/runtime/slurm-$job.out"
  echo "submitted job $job (profile $profile, $gpus GPUs)"
  echo "stop it:  scancel $job"
fi
echo "progress: tail -f $log      (ready when the gateway and status API report 'ready after')"
echo "a shell:  srun --jobid $job --overlap --pty bash -l"
