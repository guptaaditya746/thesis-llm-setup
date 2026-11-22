#!/bin/bash
# Build the stack's container image once: the verified pytorch image plus the C/C++ compiler Triton
# needs, saved as a squashfs file. Jobs started from it skip the per-start `apt-get` (which can hang for
# many minutes when the package mirror is slow). Run it on the login node; it takes one short CPU job:
#
#   slurm/build-image.sh                          # extra srun options may follow, e.g. --exclude=gammaweb08
#
# Then slurm/submit.sh uses the image automatically (LLM_SETUP_CONTAINER still overrides it).
# Optional: LLM_SETUP_IMAGE (output path, default runtime/images/pytorch-2.8.0-cuda12.9-gcc.sqsh).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
base="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"
out="${LLM_SETUP_IMAGE:-$here/runtime/images/pytorch-2.8.0-cuda12.9-gcc.sqsh}"
partial="${out%.sqsh}.partial.sqsh"
mkdir -p "$(dirname "$out")"
rm -f "$partial"

# The image import needs more than 8 GB of memory (see docs/OPERATIONS.md).
srun --partition=gammaweb --cpus-per-task=4 --mem=32G --time=00:45:00 --job-name=llm-image "$@" \
  --container-image="$base" --container-writable --container-remap-root --container-save="$partial" \
  bash -c 'set -e
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev
    apt-get clean; rm -rf /var/lib/apt/lists/*
    cc --version | head -1'

# Only a finished save replaces the image, so a cancelled build never leaves a broken one in use.
mv -f "$partial" "$out"
echo "saved $out"
