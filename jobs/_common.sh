#!/usr/bin/env bash

# Shared setup for scheduler-free server jobs.

set -euo pipefail

jobs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${DEEP_POTENTIAL_ROOT:-$(cd -- "${jobs_dir}/.." && pwd)}"
conda_env="${CONDA_ENV_NAME:-dp-jax}"

cd "$repo_root"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$conda_env"

if [[ -n "${GPU_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "${LOG_DIR:-logs}"

echo "[job] repo=$repo_root"
echo "[job] conda_env=$conda_env"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all visible devices}"
