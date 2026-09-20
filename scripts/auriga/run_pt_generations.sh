#!/usr/bin/env bash
# Cross-generation flux dM under one particle truth.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
W128RAW=$HOME/.orx/runs/309e06b7-b2d9-4865-a9f8-094fe49acbab/repo/runs/orx
W128S1=$HOME/.orx/runs/638e02be-1e18-4436-9891-0b643266e65b/repo/runs/orx
W128CLEAN=$HOME/.orx/runs/97563c61-aabf-4287-9927-7d8a4f2a13df/repo/runs/orx
W1024RAW=$HOME/.orx/runs/b26b7505-ad14-4d8e-a417-e9faf956219f/repo/runs/orx
W1024CSM=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== PT GENERATIONS PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f data/auriga/halo12_particle_truth_grids.h5 || {
  echo "PREFLIGHT FAIL: missing grids product"; exit 2; }
for d in "$W128RAW" "$W128S1" "$W128CLEAN" "$W1024RAW" "$W1024CSM"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }

"$PY" scripts/auriga/plot_pt_generations.py \
  --grids data/auriga/halo12_particle_truth_grids.h5 \
  --model w128raw="$W128RAW" --model w128S1raw="$W128S1" \
  --model w128clean="$W128CLEAN" --model w1024raw="$W1024RAW" \
  --model w1024csm="$W1024CSM" \
  --output-dir figures/pt-generations

echo '=== OUTPUTS ==='
find figures/pt-generations -type f | sort
echo '=== PT GENERATIONS DONE ==='
