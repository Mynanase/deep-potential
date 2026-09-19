#!/usr/bin/env bash
# Merged potential slices figure with vc2-normalized residuals.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S11=$HOME/.orx/runs/ff328913-7bf8-4f38-9fc9-bb1c91b34867/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== PT NORM PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f data/auriga/halo12_particle_truth_grids.h5 || {
  echo "PREFLIGHT FAIL: missing data/auriga/halo12_particle_truth_grids.h5"; exit 2; }
for d in "$BASE" "$S11" "$S1"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'

TRUTH_ARG=()
if [ -f "$TRUTH" ]; then TRUTH_ARG=(--truth "$TRUTH"); fi

"$PY" scripts/auriga/plot_pt_potential_norm.py \
  --grids data/auriga/halo12_particle_truth_grids.h5 \
  --model base="$BASE" --model s11="$S11" --model S1="$S1" \
  --output-dir figures/pt-potential-norm

"$PY" scripts/auriga/plot_pt_adjudication.py \
  --grids data/auriga/halo12_particle_truth_grids.h5 \
  --model base="$BASE" --model s11="$S11" --model S1="$S1" \
  "${TRUTH_ARG[@]}" --output-dir figures/pt-adjudication

echo '=== OUTPUTS ==='
find figures/pt-potential-norm figures/pt-adjudication -type f | sort
echo '=== PT NORM DONE ==='
