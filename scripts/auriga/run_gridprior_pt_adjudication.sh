#!/usr/bin/env bash
# Grid-prior pt-adjudication: Gauss-flux enclosed-mass bands + outer angular
# negative-density fraction of the grid-prior model (run 09faad30) against the
# committed particle-truth grid product, next to base (6855f18e) and S1
# (72240693) - the three decision-relevant w1024 clean+smooth Phis.
# Evaluation-only port of the pt-adjudication line (node fd47ea26): no
# retraining, no extra sweeps.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
GRIDPRIOR=$HOME/.orx/runs/09faad30-2d6f-482c-8d05-8194603c84f3/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== GRIDPRIOR PT-ADJUDICATION PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
for d in "$BASE" "$S1" "$GRIDPRIOR"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "grids:      $GRIDS"
echo "base:       $BASE"
echo "S1:         $S1"
echo "gridprior:  $GRIDPRIOR"

TRUTH_ARG=()
if [ -f "$TRUTH" ]; then TRUTH_ARG=(--truth "$TRUTH"); fi

"$PY" scripts/auriga/plot_pt_adjudication.py \
  --grids "$GRIDS" \
  --model base="$BASE" --model S1="$S1" --model gridprior="$GRIDPRIOR" \
  "${TRUTH_ARG[@]}" --output-dir figures/pt-adjudication-gridprior

echo '=== OUTPUTS ==='
find figures/pt-adjudication-gridprior -type f | sort
echo '=== GRIDPRIOR PT-ADJUDICATION DONE ==='
