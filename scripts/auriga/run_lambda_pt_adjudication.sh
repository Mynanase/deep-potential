#!/usr/bin/env bash
# Lambda-sweep pt-adjudication: Gauss-flux enclosed-mass bands of the
# radius-balanced grid prior at lambda 0.1 / 1 / 10 against the production
# anchors, under the committed particle-truth grid product:
#   base      6855f18e  (production co-winner)
#   S1        72240693  (gridprior parent)
#   gridprior 09faad30  (volume grid)
#   innerA    2b32eb04  (radius-balanced, lambda=1 - current route candidate)
#   lambda0p1 f6c902f6  (radius-balanced, lambda=0.1 - negativity collapsed)
#   lambda10  e1cf8f74  (radius-balanced, lambda=10 - negativity ~0.8%)
# Evaluation-only port of the pt-adjudication line (node 4b21457d): no
# retraining, no extra sweeps. Decision: dM bands per lambda point; does
# lambda=10's near-zero negativity cost mass-recovery accuracy vs lambda=1?
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
GRIDPRIOR=$HOME/.orx/runs/09faad30-2d6f-482c-8d05-8194603c84f3/repo/runs/orx
INNERA=$HOME/.orx/runs/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo/runs/orx
LAMBDA0P1=$HOME/.orx/runs/f6c902f6-fbb9-4b18-8577-1f9f56b21d3a/repo/runs/orx
LAMBDA10=$HOME/.orx/runs/e1cf8f74-775d-44b1-b567-24dd88e48cd0/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== LAMBDA PT-ADJUDICATION PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
for d in "$BASE" "$S1" "$GRIDPRIOR" "$INNERA" "$LAMBDA0P1" "$LAMBDA10"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "grids:      $GRIDS"
echo "base:       $BASE"
echo "S1:         $S1"
echo "gridprior:  $GRIDPRIOR"
echo "innerA:     $INNERA"
echo "lambda0p1:  $LAMBDA0P1"
echo "lambda10:   $LAMBDA10"

TRUTH_ARG=()
if [ -f "$TRUTH" ]; then TRUTH_ARG=(--truth "$TRUTH"); fi

"$PY" scripts/auriga/plot_pt_adjudication.py \
  --grids "$GRIDS" \
  --model base="$BASE" --model S1="$S1" --model gridprior="$GRIDPRIOR" \
  --model innerA="$INNERA" --model lambda0p1="$LAMBDA0P1" --model lambda10="$LAMBDA10" \
  "${TRUTH_ARG[@]}" --output-dir figures/pt-adjudication-lambda

echo '=== OUTPUTS ==='
find figures/pt-adjudication-lambda -type f | sort
echo '=== LAMBDA PT-ADJUDICATION DONE ==='
