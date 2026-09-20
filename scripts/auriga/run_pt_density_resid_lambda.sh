#!/usr/bin/env bash
# Lambda-round density maps + log residuals vs the committed particle-truth
# product (96^3 mass histogram midplane), in the latest pt-density-resid
# variants style, extended to FOUR models:
#   base      6855f18e  (seed-0 production anchor)
#   innerA    2b32eb04  (radius-balanced grid, lambda=1 - frozen route)
#   lambda0p1 f6c902f6  (radius-balanced grid, lambda=0.1)
#   lambda10  e1cf8f74  (radius-balanced grid, lambda=10)
# Layout 2x5: truth + four model log densities; radial profile + four
# log10(rho_m/rho_t) residual maps. Colors follow the lambda
# pt-adjudication figure. Evaluation only; no retraining.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
INNERA=$HOME/.orx/runs/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo/runs/orx
LAMBDA0P1=$HOME/.orx/runs/f6c902f6-fbb9-4b18-8577-1f9f56b21d3a/repo/runs/orx
LAMBDA10=$HOME/.orx/runs/e1cf8f74-775d-44b1-b567-24dd88e48cd0/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== LAMBDA DENSITY RESID PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
test -f scripts/auriga/plot_pt_density_resid_lambda.py || { echo "PREFLIGHT FAIL: missing script"; exit 2; }
for d in "$BASE" "$INNERA" "$LAMBDA0P1" "$LAMBDA10"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "grids:     $GRIDS"
echo "base:      $BASE"
echo "innerA:    $INNERA"
echo "lambda0p1: $LAMBDA0P1"
echo "lambda10:  $LAMBDA10"

TRUTH_ARG=()
if [ -f "$TRUTH" ]; then TRUTH_ARG=(--truth "$TRUTH"); fi

"$PY" scripts/auriga/plot_pt_density_resid_lambda.py \
  --grids "$GRIDS" \
  --model base="$BASE" --model innerA="$INNERA" \
  --model lambda0p1="$LAMBDA0P1" --model lambda10="$LAMBDA10" \
  "${TRUTH_ARG[@]}" --output-dir figures/pt-density-resid-lambda

echo '=== OUTPUTS ==='
find figures/pt-density-resid-lambda -type f | sort
echo '=== LAMBDA DENSITY RESID DONE ==='
