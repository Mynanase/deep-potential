#!/usr/bin/env bash
# Three figures for the route-relevant lambda-round set - base, lambda10,
# innerA, gridprior - in the latest published styles (no retraining):
#   1+2. shell-error-pairs.svg + shell-error-pairs-permodel.svg
#        (node 3da61b51 style: cumulative M(<r), per-shell M_shell, e/M +
#        E/M + Poisson floors, kappa(r); per-model small multiples)
#   3.   pt-generations.pdf (node dac88948 style: Gauss-flux dM vs the
#        particle monopole, truth-compare panels)
# New output dirs (...-lambda) keep the published 5-model shell-error and
# 6-generation figures untouched.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
TRUTH60=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
PARTICLES=/localdisk/kosmos/my-deep-potential/data/auriga/halo12_total_matter_particles_starframe.h5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
LAMBDA10=$HOME/.orx/runs/e1cf8f74-775d-44b1-b567-24dd88e48cd0/repo/runs/orx
INNERA=$HOME/.orx/runs/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo/runs/orx
GRIDPRIOR=$HOME/.orx/runs/09faad30-2d6f-482c-8d05-8194603c84f3/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== LAMBDA-ROUND FIGURES PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
test -f "$TRUTH60" || { echo "PREFLIGHT FAIL: missing $TRUTH60"; exit 2; }
test -f scripts/auriga/plot_shell_error_pairs.py || { echo "PREFLIGHT FAIL: missing shell-error script"; exit 2; }
test -f scripts/auriga/plot_pt_generations.py || { echo "PREFLIGHT FAIL: missing generations script"; exit 2; }
for d in "$BASE" "$LAMBDA10" "$INNERA" "$GRIDPRIOR"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "grids:     $GRIDS"
echo "truth60:   $TRUTH60"
echo "particles: $PARTICLES"
echo "base:      $BASE"
echo "lambda10:  $LAMBDA10"
echo "innerA:    $INNERA"
echo "gridprior: $GRIDPRIOR"

PARTICLE_ARG=()
if [ -f "$PARTICLES" ]; then PARTICLE_ARG=(--particles "$PARTICLES"); fi

echo '=== 1+2: shell-error-pairs + per-model (four-model set) ==='
"$PY" scripts/auriga/plot_shell_error_pairs.py \
  --grids "$GRIDS" --truth60 "$TRUTH60" "${PARTICLE_ARG[@]}" \
  --model base="$BASE" --model lambda10="$LAMBDA10" \
  --model innerA="$INNERA" --model gridprior="$GRIDPRIOR" \
  --output-dir figures/shell-error-pairs-lambda

echo '=== 3: pt-generations (four-model set) ==='
"$PY" scripts/auriga/plot_pt_generations.py \
  --grids "$GRIDS" \
  --model base="$BASE" --model lambda10="$LAMBDA10" \
  --model innerA="$INNERA" --model gridprior="$GRIDPRIOR" \
  --output-dir figures/pt-generations-lambda

echo '=== OUTPUTS ==='
find figures/shell-error-pairs-lambda figures/pt-generations-lambda -type f | sort
echo '=== LAMBDA-ROUND FIGURES DONE ==='
