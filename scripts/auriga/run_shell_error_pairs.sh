#!/usr/bin/env bash
# Paired shell errors vs particle truth (child of the inner-band
# pt-adjudication node 4b21457d): per 60-shell shell, signed e = M_model -
# M_true and absolute E = int |rho_model - rho3d| dV, kappa = E/|e|, with
# exact Poisson floors. Same five frozen models, no retraining:
#   base      6855f18e  (production co-winner)
#   S1        72240693  (gridprior parent, stratified sampling)
#   gridprior 09faad30  (volume grid)
#   innerA    2b32eb04  (radius-balanced grid - adjudication winner)
#   innerB    7016399a  (inner-weighted volume grid)
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
TRUTH60=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
PARTICLES=/localdisk/kosmos/my-deep-potential/data/auriga/halo12_total_matter_particles_starframe.h5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
GRIDPRIOR=$HOME/.orx/runs/09faad30-2d6f-482c-8d05-8194603c84f3/repo/runs/orx
INNERA=$HOME/.orx/runs/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo/runs/orx
INNERB=$HOME/.orx/runs/7016399a-4126-4943-8efb-20ec0e9cde44/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== SHELL-ERROR-PAIRS PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
test -f "$TRUTH60" || { echo "PREFLIGHT FAIL: missing $TRUTH60"; exit 2; }
for d in "$BASE" "$S1" "$GRIDPRIOR" "$INNERA" "$INNERB"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "grids:      $GRIDS"
echo "truth60:    $TRUTH60"
echo "particles:  $PARTICLES"
echo "base:       $BASE"
echo "S1:         $S1"
echo "gridprior:  $GRIDPRIOR"
echo "innerA:     $INNERA"
echo "innerB:     $INNERB"

PARTICLE_ARG=()
if [ -f "$PARTICLES" ]; then PARTICLE_ARG=(--particles "$PARTICLES"); fi

"$PY" scripts/auriga/plot_shell_error_pairs.py \
  --grids "$GRIDS" --truth60 "$TRUTH60" "${PARTICLE_ARG[@]}" \
  --model base="$BASE" --model S1="$S1" --model gridprior="$GRIDPRIOR" \
  --model innerA="$INNERA" --model innerB="$INNERB" \
  --output-dir figures/shell-error-pairs

echo '=== OUTPUTS ==='
find figures/shell-error-pairs -type f | sort
echo '=== SHELL-ERROR-PAIRS DONE ==='
