#!/usr/bin/env bash
# Phi-direction mass profiles per shell in fixed theta bands (child of the
# paired-shell-error line): per selected shell and theta band (equatorial
# 45-135 deg primary; north/south mid-latitude; |cos theta|>0.85 excluded),
# mass per phi bin - truth = exact particle sum, models = Laplacian-density
# quadrature; signed DeltaM/M_true per bin with Sigma / Sigma|.| annotation.
# Same five frozen models, no retraining.
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
# Auto-pick the first idle card (used < 1000 MiB) unless the caller pinned
# CUDA_VISIBLE_DEVICES; mirrors the gpu-server-env queueing policy.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits \
    | awk -F, '$2 < 1000 && !found {print $1; found=1}')
  export CUDA_VISIBLE_DEVICES
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== PHI-PROFILES PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
test -f "$TRUTH60" || { echo "PREFLIGHT FAIL: missing $TRUTH60"; exit 2; }
test -f "$PARTICLES" || { echo "PREFLIGHT FAIL: missing $PARTICLES"; exit 2; }
for d in "$BASE" "$S1" "$GRIDPRIOR" "$INNERA" "$INNERB"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "grids:      $GRIDS"
echo "truth60:    $TRUTH60"
echo "particles:  $PARTICLES"

echo "=== PHI-PROFILES RUN ==="
"$PY" scripts/auriga/plot_phi_profiles.py \
  --grids "$GRIDS" --truth60 "$TRUTH60" --particles "$PARTICLES" \
  --model base="$BASE" --model S1="$S1" --model gridprior="$GRIDPRIOR" \
  --model innerA="$INNERA" --model innerB="$INNERB" \
  --shell-r 1.5 --shell-r 4.4 --shell-r 9.6 --shell-r 20 --shell-r 40 --shell-r 62 \
  --midlat-r 4.4 --midlat-r 9.6 --midlat-r 62 \
  --output-dir figures/phi-profiles

echo '=== OUTPUTS ==='
find figures/phi-profiles -type f | sort
echo '=== PHI-PROFILES DONE ==='
