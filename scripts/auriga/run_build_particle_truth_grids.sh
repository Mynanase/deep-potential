#!/usr/bin/env bash
# Build the particle-truth grid data product (density + potential, one h5).
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
ASSET=/localdisk/kosmos/my-deep-potential/data/auriga/halo12_total_matter_particles_starframe.h5
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== BUILD PARTICLE-TRUTH GRIDS PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$ASSET" || { echo "PREFLIGHT FAIL: missing $ASSET"; exit 2; }
"$PY" -c 'import jax, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'

TRUTH_ARG=()
if [ -f "$TRUTH" ]; then TRUTH_ARG=(--truth "$TRUTH"); fi
"$PY" scripts/auriga/build_particle_truth_grids.py \
  --asset "$ASSET" --output data/auriga/halo12_particle_truth_grids.h5 \
  "${TRUTH_ARG[@]}"

echo '=== OUTPUT ==='
ls -la data/auriga/halo12_particle_truth_grids.h5
sha256sum data/auriga/halo12_particle_truth_grids.h5
echo '=== BUILD PARTICLE-TRUTH GRIDS DONE ==='
