#!/usr/bin/env bash
# Read-only density-oscillation spectra for the negative-density study:
# five frozen w1024 csmooth Phis (base / S1 / gridprior / innerA / lambda10)
# vs the particle-truth rho3d grid product. No retraining.
set -eo pipefail
PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
RUNS=/home/qiutao/.orx/runs
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== DENSITY-SPECTRUM PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
"$PY" -c 'import jax, h5py, scipy, matplotlib; print("deps ok; jax", jax.__version__)'
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'

declare -A M
M[base]=$RUNS/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
M[S1]=$RUNS/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
M[gridprior]=$RUNS/09faad30-2d6f-482c-8d05-8194603c84f3/repo/runs/orx
M[innerA]=$RUNS/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo/runs/orx
M[lambda10]=$RUNS/e1cf8f74-775d-44b1-b567-24dd88e48cd0/repo/runs/orx
for k in base S1 gridprior innerA lambda10; do
  test -f "${M[$k]}/models/Phi/potential-10_model.eqx" || { echo "PREFLIGHT FAIL: missing model for $k at ${M[$k]}"; exit 2; }
  echo "$k: ${M[$k]}"
done

"$PY" -u scripts/auriga/plot_density_spectrum.py \
  --grids "$GRIDS" \
  --model base="${M[base]}" --model S1="${M[S1]}" \
  --model gridprior="${M[gridprior]}" --model innerA="${M[innerA]}" \
  --model lambda10="${M[lambda10]}"
echo '=== DENSITY-SPECTRUM DONE ==='
