#!/usr/bin/env bash
# Meridional 2D potential slices and truth residuals for the three adjudicated
# w1024 clean+smooth Phis.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S11=$HOME/.orx/runs/ff328913-7bf8-4f38-9fc9-bb1c91b34867/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== S1 POTENTIAL 2D PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$TRUTH" || { echo "PREFLIGHT FAIL: missing $TRUTH"; exit 2; }
for d in "$BASE" "$S11" "$S1"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }

"$PY" scripts/auriga/plot_s1_potential_2d.py \
  --model base="$BASE" --model s11="$S11" --model S1="$S1" \
  --truth "$TRUTH" --output-dir figures/s1-potential-2d

echo '=== OUTPUTS ==='
find figures/s1-potential-2d -type f | sort
echo '=== S1 POTENTIAL 2D DONE ==='

