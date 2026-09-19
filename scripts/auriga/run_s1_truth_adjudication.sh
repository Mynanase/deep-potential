#!/usr/bin/env bash
# Enclosed-mass truth adjudication for the S1 line on the frozen clean+smooth
# route: three w1024 Phis (base seed-0, s11 independent reshuffle, S1 stratified
# quotas) vs the Auriga Halo12 total-matter truth, using the audited Gauss-flux
# quadrature reused by compare_phi_truth.py (2048 Sobol dirs, x64, annulus dM
# anchored at the first truth edge >= 1 kpc).
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S11=$HOME/.orx/runs/ff328913-7bf8-4f38-9fc9-bb1c91b34867/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== S1 TRUTH ADJUDICATION PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$TRUTH" || { echo "PREFLIGHT FAIL: missing $TRUTH"; exit 2; }
for d in "$BASE" "$S11" "$S1"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, h5py, numpy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
echo "truth: $TRUTH"
echo "base:  $BASE"
echo "s11:   $S11"
echo "S1:    $S1"

"$PY" scripts/auriga/compare_phi_truth.py \
  --truth "$TRUTH" \
  --model base="$BASE" --model s11="$S11" --model S1="$S1" \
  --r-outer 70 --n-dirs 2048 --output-dir runs/truth_compare
echo '=== S1 TRUTH ADJUDICATION DONE ==='
