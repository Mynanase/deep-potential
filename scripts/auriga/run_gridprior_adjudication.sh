#!/usr/bin/env bash
# Grid-prior adjudication: audited Gauss-flux enclosed-mass comparison of the
# grid-decoupled negative-density prior model (run 09faad30) against the three
# adjudicated w1024 clean+smooth Phis on identical grids, plus the 2D density
# slice figure (truth + base/S1/gridprior) in the published log-color style.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
BASE=$HOME/.orx/runs/6855f18e-857c-48b6-81e7-baffff01de1c/repo/runs/orx
S11=$HOME/.orx/runs/ff328913-7bf8-4f38-9fc9-bb1c91b34867/repo/runs/orx
S1=$HOME/.orx/runs/72240693-c2f7-49fa-9d26-ae6962368473/repo/runs/orx
GRIDPRIOR=$HOME/.orx/runs/09faad30-2d6f-482c-8d05-8194603c84f3/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== GRIDPRIOR ADJUDICATION PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$TRUTH" || { echo "PREFLIGHT FAIL: missing $TRUTH"; exit 2; }
for d in "$BASE" "$S11" "$S1" "$GRIDPRIOR"; do
  test -d "$d/models/Phi" || { echo "PREFLIGHT FAIL: missing $d/models/Phi"; exit 2; }
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'
echo "truth:      $TRUTH"
echo "base:       $BASE"
echo "s11:        $S11"
echo "S1:         $S1"
echo "gridprior:  $GRIDPRIOR"

echo '=== PART 1: compare_phi_truth enclosed-mass adjudication (4 models) ==='
"$PY" scripts/auriga/compare_phi_truth.py \
  --truth "$TRUTH" \
  --model base="$BASE" --model s11="$S11" --model S1="$S1" --model gridprior="$GRIDPRIOR" \
  --r-outer 70 --n-dirs 2048 --output-dir runs/truth_compare
echo '--- published anchors (run 0454cb6b, identical estimator): band medians [%] ---'
echo 'base:    2-10: 2.34  10-30: 6.64  30-50: 9.11  50-70: 7.46'
echo 's11:     2-10: 2.55  10-30: 6.89  30-50: 10.98 50-70: 10.46'
echo 'S1:      2-10: 1.84  10-30: 6.54  30-50: 9.32  50-70: 11.40'

echo '=== PART 2: 2D density slices (truth + base/S1/gridprior) ==='
"$PY" scripts/auriga/plot_s1_density_2d.py \
  --model base="$BASE" --model S1="$S1" --model gridprior="$GRIDPRIOR" \
  --truth "$TRUTH" --output-dir figures/gridprior-density-2d

echo '=== OUTPUTS ==='
find runs/truth_compare figures/gridprior-density-2d -type f | sort
echo '=== GRIDPRIOR ADJUDICATION DONE ==='

