#!/usr/bin/env bash
# Direction B: w1024 on the union-removed population (clump + smooth sets).
# Step 1 converts the old-format (raw kpc / km/s) union-removed export to the
# new training format via prepare_data.py; step 2 is the fixed three-stage
# pipeline (config committed on this branch).
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
OLD=$SRC/data/auriga/halo12_all_mass_clean_outer_clump_smooth.h5
NEW=$SRC/data/auriga/halo12-clean-smooth.h5
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== W1024-CSMOOTH PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$OLD" || { echo "PREFLIGHT FAIL: missing $OLD"; exit 2; }
test -f "$NEW" && { echo "PREFLIGHT FAIL: $NEW already exists (prepare refuses overwrite); move it aside or reuse it"; exit 2; }
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import h5py; f=h5py.File("'"$OLD"'","r"); a=dict(f.attrs); print("source attrs:", {k:a[k] for k in ("cleaning_registry_set","cleaning_removed_count","length_unit","velocity_unit") if k in a}, "n:", f["eta"].shape[0])'

echo '=== STEP 1: convert to new format (seed 0, mass weighting) ==='
"$PY" scripts/auriga/prepare_data.py --input "$OLD" --output "$NEW" --seed 0 --weighting mass
test -f "$NEW" || { echo "FAIL: conversion produced no output"; exit 2; }
"$PY" -c 'import h5py; f=h5py.File("'"$NEW"'","r"); a=dict(f.attrs); a.update(dict(f["eta"].attrs)); print("NEW attrs:", {k:a[k] for k in ("r_in","r_out","train_r_max_kpc","weighting","shuffle_seed") if k in a}, "n:", f["eta"].shape[0])'

echo '=== STEP 2: fixed three-stage w1024 pipeline ==='
DATA_ROOT=$SRC/data/auriga
DATA=$(cat scripts/auriga/orx_data_path.txt 2>/dev/null || echo "$DATA_ROOT/halo12-clean-smooth.h5")
case "$DATA" in /*) ;; *) DATA="$DATA_ROOT/$DATA";; esac
test -f "$DATA" || { echo "PREFLIGHT FAIL: missing $DATA"; exit 2; }
mkdir -p runs/orx
cp scripts/auriga/options.json runs/orx/options.json
run() { "$PY" -u scripts/fit_all.py --input "$DATA" --run-dir runs/orx "$@"; }
run --flow-training && run --flow-sampling && run --potential-training

echo '=== FINAL LOSSES ==='
for f in runs/orx/models/df/flow/flow-*_loss.json runs/orx/models/Phi/potential-*_loss.json; do
  echo "-- $f"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: (v[-1] if isinstance(v, list) else v) for k, v in d.items()})' "$f"
done
echo '=== W1024-CSMOOTH DONE ==='
