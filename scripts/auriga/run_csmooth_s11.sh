#!/usr/bin/env bash
# Node: independent-seed reshuffle evaluation on the clean+smooth population.
# Step 1 creates halo12-clean-smooth-s11.h5 from the same raw source with
# prepare_data.py --seed 11 (verify-and-reuse, atomic rename), so the held-out
# region is decorrelated from every seed-0 file ordering.  Step 2 runs the
# fixed three-stage pipeline with this branch's options.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
OLD=$SRC/data/auriga/halo12_all_mass_clean_outer_clump_smooth.h5
SEED0=$SRC/data/auriga/halo12-clean-smooth.h5
NEW=$SRC/data/auriga/halo12-clean-smooth-s11.h5
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== S11 PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$OLD" || { echo "PREFLIGHT FAIL: missing $OLD"; exit 2; }
test -f "$SEED0" || { echo "PREFLIGHT FAIL: missing seed-0 reference $SEED0"; exit 2; }
test -f scripts/auriga/options.json || { echo "PREFLIGHT FAIL: missing options.json"; exit 2; }
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }

echo '=== STEP 1: prepare s11 file (verify-and-reuse) ==='
if test -f "$NEW"; then
  "$PY" scripts/auriga/check_s11_file.py "$NEW" "$SEED0"
  echo "reusing existing s11 file: $NEW"
else
  TMP="$NEW.tmp.$$"
  trap 'rm -f "$TMP"' EXIT
  "$PY" scripts/auriga/prepare_data.py --input "$OLD" --output "$TMP" --seed 11 --weighting mass
  "$PY" scripts/auriga/check_s11_file.py "$TMP" "$SEED0"
  if test -f "$NEW"; then
    "$PY" scripts/auriga/check_s11_file.py "$NEW" "$SEED0"
    rm -f "$TMP"
    echo "another run created $NEW concurrently; reusing it"
  else
    mv "$TMP" "$NEW"
    trap - EXIT
    echo "created s11 file: $NEW"
  fi
fi

echo '=== STEP 2: fixed three-stage pipeline on s11 data ==='
DATA="$NEW"
mkdir -p runs/orx
cp scripts/auriga/options.json runs/orx/options.json
run() { "$PY" -u scripts/fit_all.py --input "$DATA" --run-dir runs/orx "$@"; }
run --flow-training && run --flow-sampling && run --potential-training

echo '=== FINAL LOSSES ==='
for f in runs/orx/models/df/flow/flow-*_loss.json runs/orx/models/Phi/potential-*_loss.json; do
  echo "-- $f"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: (v[-1] if isinstance(v, list) else v) for k, v in d.items()})' "$f"
done
echo '=== S11 DONE ==='
