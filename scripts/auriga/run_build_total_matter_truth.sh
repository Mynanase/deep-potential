#!/usr/bin/env bash
# Build the particle-level total-matter truth asset in the model (star) frame.
set -eo pipefail
PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"
echo '=== BUILD TOTAL-MATTER TRUTH PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -d /localdisk/kosmos/my-deep-potential/data/halo12_raw/snapdir_127 || { echo "PREFLIGHT FAIL: missing snapdir"; exit 2; }
test -f /localdisk/kosmos/my-deep-potential/data/halo_12_stars.hdf5 || { echo "PREFLIGHT FAIL: missing stars"; exit 2; }
test -f /localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5 || { echo "PREFLIGHT FAIL: missing truth"; exit 2; }
"$PY" -c 'import h5py, numpy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" scripts/auriga/build_total_matter_truth.py
echo '=== BUILD TOTAL-MATTER TRUTH DONE ==='

