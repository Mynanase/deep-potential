#!/usr/bin/env bash
# Node A: CPU-only diagnosis of the seed-0 evaluation-protocol bias.
# Measures order correlation across the three seed-0 files, the old strict
# pool's radial composition, the pid-hash independent-reshuffle protocol,
# within-bin reweighting recovery, and truth half-split W1 floors.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
DATA_ROOT=${DPJAX_DATA_ROOT:-/localdisk/kosmos/my-deep-potential/data/auriga}
export DPJAX_DATA_ROOT="$DATA_ROOT"
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== EVAL-PROTOCOL-BIAS PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
for f in halo12.h5 halo12-clean.h5 halo12-clean-smooth.h5; do
  test -f "$DATA_ROOT/$f" || { echo "PREFLIGHT FAIL: missing $DATA_ROOT/$f"; exit 2; }
done
"$PY" -c 'import numpy, h5py, pandas, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }

echo '=== SELFTEST ==='
"$PY" scripts/auriga/df_eval_protocol_bias.py --selftest

echo '=== MAIN (CPU only, no jax) ==='
"$PY" -u scripts/auriga/df_eval_protocol_bias.py
echo '=== EVAL-PROTOCOL-BIAS DONE ==='
