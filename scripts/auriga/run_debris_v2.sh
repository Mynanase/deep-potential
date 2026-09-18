#!/usr/bin/env bash
# Direction A v2: velocity-gate debris detection.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
export JAX_PLATFORMS=cpu

echo '=== DEBRIS V2 PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
for f in "$SRC/data/auriga/clump_pid_registry.npz" "$SRC/data/auriga/halo12.h5" "$SRC/data/auriga/halo12-clean.h5"; do
  test -f "$f" || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
test -f "$SRC/data/auriga/clump_debris_candidates_v2.npz" && { echo "PREFLIGHT FAIL: output exists"; exit 2; }

mkdir -p runs data/auriga
ln -sfn "$SRC/data/auriga/halo12.h5" data/auriga/halo12.h5
ln -sfn "$SRC/data/auriga/halo12-clean.h5" data/auriga/halo12-clean.h5
ln -sfn "$SRC/data/auriga/clump_pid_registry.npz" data/auriga/clump_pid_registry.npz
DEBRIS_OUT=$SRC/data/auriga/clump_debris_candidates_v2.npz "$PY" -u scripts/auriga/df_debris_v2.py
echo '=== DEBRIS V2 DONE ==='
