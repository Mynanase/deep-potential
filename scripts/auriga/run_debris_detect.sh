#!/usr/bin/env bash
# Direction A: residual-debris detection on the clean population.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export JAX_PLATFORMS=cpu

echo '=== DEBRIS DETECT PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$SRC/data/auriga/clump_pid_registry.npz" || { echo "PREFLIGHT FAIL: missing registry"; exit 2; }
test -f "$SRC/data/auriga/halo12.h5" || { echo "PREFLIGHT FAIL: missing halo12.h5"; exit 2; }
test -f "$SRC/data/auriga/halo12-clean.h5" || { echo "PREFLIGHT FAIL: missing halo12-clean.h5"; exit 2; }
test -f "$SRC/data/auriga/clump_debris_candidates.npz" && { echo "PREFLIGHT FAIL: output registry already exists"; exit 2; }

mkdir -p runs data/auriga
ln -sfn "$SRC/data/auriga/halo12.h5" data/auriga/halo12.h5
ln -sfn "$SRC/data/auriga/halo12-clean.h5" data/auriga/halo12-clean.h5
ln -sfn "$SRC/data/auriga/clump_pid_registry.npz" data/auriga/clump_pid_registry.npz
DEBRIS_OUT=$SRC/data/auriga/clump_debris_candidates.npz "$PY" -u scripts/auriga/df_debris_detect.py
echo '=== DEBRIS DETECT DONE ==='
