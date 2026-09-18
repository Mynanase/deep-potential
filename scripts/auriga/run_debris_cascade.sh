#!/usr/bin/env bash
# Debris cascade comparison (55-65 kpc): no removal vs two removals.
# Data-level diagnostic, no training: raw halo12 -> union-removed ->
# union + v2 velocity-gate debris removed (frozen candidate snapshot).
# CPU-only; runs on the gpu host like every orx run for this project.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== DEBRIS CASCADE PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
for f in "$SRC/data/halo_12_stars.hdf5" \
         "$SRC/data/auriga/halo12_all_mass_clean_outer_clump_smooth.h5" \
         "$SRC/data/auriga/clump_debris_candidates.npz" \
         "$SRC/data/auriga/clump_debris_candidates_v2.npz"; do
  test -f "$f" || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
"$PY" -c 'import h5py, numpy, scipy, matplotlib' || { echo "PREFLIGHT FAIL: deps"; exit 2; }

# The clump PID registry is not committed (large binary); mirror the symlink
# layout used by earlier runs (e.g. e94c3979) into the run snapshot.
mkdir -p data/auriga
test -e data/auriga/clump_pid_registry.npz \
  || ln -s "$SRC/data/auriga/clump_pid_registry.npz" data/auriga/clump_pid_registry.npz

echo '=== DEBRIS CASCADE RUN ==='
"$PY" -u scripts/auriga/df_debris_cascade.py
echo '=== DEBRIS CASCADE DONE ==='
