#!/usr/bin/env bash
# Velocity-only debris removal exploration on raw halo12 (this branch:
# dfc069a0). No spatial clustering, no registry-based selection; compares
# template gate / iterative gate / per-bin GMM / 1D vT tail cuts against the
# frozen v1+v2 candidate sets and the cascade reference (ba7750a4).
# Data-level diagnostic, no training; CPU-only on the gpu host.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== VELOCITY REMOVAL PREFLIGHT ==='
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

echo '=== VELOCITY REMOVAL RUN ==='
"$PY" -u scripts/auriga/df_velocity_removal_explore.py
echo '=== VELOCITY REMOVAL DONE ==='
