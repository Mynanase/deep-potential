#!/usr/bin/env bash
# fig_clump_conditional_vT with the clump-removed truth reference.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
REG=$SRC/data/auriga/clump_pid_registry.npz
W1024FULL=/home/qiutao/.orx/runs/b26b7505-ad14-4d8e-a417-e9faf956219f/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== CLUMP FIG REFERENCE PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$REG" || { echo "PREFLIGHT FAIL: missing $REG"; exit 2; }
test -f "$W1024FULL/models/df/flow/flow-21_model.eqx" || { echo "PREFLIGHT FAIL: missing w1024full"; exit 2; }
for f in "$SRC/data/auriga/halo12.h5" "$SRC/data/auriga/halo12-clean.h5"; do
  test -f "$f" || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
for d in halo12-baseline halo12-cap-w1024; do
  ls "$SRC/runs/$d/models/df/flow/"flow-[0-9]*_model.eqx >/dev/null 2>&1 \
    || { echo "PREFLIGHT FAIL: no flow checkpoint under $SRC/runs/$d"; exit 2; }
done
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py, numpy, scipy, matplotlib' \
  || { echo 'PREFLIGHT FAIL: dp-jax env lacks deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b, jax.devices()); exit(0 if b != "cpu" else 1)'

mkdir -p runs data/auriga
ln -sfn "$SRC/data/auriga/halo12.h5" data/auriga/halo12.h5
ln -sfn "$SRC/data/auriga/halo12-clean.h5" data/auriga/halo12-clean.h5
ln -sfn "$REG" data/auriga/clump_pid_registry.npz
ln -sfn "$W1024FULL" runs/w1024full
ln -sfn "$SRC/runs/halo12-baseline" runs/halo12-baseline
ln -sfn "$SRC/runs/halo12-cap-w1024" runs/halo12-cap-w1024

"$PY" -u scripts/auriga/df_clump_fig_reference.py
echo '=== CLUMP FIG REFERENCE DONE ==='
