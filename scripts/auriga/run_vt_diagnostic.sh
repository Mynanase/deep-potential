#!/usr/bin/env bash
# Standalone vT-direction diagnostic (45-75 kpc) on the phase-1 audit
# artifacts: capacity vs training-data decomposition for the vT W1 gap.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
AUDIT=/home/qiutao/.orx/runs/5c4bf67d-6a74-4558-881d-571635c5f025/repo/runs/halo12-phase1-df-audit-20260916
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== VT DIAGNOSTIC PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
for f in "$AUDIT/eval_protocol.npz" "$AUDIT/model_w128.npz" "$AUDIT/model_w512.npz" \
         "$AUDIT/model_w1024.npz" "$AUDIT/per_bin_metrics.csv" \
         "$AUDIT/w1_noise_floor.csv" "$AUDIT/direction_metrics.csv" \
         "$SRC/data/auriga/halo12.h5" "$SRC/data/auriga/halo12-clean.h5"; do
  test -f "$f" || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
for d in halo12-baseline halo12-cap-w512 halo12-cap-w1024; do
  ls "$SRC/runs/$d/models/df/flow/"flow-[0-9]*_model.eqx >/dev/null 2>&1 \
    || { echo "PREFLIGHT FAIL: no flow checkpoint under $SRC/runs/$d"; exit 2; }
done
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py, numpy, pandas, scipy, matplotlib' \
  || { echo 'PREFLIGHT FAIL: dp-jax env lacks deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b, jax.devices()); exit(0 if b != "cpu" else 1)'

mkdir -p runs data/auriga
ln -sfn "$AUDIT" runs/halo12-phase1-df-audit-20260916
ln -sfn "$SRC/data/auriga/halo12.h5" data/auriga/halo12.h5
ln -sfn "$SRC/data/auriga/halo12-clean.h5" data/auriga/halo12-clean.h5
for d in halo12-baseline halo12-cap-w512 halo12-cap-w1024; do
  ln -sfn "$SRC/runs/$d" "runs/$d"
done
echo "linked audit artifacts: $AUDIT"

"$PY" -u scripts/auriga/df_phase1_vt_diagnostic.py
echo '=== VT DIAGNOSTIC DONE ==='
