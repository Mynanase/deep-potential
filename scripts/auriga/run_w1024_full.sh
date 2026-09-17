#!/usr/bin/env bash
# w1024 on the FULL halo12 population (including the outer clump stream):
# the project's fixed three-stage pipeline (verbatim), then a substructure
# evaluation appendix on the freshly trained checkpoint.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
DATA_ROOT=$SRC/data/auriga
AUDIT=/home/qiutao/.orx/runs/5c4bf67d-6a74-4558-881d-571635c5f025/repo/runs/halo12-phase1-df-audit-20260916
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== W1024-FULL PREFLIGHT ==='
DATA=$(cat scripts/auriga/orx_data_path.txt 2>/dev/null || echo "$DATA_ROOT/halo12.h5")
case "$DATA" in /*) ;; *) DATA="$DATA_ROOT/$DATA";; esac
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$DATA" || { echo "PREFLIGHT FAIL: missing $DATA"; exit 2; }
test -f "$AUDIT/eval_protocol.npz" || { echo "PREFLIGHT FAIL: missing $AUDIT/eval_protocol.npz"; exit 2; }
for f in "$SRC/data/auriga/halo12.h5" "$SRC/data/auriga/halo12-clean.h5"; do
  test -f "$f" || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
for d in halo12-baseline halo12-cap-w1024; do
  ls "$SRC/runs/$d/models/df/flow/"flow-[0-9]*_model.eqx >/dev/null 2>&1 \
    || { echo "PREFLIGHT FAIL: no flow checkpoint under $SRC/runs/$d"; exit 2; }
done
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py, numpy, pandas, scipy, matplotlib' \
  || { echo 'PREFLIGHT FAIL: dp-jax env lacks deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b, jax.devices()); exit(0 if b != "cpu" else 1)'
echo "INPUT DATA: $DATA"
"$PY" -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); a=dict(f.attrs); a.update(dict(f["eta"].attrs)); print("DATA attrs:", {k:a[k] for k in ("r_in","r_out","train_r_max_kpc","weighting") if k in a}, "n:", f["eta"].shape[0])' "$DATA"

mkdir -p runs data/auriga
ln -sfn "$SRC/data/auriga/halo12.h5" data/auriga/halo12.h5
ln -sfn "$SRC/data/auriga/halo12-clean.h5" data/auriga/halo12-clean.h5
ln -sfn "$AUDIT" runs/halo12-phase1-df-audit-20260916
ln -sfn "$SRC/runs/halo12-baseline" runs/halo12-baseline
ln -sfn "$SRC/runs/halo12-cap-w1024" runs/halo12-cap-w1024

# === fixed three-stage pipeline (identical to the project run command) ===
mkdir -p runs/orx
cp scripts/auriga/options.json runs/orx/options.json
run() { "$PY" -u scripts/fit_all.py --input "$DATA" --run-dir runs/orx "$@"; }
run --flow-training && run --flow-sampling && run --potential-training

echo '=== FINAL LOSSES ==='
for f in runs/orx/models/df/flow/flow-*_loss.json runs/orx/models/Phi/potential-*_loss.json; do
  echo "-- $f"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: (v[-1] if isinstance(v, list) else v) for k, v in d.items()})' "$f"
done

echo '=== SUBSTRUCTURE EVAL ==='
"$PY" -u scripts/auriga/df_substructure_eval.py
echo '=== W1024-FULL DONE ==='
