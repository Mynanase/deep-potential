#!/usr/bin/env bash
# Phase-2 continuation on the w1024 DF capacity winner (phase-1 audit run
# 5c4bf67d): flow-sampling + potential-training from the frozen
# halo12-cap-w1024 checkpoints, in a fresh run dir, without retraining the DF.
#
# Conventions (AGENTS.md): dp-jax conda env; JAX_PLATFORMS=cuda;
# XLA_PYTHON_CLIENT_PREALLOCATE=false; layered seeds stay in options.json
# (flow_sampling.seed=1, Phi.seed=2) -- no --seed override anywhere.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
CKPT=$SRC/runs/halo12-cap-w1024
DATA=$SRC/data/auriga/halo12-clean.h5
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== W1024 PHASE-2 PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$DATA" || { echo "PREFLIGHT FAIL: missing $DATA"; exit 2; }
test -f "$CKPT/options.json" || { echo "PREFLIGHT FAIL: missing $CKPT/options.json"; exit 2; }
ls "$CKPT"/models/df/flow/flow-[0-9]*_model.eqx >/dev/null 2>&1 \
  || { echo "PREFLIGHT FAIL: no conditional flow checkpoint in $CKPT"; exit 2; }
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py' \
  || { echo 'PREFLIGHT FAIL: dp-jax env lacks deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b, jax.devices()); exit(0 if b != "cpu" else 1)'
echo "INPUT DATA: $DATA"
"$PY" -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); a=dict(f.attrs); a.update(dict(f["eta"].attrs)); print("DATA attrs:", {k:a[k] for k in ("r_in","r_out","train_r_max_kpc","weighting") if k in a}, "n:", f["eta"].shape[0])' "$DATA"

# Fresh run dir: options from the checkpoint run, flow checkpoints linked
# read-only (never retrained, never modified).
mkdir -p runs/orx/models/df/flow
cp "$CKPT/options.json" runs/orx/options.json
"$PY" - <<'PYEOF'
import json
o = json.load(open("runs/orx/options.json"))
# grad_batch_size only chunks the derivative evaluation in
# flow_sampling.calculate_log_prob_and_derivatives (no RNG involvement);
# 64 -> 256 matches the boundary nodes and cuts wall time ~4x. The sample
# stream is set by sample_batch_size, which is unchanged.
o["flow_sampling"]["grad_batch_size"] = 256
json.dump(o, open("runs/orx/options.json", "w"), indent=2)
print("flow_sampling:", o["flow_sampling"])
print("Phi:", {k: o["Phi"][k] for k in ("seed", "n_epochs_noselfn", "batch_size")},
      "width", o["Phi"]["potential_nn_opts"]["width"])
PYEOF
for f in "$CKPT"/models/df/flow/*; do
  ln -sfn "$f" "runs/orx/models/df/flow/$(basename "$f")"
done
echo "linked flow checkpoint files: $(ls runs/orx/models/df/flow | wc -l)"

run() { "$PY" -u scripts/fit_all.py --input "$DATA" --run-dir runs/orx "$@"; }
run --flow-sampling
run --potential-training

echo '=== W1024 PHASE-2 FINAL LOSSES ==='
for f in runs/orx/models/Phi/potential-*_loss.json; do
  echo "-- $f"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: (v[-1] if isinstance(v, list) else v) for k, v in d.items()})' "$f"
done
echo '=== W1024 PHASE-2 DONE ==='
