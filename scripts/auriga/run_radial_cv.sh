#!/usr/bin/env bash
# Radial conditional-velocity review (plan:
# artifacts/df-phase1-audit/radial-conditional-velocity-audit-plan.md, section 4.1).
#
# Rebuilds the phase-1 evaluation protocol on this host, then compares the
# held-out strict-common stars with K=4 fresh conditional draws at their own
# positions for the three capacity checkpoints (w128 / w512 / w1024, flow-19/20/21).
# The metrics script prints the full numeric summary; figures are produced
# separately from the persisted arrays (df_phase1_radial_cv_figures.py).
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
AUDIT="${RCV_AUDIT_DIR:-$SRC/runs/halo12-phase1-df-audit-20260916}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
# The host is shared with training runs, so unless the caller pins a device,
# take the card with the most free memory; the flows here need under 2 GB.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
fi
export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== RADIAL CONDITIONAL-VELOCITY PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py, numpy, pandas, scipy' \
  || { echo 'PREFLIGHT FAIL: dp-jax env lacks deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b, jax.devices()); exit(0 if b != "cpu" else 1)'
for f in "$SRC/data/auriga/halo12.h5" "$SRC/data/auriga/halo12-clean.h5"; do
  test -f "$f" || { echo "PREFLIGHT FAIL: missing $f"; exit 2; }
done
for d in halo12-baseline halo12-cap-w512 halo12-cap-w1024; do
  test -d "$SRC/runs/$d/models/df/flow" \
    || { echo "PREFLIGHT FAIL: no flow checkpoints under $SRC/runs/$d"; exit 2; }
done

# the K=1 cross-check needs the persisted phase-1 audit products; the review
# itself does not, so a missing audit directory is reported and only degrades
# the cross-check
if test -f "$AUDIT/model_w1024.npz"; then
  echo "k1 cross-check source: $AUDIT"
else
  echo "WARNING: $AUDIT/model_w1024.npz missing -> k1 cross-check will be skipped"
fi

# the evaluation protocol is regenerated here from the two inputs, so this run
# does not depend on a previous run's snapshot directory
mkdir -p data/auriga runs
ln -sfn "$SRC/data/auriga/halo12.h5" data/auriga/halo12.h5
ln -sfn "$SRC/data/auriga/halo12-clean.h5" data/auriga/halo12-clean.h5
for d in halo12-baseline halo12-cap-w512 halo12-cap-w1024; do
  ln -sfn "$SRC/runs/$d" "runs/$d"
done

echo '=== STEP 1/2: evaluation protocol (strict common held-out set) ==='
"$PY" scripts/auriga/df_phase1_data_audit.py

echo '=== STEP 2/2: radial conditional-velocity metrics (GPU) ==='
RCV_AUDIT_DIR="$AUDIT" "$PY" -u scripts/auriga/df_phase1_radial_cv_metrics.py

echo '=== RADIAL CONDITIONAL-VELOCITY DONE ==='
ls -1 runs/halo12-radial-cv-20260917
