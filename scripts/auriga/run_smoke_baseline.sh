#!/usr/bin/env bash
# Minimal runnable Auriga Halo12 smoke baseline (512 particles, options-smoke.json).
#
# Runs on the gpu ssh host with the preexisting dp-jax conda environment;
# no dependency installation. It prints an environment fingerprint, runs the
# three fit_all.py stages, the physical-unit axis plot, and a summary block.
# Fixed contract for this node:
#   input    data/auriga/halo12-smoke.h5 (512 particles, shuffle seed 0)
#   options  scripts/auriga/options-smoke.json (df seed 0, sampling seed 1, Phi seed 2)
#   run dir  runs/halo12-smoke-baseline
set -euo pipefail

INPUT=data/auriga/halo12-smoke.h5
RUN_DIR=runs/halo12-smoke-baseline
PYTHON=/home/qiutao/miniforge3/envs/dp-jax/bin/python
[ -x "$PYTHON" ] || { echo "dp-jax conda python not found at $PYTHON" >&2; exit 1; }

env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -c 'import jax, importlib.metadata as m; print("python env: dp-jax conda"); print("jax", jax.__version__, jax.devices()); print("equinox", m.version("equinox"), "optax", m.version("optax"), "diffrax", m.version("diffrax"), "flowjax", m.version("flowjax"), "h5py", m.version("h5py"), "numpy", m.version("numpy"))'

mkdir -p "$RUN_DIR"
cp scripts/auriga/options-smoke.json "$RUN_DIR/options.json"

run_fit() {
    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PYTHON" -u scripts/fit_all.py --input "$INPUT" --run-dir "$RUN_DIR" "$@"
}

run_fit --flow-training
run_fit --flow-sampling
run_fit --potential-training

env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PYTHON" scripts/auriga/plot_potential.py \
    --input "$INPUT" --potential-dir "$RUN_DIR/models/Phi" --output-dir "$RUN_DIR/plots"

"$PYTHON" scripts/auriga/smoke_summary.py --input "$INPUT" --run-dir "$RUN_DIR"
