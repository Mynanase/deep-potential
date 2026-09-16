#!/usr/bin/env bash
# Minimal runnable Auriga Halo12 smoke baseline (512 particles, options-smoke.json).
#
# Self-contained on a Linux CUDA host: on first use it creates an isolated
# venv (Python 3.11 from PATH or ~/miniforge3/envs/dp-jax), installs the
# pinned scripts/auriga/requirements.txt plus the CUDA 12 JAX wheel, then
# runs the three fit_all.py stages, the physical-unit axis plot, and a
# summary block. Fixed contract for this node:
#   input    data/auriga/halo12-smoke.h5 (512 particles, shuffle seed 0)
#   options  scripts/auriga/options-smoke.json (df seed 0, sampling seed 1, Phi seed 2)
#   run dir  runs/halo12-smoke-baseline
set -euo pipefail

INPUT=data/auriga/halo12-smoke.h5
RUN_DIR=runs/halo12-smoke-baseline
STATE_ROOT="$HOME/halo12-orx"
VENV="$STATE_ROOT/venv"
export MPLCONFIGDIR="$STATE_ROOT/mpl"

if [ ! -x "$VENV/bin/python" ]; then
    mkdir -p "$STATE_ROOT" "$MPLCONFIGDIR"
    BASE_PY=""
    for cand in python3.11 "$HOME/miniforge3/envs/dp-jax/bin/python" python3.12; do
        if command -v "$cand" >/dev/null 2>&1; then BASE_PY="$cand"; break; fi
    done
    [ -n "$BASE_PY" ] || { echo "No Python 3.11+ interpreter found" >&2; exit 1; }
    echo "Creating venv at $VENV with $BASE_PY"
    "$BASE_PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r scripts/auriga/requirements.txt
    "$VENV/bin/python" -m pip install "jax[cuda12]==0.10.2"
fi
"$VENV/bin/python" -m pip check

mkdir -p "$RUN_DIR"
cp scripts/auriga/options-smoke.json "$RUN_DIR/options.json"

run_fit() {
    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$VENV/bin/python" -u scripts/fit_all.py --input "$INPUT" --run-dir "$RUN_DIR" "$@"
}

run_fit --flow-training
run_fit --flow-sampling
run_fit --potential-training

env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$VENV/bin/python" scripts/auriga/plot_potential.py \
    --input "$INPUT" --potential-dir "$RUN_DIR/models/Phi" --output-dir "$RUN_DIR/plots"

"$VENV/bin/python" scripts/auriga/smoke_summary.py --input "$INPUT" --run-dir "$RUN_DIR"
