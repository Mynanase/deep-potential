#!/usr/bin/env bash

# Train the static potential Phi on Halo12 with a frozen DF (CBE residual A).
#
# Required env:
#   CONFIG      – Phi config path (e.g. configs/phi_halo12_static_v1.yaml)
#   RUN_DIR     – output run directory (e.g. runs/halo_12/phi_static_v1_seed42)
#   DF_RUN_DIR  – completed DF run directory (frozen; must contain
#                 config.yaml, normalizer.npz, ckpt/)
# Optional env:
#   DATA_PATH   – input HDF5 (defaults to data/auriga/halo12_all_mass.h5)
#   LOGGER      – wandb|csv|tensorboard|wandb+tb (default wandb)
#   WANDB_PROJECT, RUN_NAME, OVERRIDE, RESUME, INIT_PARAMS

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${CONFIG:?Set CONFIG to a Phi config path.}"
: "${RUN_DIR:?Set RUN_DIR to the output run directory.}"
: "${DF_RUN_DIR:?Set DF_RUN_DIR to a completed DF run directory (frozen).}"

data_path="${DATA_PATH:-data/auriga/halo12_all_mass.h5}"
run_name="${RUN_NAME:-$(basename "$RUN_DIR")}"
logger="${LOGGER:-wandb}"
wandb_project="${WANDB_PROJECT:-deep-potential}"

args=(
    python -m experiments.train_phi
    --config "$CONFIG"
    --data "$data_path"
    --df-run-dir "$DF_RUN_DIR"
    --run-dir "$RUN_DIR"
    --logger "$logger"
    --project "$wandb_project"
    --run-name "$run_name"
)

if [[ -n "${OVERRIDE:-}" ]]; then
    args+=(--override "$OVERRIDE")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    args+=(--resume)
fi
if [[ -n "${INIT_PARAMS:-}" ]]; then
    args+=(--init-params "$INIT_PARAMS")
fi

"${args[@]}"