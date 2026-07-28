#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${CONFIG:?Set CONFIG to a Halo12 DF config path.}"
: "${RUN_DIR:?Set RUN_DIR to the output run directory.}"

data_path="${DATA_PATH:-data/halo_12_train.h5}"
run_name="${RUN_NAME:-$(basename "$RUN_DIR")}"
logger="${LOGGER:-wandb}"
wandb_project="${WANDB_PROJECT:-deep-potential}"

args=(
    python -m experiments.train_df
    --config "$CONFIG"
    --data "$data_path"
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

"${args[@]}"
