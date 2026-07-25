#!/usr/bin/env bash

# Scheduler-free end-to-end Halo12 DF pipeline.

set -euo pipefail

jobs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${INPUT_PATH:?Set INPUT_PATH to halo_12_stars.hdf5.}"

export OUTPUT_PATH="${OUTPUT_PATH:-data/auriga/halo12_all_mass.h5}"
export DATA_PATH="${DATA_PATH:-$OUTPUT_PATH}"
export RUN_ROOT="${RUN_ROOT:-runs/halo_12/df_ffjord_v23_mass}"
export SEEDS="${SEEDS:-42,43,44,45}"

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
    bash "${jobs_dir}/prepare_halo12_df.sh"
fi
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    bash "${jobs_dir}/train_halo12_df_ensemble.sh"
fi
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    bash "${jobs_dir}/eval_halo12_df_ensemble.sh"
fi

echo "[pipeline] Halo12 DF pipeline completed."
