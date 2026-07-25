#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${DATA_PATH:?Set DATA_PATH to the prepared Halo12 HDF5 file.}"

config="${CONFIG:-configs/df_halo12_ffjord_v23_mass.yaml}"
run_root="${RUN_ROOT:-runs/halo_12/df_ffjord_v23_mass}"
logger="${LOGGER:-csv}"
wandb_project="${WANDB_PROJECT:-deep-potential}"
seed_list="${SEEDS:-42,43,44,45}"
resume="${RESUME:-0}"

IFS=',' read -r -a seeds <<< "$seed_list"
if [[ "${#seeds[@]}" -lt 1 ]]; then
    echo "SEEDS must contain at least one integer." >&2
    exit 1
fi

mkdir -p "$run_root/logs"

for seed_text in "${seeds[@]}"; do
    seed="${seed_text//[[:space:]]/}"
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed in SEEDS: $seed_text" >&2
        exit 1
    fi

    split_seed=$((1000 + seed))
    run_dir="${run_root}/seed_${seed}"
    override="$(printf '{"seed": %d, "data": {"split_seed": %d}}' "$seed" "$split_seed")"
    log_file="${run_root}/logs/seed_${seed}.log"

    args=(
        python -m experiments.train_df
        --config "$config"
        --data "$DATA_PATH"
        --run-dir "$run_dir"
        --override "$override"
        --logger "$logger"
        --project "$wandb_project"
        --run-name "halo12-v23-seed-${seed}"
    )
    if [[ "$resume" == "1" ]]; then
        args+=(--resume)
    fi

    echo "[ensemble] starting seed=$seed, log=$log_file"
    "${args[@]}" 2>&1 | tee "$log_file"
    echo "[ensemble] completed seed=$seed"
done
