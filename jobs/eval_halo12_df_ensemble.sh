#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${DATA_PATH:?Set DATA_PATH to the prepared Halo12 HDF5 file.}"

run_root="${RUN_ROOT:-runs/halo_12/df_ffjord_v23_mass}"
output_dir="${OUTPUT_DIR:-${run_root}/ensemble_evaluation}"
seed_list="${SEEDS:-42,43,44,45}"
n_samples="${N_SAMPLES_PER_MODEL:-262144}"
n_score_points="${N_SCORE_POINTS:-32768}"

IFS=',' read -r -a seeds <<< "$seed_list"
run_args=()
for seed_text in "${seeds[@]}"; do
    seed="${seed_text//[[:space:]]/}"
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed in SEEDS: $seed_text" >&2
        exit 1
    fi
    run_dir="${run_root}/seed_${seed}"
    if [[ ! -d "$run_dir" ]]; then
        echo "Missing trained run directory: $run_dir" >&2
        exit 1
    fi
    run_args+=(--run-dir "$run_dir")
done

if [[ "${#seeds[@]}" -lt 2 ]]; then
    echo "Ensemble evaluation requires at least two seeds." >&2
    exit 1
fi

python -m experiments.eval_auriga_df \
    --data "$DATA_PATH" \
    "${run_args[@]}" \
    --output-dir "$output_dir" \
    --n-samples-per-model "$n_samples" \
    --n-score-points "$n_score_points"
