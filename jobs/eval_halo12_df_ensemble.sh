#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${DATA_PATH:?Set DATA_PATH to the prepared Halo12 HDF5 file.}"

run_root="${RUN_ROOT:-runs/halo_12/df_ffjord_v23_mass}"
output_dir="${OUTPUT_DIR:-${run_root}/ensemble_evaluation}"
seed_list="${SEEDS:-42,43,44,45}"
n_samples="${N_SAMPLES_PER_MODEL:-262144}"
n_score_points="${N_SCORE_POINTS:-32768}"
n_theta_bins="${N_THETA_BINS:-6}"
n_phi_bins="${N_PHI_BINS:-8}"
n_velocity_bins="${N_VELOCITY_BINS:-64}"
spatial_r_bins="${SPATIAL_R_BINS:-48}"
spatial_z_bins="${SPATIAL_Z_BINS:-48}"
spatial_r_max="${SPATIAL_R_MAX:-75}"
spatial_z_max="${SPATIAL_Z_MAX:-75}"
spatial_min_cell_count="${SPATIAL_MIN_CELL_COUNT:-5}"
plot_dpi="${PLOT_DPI:-150}"

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

if [[ "${#seeds[@]}" -lt 1 ]]; then
    echo "DF evaluation requires at least one seed." >&2
    exit 1
fi

python -m experiments.eval_auriga_df \
    --data "$DATA_PATH" \
    "${run_args[@]}" \
    --output-dir "$output_dir" \
    --n-samples-per-model "$n_samples" \
    --n-score-points "$n_score_points" \
    --n-theta-bins "$n_theta_bins" \
    --n-phi-bins "$n_phi_bins" \
    --n-velocity-bins "$n_velocity_bins" \
    --spatial-r-bins "$spatial_r_bins" \
    --spatial-z-bins "$spatial_z_bins" \
    --spatial-r-max "$spatial_r_max" \
    --spatial-z-max "$spatial_z_max" \
    --spatial-min-cell-count "$spatial_min_cell_count"

if [[ "${SKIP_PLOT:-0}" != "1" ]]; then
    python -m experiments.plot_auriga_df \
        --eval-dir "$output_dir" \
        "${run_args[@]}" \
        --dpi "$plot_dpi"
fi
