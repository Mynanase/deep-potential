#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${DATA_PATH:?Set DATA_PATH to a prepared Halo12 HDF5 file with acceleration.}"
: "${RUN_DIR:?Set RUN_DIR to one trained Halo12 DF run directory.}"

output_dir="${OUTPUT_DIR:-${RUN_DIR}/eval/auriga_df_acceleration}"
subset="${SUBSET:-validation}"
n_eval="${N_EVAL:-65536}"
batch_size="${BATCH_SIZE:-1024}"
radial_bins="${RADIAL_BINS:-12}"
acceleration_scale="${TRUTH_ACCELERATION_SCALE:-1.0}"

python -m experiments.eval_auriga_df_acceleration \
    --data "$DATA_PATH" \
    --df-run-dir "$RUN_DIR" \
    --out-dir "$output_dir" \
    --subset "$subset" \
    --n-eval "$n_eval" \
    --batch-size "$batch_size" \
    --radial-bins "$radial_bins" \
    --truth-acceleration-scale "$acceleration_scale"
