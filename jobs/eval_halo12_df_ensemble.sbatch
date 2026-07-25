#!/usr/bin/env bash
#SBATCH --job-name=halo12-df-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the prepared Halo12 HDF5 file.}"

repo_root="${DEEP_POTENTIAL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
run_root="${RUN_ROOT:-runs/halo_12/df_ffjord_v23_mass}"
output_dir="${OUTPUT_DIR:-${run_root}/ensemble_evaluation}"
conda_env="${CONDA_ENV_NAME:-dp-jax}"

cd "$repo_root"
eval "$(conda shell.bash hook)"
conda activate "$conda_env"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python -m experiments.eval_auriga_df \
    --data "$DATA_PATH" \
    --run-dir "${run_root}/seed_42" \
    --run-dir "${run_root}/seed_43" \
    --run-dir "${run_root}/seed_44" \
    --run-dir "${run_root}/seed_45" \
    --output-dir "$output_dir" \
    --n-samples-per-model 262144 \
    --n-score-points 32768
