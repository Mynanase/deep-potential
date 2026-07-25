#!/usr/bin/env bash
#SBATCH --job-name=halo12-df-ens
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-3
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the prepared Halo12 HDF5 file.}"

repo_root="${DEEP_POTENTIAL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
config="${CONFIG:-configs/df_halo12_ffjord_v23_mass.yaml}"
run_root="${RUN_ROOT:-runs/halo_12/df_ffjord_v23_mass}"
logger="${LOGGER:-wandb}"
wandb_project="${WANDB_PROJECT:-deep-potential}"
conda_env="${CONDA_ENV_NAME:-dp-jax}"
seed=$((42 + SLURM_ARRAY_TASK_ID))
split_seed=$((1042 + SLURM_ARRAY_TASK_ID))
run_dir="${run_root}/seed_${seed}"
override="$(printf '{"seed": %d, "data": {"split_seed": %d}}' "$seed" "$split_seed")"

cd "$repo_root"
eval "$(conda shell.bash hook)"
conda activate "$conda_env"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python -m experiments.train_df \
    --config "$config" \
    --data "$DATA_PATH" \
    --run-dir "$run_dir" \
    --override "$override" \
    --logger "$logger" \
    --project "$wandb_project" \
    --run-name "halo12-v23-seed-${seed}"
