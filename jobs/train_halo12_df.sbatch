#!/usr/bin/env bash
#SBATCH --job-name=halo12-df
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

set -euo pipefail

: "${CONFIG:?Set CONFIG to a Halo12 DF config path.}"
: "${RUN_DIR:?Set RUN_DIR to the output run directory.}"

repo_root="${DEEP_POTENTIAL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
data_path="${DATA_PATH:-data/halo_12_train.h5}"
run_name="${RUN_NAME:-$(basename "$RUN_DIR")}"
logger="${LOGGER:-wandb}"
wandb_project="${WANDB_PROJECT:-deep-potential}"
conda_env="${CONDA_ENV_NAME:-dp-jax}"

cd "$repo_root"
eval "$(conda shell.bash hook)"
conda activate "$conda_env"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python -m experiments.train_df \
    --config "$CONFIG" \
    --data "$data_path" \
    --run-dir "$RUN_DIR" \
    --logger "$logger" \
    --project "$wandb_project" \
    --run-name "$run_name"
