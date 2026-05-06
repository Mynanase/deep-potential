#!/usr/bin/env bash
#SBATCH --job-name=halo12-df-v18
#SBATCH --output=runs/halo_12/df_ffjord_v18/%j.out
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

set -euo pipefail
cd /localdisk/kosmos/my-deep-potential
source /home/qiutao/miniforge3/etc/profile.d/conda.sh
conda activate dp-jax

export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python experiments/train_df.py \
    --config configs/df_halo12_ffjord_v18.yaml \
    --data data/halo_12_train.h5 \
    --run-dir runs/halo_12/df_ffjord_v18 \
    --logger wandb \
    --project dp-plummer \
    --run-name halo12-df-v18