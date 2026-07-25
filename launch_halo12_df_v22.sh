#!/usr/bin/env bash
#SBATCH --job-name=halo12-df-v22
#SBATCH --output=runs/halo_12/df_ffjord_v22/%j.out
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

# v22 ablation: same architecture as v21 (12-layer FFJORD, k/jac reg 1e-5)
# but transform: none. Test whether CoordinateTransform is necessary
# to keep training stable.
# Expected: divergence / numerical instability (mirrors v10-v18 era).

set -euo pipefail
cd /localdisk/kosmos/my-deep-potential
source /home/qiutao/miniforge3/etc/profile.d/conda.sh
conda activate dp-jax

# 双 GPU: 用户说明可用 1-4 号,取与 v21 一致的 2,3 习惯
export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python experiments/train_df.py \
    --config configs/df_halo12_ffjord_v22.yaml \
    --data data/halo_12_train.h5 \
    --run-dir runs/halo_12/df_ffjord_v22 \
    --logger wandb \
    --project dp-plummer \
    --run-name halo12-df-v22