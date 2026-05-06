#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN_DIR="runs/halo_12/df_ffjord_v14"

LOG_FILE="$RUN_DIR/training.log"

nohup /home/qiutao/miniforge3/envs/dp-jax/bin/python \
    experiments/train_df.py \
    --config configs/df_halo12_ffjord_v14.yaml \
    --data data/halo_12_train.h5 \
    --run-dir "$RUN_DIR" \
    --logger wandb \
    --project dp-plummer \
    --run-name halo12-df-v14-short
    > "$LOG_FILE" 2>&1 &

echo $! > "$RUN_DIR/pid.txt"
echo "Training started, PID=$(cat $RUN_DIR/pid.txt), log=$LOG_FILE"