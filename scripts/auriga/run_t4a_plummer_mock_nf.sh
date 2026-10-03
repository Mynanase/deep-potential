#!/usr/bin/env bash
# T4a: train a full-data Plummer mock FFJORD and certify its score cache.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${T4A_PYTHON:-/home/qiutao/miniforge3/envs/dp-jax/bin/python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=${T4A_JAX_PLATFORMS:-cuda}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[t4a.runner] effective python=$PY backend=$JAX_PLATFORMS"
"$PY" -m pytest tests/test_t4a_plummer_nf.py -q
"$PY" -u scripts/auriga/t4a_plummer_nf.py \
  --mock runs/nf-score-audit/t4a/plummer_full_mock.h5 \
  --run-dir runs/nf-score-audit/t4a/nf \
  --cache-dir runs/nf-score-audit/t4a/cache \
  --epochs "${T4A_EPOCHS:-32}" \
--batch-size "${T4A_BATCH_SIZE:-8192}" \
--max-batches-per-epoch "${T4A_MAX_BATCHES_PER_EPOCH:-32}"
cache_dir=runs/nf-score-audit/t4a/cache
test -f "$cache_dir/manifest.json"
test -f runs/nf-score-audit/t4a/nf/final.mpck
echo "[t4a.runner] done"
