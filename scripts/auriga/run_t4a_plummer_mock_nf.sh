#!/usr/bin/env bash
# T4a: train the project conditional NF on the frozen full Plummer mock.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${T4A_PYTHON:-/home/qiutao/miniforge3/envs/dp-jax/bin/python}
MOCK_SRC=${T4A_MOCK_SOURCE:-/home/qiutao/.orx/runs/fade7ea7-d38c-4796-8450-8daa0139393e/repo/runs/nf-score-audit/t3/plummer_full_mock.h5}
MOCK=runs/nf-score-audit/t4a/plummer_full_mock.h5
export JAX_PLATFORMS=${T4A_JAX_PLATFORMS:-cuda}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

echo "[t4a.runner] effective python=$PY backend=$JAX_PLATFORMS"
test -x "$PY" || { echo "[t4a.runner] missing python $PY"; exit 2; }
test -f "$MOCK_SRC" || { echo "[t4a.runner] missing frozen T3 mock $MOCK_SRC"; exit 2; }
mkdir -p "$(dirname "$MOCK")"
if [[ ! -f "$MOCK" ]]; then cp "$MOCK_SRC" "$MOCK"; fi
"$PY" -m pytest tests/test_t4a_plummer_nf.py -q
"$PY" -u scripts/auriga/t4a_plummer_nf.py \
  --mock "$MOCK" \
  --run-dir runs/nf-score-audit/t4a/nf \
  --cache-dir runs/nf-score-audit/t4a/cache
test -f runs/nf-score-audit/t4a/cache/manifest.json
test -f runs/nf-score-audit/t4a/nf/metadata.json
echo "[t4a.runner] done"
