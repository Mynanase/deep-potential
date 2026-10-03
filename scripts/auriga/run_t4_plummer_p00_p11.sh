#!/usr/bin/env bash
# T4: minimal paired P00/P11 Phi experiment on the frozen full Plummer mock.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
MOCK=${T4_MOCK_SOURCE:-$HOME/.local/share/openresearch/local-runs/fade7ea7-d38c-4796-8450-8daa0139393e/repo/runs/nf-score-audit/t3/plummer_full_mock.h5}
T4A_SNAP=/home/qiutao/.orx/runs/14e2e76a-d059-42df-8f7d-28694fff1701/repo/runs/nf-score-audit/t4a
OUT=runs/nf-score-audit/t4-p00-p11
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1

test -x "$PY" && test -f "$MOCK"
test -f "$T4A_SNAP/nf/flow-21_model.eqx" && test -f "$T4A_SNAP/cache/manifest.json"
mkdir -p "$OUT"
if [[ ! -f "$OUT/nf/flow-21_model.eqx" ]]; then cp -a "$T4A_SNAP/nf" "$OUT/"; fi
if [[ ! -f "$OUT/t4a-manifest.json" ]]; then cp "$T4A_SNAP/cache/manifest.json" "$OUT/t4a-manifest.json"; fi
"$PY" -m pytest tests/test_t4_plummer_p00_p11.py -q
"$PY" -u scripts/auriga/t4_plummer_p00_p11.py --mock "$MOCK" --flow-dir "$OUT/nf" --out-dir "$OUT"
test -f "$OUT/manifest.json"
echo "[t4.runner] done"
