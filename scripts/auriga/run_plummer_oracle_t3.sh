#!/usr/bin/env bash
# T3 runner: local CPU-only Plummer oracle, mock regeneration, and analytic
# diagnostics.  No training and no Auriga data access.
set -euo pipefail

# Full-suite compatibility environment (includes e3nn_jax / flowjax).
PY=/Users/qttao/.local/share/openresearch/worktrees/686a3ce0-1646-4fe7-9155-9fa5762f4020/chat_128b5ccc-f2a2-48c3-8422-d34fb74cb5cf/.venv/bin/python
OUT=runs/nf-score-audit/t3
cd "$(dirname "$0")/../.."

test -x "$PY" || { echo "[t3.runner] missing python $PY"; exit 1; }
test -f scripts/plummer/plummer_oracle.py || { echo "[t3.runner] missing oracle script"; exit 1; }
test -f scripts/auriga/local_force_inversion.py || { echo "[t3.runner] missing force inversion"; exit 1; }

mkdir -p "$OUT"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/dpjax-t3-mpl

echo "[t3.runner] analytic CPU tests"
"$PY" -m pytest tests/test_plummer_oracle.py -q

if [ -f "$OUT/plummer_full_mock.h5" ]; then
  echo "[t3.runner] full mock already exists; refusing to overwrite"
  exit 2
fi
echo "[t3.runner] regenerate frozen full mock (N=1,619,615; local CPU)"
"$PY" -u scripts/plummer/plummer_oracle.py \
  --output "$OUT/plummer_full_mock.h5" \
  --metrics "$OUT/metrics.json"

echo "[t3.runner] summary"
"$PY" - "$OUT/metrics.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
mock = m["mock"]
print("mock_n:", mock["n"])
print("band_counts:", mock["counts"])
print("band_fractions:", mock["fractions"])
print("frozen_rel_residuals:", mock["frozen_rel_residual"])
print("band_gate:", mock["pass_vs_frozen_abs_rel_le_0.25"])
print("sampler_discretization:", m["sampler_discretization"])
print("O1:", m["o0_o1_a5"]["O1"])
print("A5_gradient_error:", m["o0_o1_a5"]["A5"]["gradient_error"])
if not mock["pass_vs_frozen_abs_rel_le_0.25"]:
    raise SystemExit("band gate failed")
EOF
echo "[t3.runner] done"
