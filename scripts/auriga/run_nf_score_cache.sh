#!/usr/bin/env bash
# T1 runner: control score numerical-chain audit + B/C fixed-point cache.
# Plan docs/nf-score-audit-plan.md (card T1); orchestration node 45b0446d
# only, never by hand.  The stage is COMMITTED CODE read from
# scripts/auriga/nf_score_stage.txt ('pilot' first; 'full' requires its own
# commit and explicit authorization).  Budgets (card T1): pilot <= 30 min
# single GPU, full <= 2 h single GPU.
set -euo pipefail

PY=/localdisk/kosmos/my-deep-potential/.venv-halo/bin/python
CONTROL_REPO=/home/qiutao/.orx/runs/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo
CONTROL_RUN_DIR=$CONTROL_REPO/runs/orx
POP=/localdisk/kosmos/my-deep-potential/data/auriga/halo12-clean-smooth.h5
MANIFEST=docs/nf-score-audit/t0-manifest.json
STAGE_FILE=scripts/auriga/nf_score_stage.txt

cd "$(dirname "$0")/../.."

STAGE=$(tr -d "[:space:]" < "$STAGE_FILE")
case "$STAGE" in
  pilot) AUDIT_BUDGET_S=600;  CACHE_BUDGET_S=1200; OUT=runs/nf-score-audit/t1-pilot ;;
  full)  AUDIT_BUDGET_S=900;  CACHE_BUDGET_S=6300; OUT=runs/nf-score-audit/t1-cache  ;;
  *) echo "[t1.runner] unknown stage '$STAGE' in $STAGE_FILE (expected pilot|full)"; exit 2 ;;
esac

test -x "$PY"                || { echo "[t1.runner] missing python $PY"; exit 1; }
test -f "$CONTROL_RUN_DIR/data/df_gradients.h5" || { echo "[t1.runner] missing control df_gradients.h5"; exit 1; }
test -f "$POP"               || { echo "[t1.runner] missing population $POP"; exit 1; }
test -f "$MANIFEST"          || { echo "[t1.runner] missing t0 manifest $MANIFEST"; exit 1; }
test -f scripts/auriga/audit_df_constraints.py || { echo "[t1.runner] missing audit script"; exit 1; }
test -f scripts/auriga/nf_score_cache.py       || { echo "[t1.runner] missing cache script"; exit 1; }

# buffer the full query first: awk's early exit would SIGPIPE nvidia-smi
# and pipefail would abort the runner (run cb9885ee, exit 141)
GPU_TABLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
GPU=$(printf '%s\n' "$GPU_TABLE" | awk -F", " '$2 < 1000 {print $1; exit}')
if [ -z "$GPU" ]; then GPU=0; fi
echo "[t1.runner] stage=$STAGE gpu=$GPU out=$OUT audit_budget=$AUDIT_BUDGET_S cache_budget=$CACHE_BUDGET_S"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT"

echo "[t1.runner] === numerical-chain audit ==="
set +e
timeout "$AUDIT_BUDGET_S" "$PY" -u scripts/auriga/audit_df_constraints.py score-chain \
  --control-repo "$CONTROL_REPO" --control-run-dir "$CONTROL_RUN_DIR" \
  --manifest "$MANIFEST" --out-dir "$OUT" --force
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "[t1.runner] audit exited rc=$RC (hash mismatch / non-finite / timeout: T1 stop condition, no cache)"
  exit "$RC"
fi

echo "[t1.runner] === fixed-point cache ($STAGE) ==="
set +e
timeout "$CACHE_BUDGET_S" "$PY" -u scripts/auriga/nf_score_cache.py --stage "$STAGE" \
  --control-repo "$CONTROL_REPO" --control-run-dir "$CONTROL_RUN_DIR" \
  --manifest "$MANIFEST" --population "$POP" --out-dir "$OUT"
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "[t1.runner] cache build exited rc=$RC"
  exit "$RC"
fi

echo "[t1.runner] === summary ==="
"$PY" - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
audit = json.load(open(out + "/audit_score_chain.json"))
print("audit gates:", json.dumps(audit["gates"], sort_keys=True))
m = json.load(open(out + "/metrics.json"))
print("cache points:", m["n_points"], "certified:",
      json.load(open(out + "/manifest.json"))["certification"])
print("phi_fd grad_ok/lap_ok:", m["phi_fd_spot_check"].get("grad_ok"),
      m["phi_fd_spot_check"].get("lap_ok"))
print("extrapolation:", json.dumps(m.get("extrapolation", {}), indent=1))
EOF
echo "[t1.runner] done"
