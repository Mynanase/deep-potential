#!/usr/bin/env bash
# T2 runner: weak Stein score test calibration, then (stage 'full') the gated
# exploratory real-heldout stage.  Card T2 of docs/nf-score-audit-plan.md;
# local CPU only, no model loading, no training, no server access.
#
# Stage is COMMITTED CODE read from scripts/auriga/nf_weak_score_stage.txt:
#   calibrate  mock calibration with gates (first stage)
#   full       + exploratory real estimates; requires the certified T1 cache
#              copied read-only into runs/nf-score-audit/t1-cache-local
#              (sha256 must match docs/nf-score-audit/t1-cache-registry.json;
#              if the files are absent the real stage skips cleanly and no
#              Auriga numbers are reported)
set -euo pipefail

cd "$(dirname "$0")/../.."

STAGE_FILE=scripts/auriga/nf_weak_score_stage.txt
OUT=runs/nf-score-audit/t2-weak-score
CACHE_DIR="${WEAK_SCORE_CACHE_DIR:-runs/nf-score-audit/t1-cache-local}"
CAL_BUDGET_S=7200
REAL_BUDGET_S=3600
REQUIREMENTS=scripts/auriga/requirements-weak-score.txt

STAGE=$(tr -d "[:space:]" < "$STAGE_FILE")
case "$STAGE" in
  calibrate|full) ;;
  *) echo "[t2.runner] unknown stage '$STAGE' in $STAGE_FILE (expected calibrate|full)"; exit 2 ;;
esac

test -f scripts/auriga/nf_weak_score_tests.py || { echo "[t2.runner] missing script"; exit 1; }

# self-contained CPU environment (the orchestration snapshot carries no venv);
# pinned numpy/scipy/h5py only - the weak-score script never imports jax
if [ -z "${WEAK_SCORE_PY:-}" ]; then
  PY=.venv-weak/bin/python
  if [ ! -x "$PY" ]; then
    echo "[t2.runner] building pinned venv from $REQUIREMENTS"
    python3 -m venv .venv-weak
    .venv-weak/bin/pip install --quiet --upgrade pip
    .venv-weak/bin/pip install --quiet -r "$REQUIREMENTS"
  fi
else
  PY="$WEAK_SCORE_PY"
fi
test -x "$PY" || { echo "[t2.runner] missing python $PY"; exit 1; }

export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT"
echo "[t2.runner] stage=$STAGE py=$PY out=$OUT"

echo "[t2.runner] === calibration (gates enforced) ==="
set +e
"$PY" -u scripts/auriga/nf_weak_score_tests.py calibrate --out-dir "$OUT" --budget-s "$CAL_BUDGET_S"
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "[t2.runner] calibration exited rc=$RC (gate failure = T2 stop condition: fix the test first)"
  exit "$RC"
fi

if [ "$STAGE" = "full" ]; then
  echo "[t2.runner] === exploratory real-heldout stage (gated on certified T1 cache) ==="
  set +e
  "$PY" -u scripts/auriga/nf_weak_score_tests.py real \
    --cache-dir "$CACHE_DIR" --out-dir "$OUT" --budget-s "$REAL_BUDGET_S"
  RC=$?
  set -e
  if [ "$RC" -ne 0 ]; then
    echo "[t2.runner] real stage exited rc=$RC"
    exit "$RC"
  fi
fi

echo "[t2.runner] === summary ==="
"$PY" - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
m = json.load(open(out + "/metrics.json"))
print("gates:", json.dumps(m["gates"], sort_keys=True))
null = m["arms"]["null"]
print("null FPR univariate/maxT/familyQ:", round(null["fpr_per_stat_mean_0.05"], 3),
      round(null["fpr_maxT_splithalf_0.05"], 3), round(null["fpr_familyQ_empirical_0.05"], 3),
      "(chi2 nominal:", str(round(null["fpr_familyQ_chi2_0.05"], 3)), "- diagnostic)")
print("S_m0 power @0.25/1/2 eps_ref:",
      round(m["arms"]["S_m0_e0.25"]["power_matched_0.05"], 2),
      round(m["arms"]["S_m0_e1"]["power_matched_0.05"], 2),
      round(m["arms"]["S_m0_e2"]["power_matched_0.05"], 2))
s = m["resampling_sensitivity"]
print("resampling cluster/iid:", round(s["width_ratio_cluster_over_iid"], 2),
      "block/iid:", round(s["width_ratio_block_over_iid"], 2),
      "coverage cluster:", round(s["coverage95_cluster"], 2))
try:
    r = json.load(open(out + "/metrics_real.json"))
    print("real stage: EXPLORATORY n_window =", r["n_window"],
          "flagged =", r["flagged"], "maxT p:", r["maxT_p_values"])
except FileNotFoundError:
    print("real stage: not run (stage or certified local cache)")
EOF
echo "[t2.runner] done"
