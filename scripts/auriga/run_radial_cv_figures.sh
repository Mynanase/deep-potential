#!/usr/bin/env bash
# Display layer and figures for the radial conditional-velocity review
# (plan: artifacts/df-phase1-audit/radial-conditional-velocity-audit-plan.md, section 3).
#
# The metrics node (parent) produced the numbers and persisted the samples; this
# node defines the display window in km/s, reports the out-of-window mass
# fraction for every series, and draws the six figures the plan requires.  It
# reloads no model and recomputes no metric, so it runs on CPU.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
# products of the frozen metrics node: set RCV_PARENT_RUN to pin one, otherwise
# take the most recent completed copy on this host (a directory that already
# holds both rcv_summary.json and rcv_arrays.npz, so a still-running parent is
# never picked up half-written)
RCV_PARENT_RUN="${RCV_PARENT_RUN:-}"
if [ -z "$RCV_PARENT_RUN" ]; then
  for d in $(ls -dt /home/qiutao/.orx/runs/*/repo/runs/halo12-radial-cv-20260917 2>/dev/null); do
    if [ -f "$d/rcv_summary.json" ] && [ -f "$d/rcv_arrays.npz" ]; then
      RCV_PARENT_RUN="$d"; break
    fi
  done
fi
PARENT="$RCV_PARENT_RUN"
OUT=runs/halo12-radial-cv-20260917
export MPLCONFIGDIR=/tmp/orx-mpl JAX_PLATFORMS=cpu
mkdir -p "$MPLCONFIGDIR"

echo '=== RADIAL CONDITIONAL-VELOCITY FIGURES PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
"$PY" -c 'import numpy, pandas, scipy, matplotlib' \
  || { echo 'PREFLIGHT FAIL: dp-jax env lacks plotting deps'; exit 2; }
for f in rcv_arrays.npz rcv_stats.csv rcv_corr.csv rcv_reference.csv; do
  test -f "$PARENT/$f" || { echo "PREFLIGHT FAIL: missing $PARENT/$f"; exit 2; }
done
echo "parent products: $PARENT"

# copy the parent's persisted products into this run's own directory, so this
# node's outputs are self-contained and the frozen parent is never written to
mkdir -p "$OUT"
cp "$PARENT"/rcv_arrays.npz "$PARENT"/rcv_stats.csv "$PARENT"/rcv_corr.csv \
   "$PARENT"/rcv_reference.csv "$PARENT"/rcv_support.csv "$PARENT"/rcv_summary.json \
   "$PARENT"/rcv_tables.md "$PARENT"/rcv_block_sensitivity.csv \
   "$PARENT"/rcv_ckpt_stability.csv "$OUT"/ 2>/dev/null || true
for f in rcv_arrays.npz rcv_stats.csv rcv_corr.csv rcv_reference.csv; do
  test -f "$OUT/$f" || { echo "PREFLIGHT FAIL: could not stage $f"; exit 2; }
done

echo '=== STEP 1/1: display window (km/s), out-of-window mass, and figures ==='
"$PY" -u scripts/auriga/df_phase1_radial_cv_figures.py --run-dir "$OUT"

echo '=== RADIAL CONDITIONAL-VELOCITY FIGURES DONE ==='
ls -1 "$OUT" "$OUT/figs"
