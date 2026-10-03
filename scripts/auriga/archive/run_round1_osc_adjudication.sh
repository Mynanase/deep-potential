#!/usr/bin/env bash
# Phase-2 round-1 oscillation adjudication: the three suppression variants
# vs the frozen anchors, under the committed 96^3 particle-truth product:
#   innerA    2b32eb04  (phase-2 control, radius-balanced grid, lambda=1)
#   lambda10  e1cf8f74  (reference regime, lambda=10)
#   oscpair   b502d4b4  (round-1: second-difference bending penalty)
#   snceiling e33dd48e  (round-1: spectral-norm ceiling at init sigma)
#   anneal    d3fa2716  (round-1: lambda cosine anneal 10 -> 1)
# Evaluation-only: density-oscillation spectra (plot_density_spectrum.py
# ported from the phase-1 osc-spectra line, 5f80c54) + Gauss-flux
# enclosed-mass pt-adjudication. Reads follow docs/history/phase2-baseline.md:
# quantitative domain lambda >= 10 kpc (f_hi<10 column, radial dM bands,
# 30-70 kpc negative band); <5 kpc columns are model-diagnostic only.
# No retraining.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
TRUTH=/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5
RUNS=$HOME/.orx/runs
INNERA=$RUNS/2b32eb04-d8bf-4c03-a3a9-98d0f0db1192/repo/runs/orx
LAMBDA10=$RUNS/e1cf8f74-775d-44b1-b567-24dd88e48cd0/repo/runs/orx
OSCPAIR=$RUNS/b502d4b4-3ded-4e28-834b-2d03fa267720/repo/runs/orx
SNCEILING=$RUNS/e33dd48e-9f60-44b6-857a-4f188d566e46/repo/runs/orx
ANNEAL=$RUNS/d3fa2716-f436-4617-82cb-28b5105652d9/repo/runs/orx
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== ROUND-1 OSC ADJUDICATION PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
declare -A M
M[innerA]=$INNERA
M[lambda10]=$LAMBDA10
M[oscpair]=$OSCPAIR
M[snceiling]=$SNCEILING
M[anneal]=$ANNEAL
for k in innerA lambda10 oscpair snceiling anneal; do
  test -f "${M[$k]}/models/Phi/potential-10_model.eqx" || { echo "PREFLIGHT FAIL: missing $k checkpoint at ${M[$k]}"; exit 2; }
  echo "$k: ${M[$k]}"
done
"$PY" -c 'import jax, matplotlib, h5py, numpy, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import jax; b=jax.default_backend(); print("JAX", jax.__version__, "backend", b); exit(0 if b != "cpu" else 1)'

echo '=== DENSITY-OSCILLATION SPECTRA (5 models vs 96^3 truth) ==='
"$PY" -u scripts/auriga/plot_density_spectrum.py \
  --grids "$GRIDS" \
  --model innerA="$INNERA" --model lambda10="$LAMBDA10" \
  --model oscpair="$OSCPAIR" --model snceiling="$SNCEILING" --model anneal="$ANNEAL" \
  --output-dir figures/density-spectrum-round1

echo '=== PT-ADJUDICATION (Gauss-flux enclosed mass, 5 models) ==='
TRUTH_ARG=()
if [ -f "$TRUTH" ]; then TRUTH_ARG=(--truth "$TRUTH"); fi
"$PY" scripts/auriga/plot_pt_adjudication.py \
  --grids "$GRIDS" \
  --model innerA="$INNERA" --model lambda10="$LAMBDA10" \
  --model oscpair="$OSCPAIR" --model snceiling="$SNCEILING" --model anneal="$ANNEAL" \
  "${TRUTH_ARG[@]}" --output-dir figures/pt-adjudication-round1

echo '=== ROUND-1 SUMMARY (quantitative: lambda >= 10 kpc; <5 kpc diagnostic only) ==='
"$PY" - <<'PYEOF'
import json
d = json.load(open("figures/density-spectrum-round1/density_spectrum.json"))
print("model        f_hi<10   f_hi<5    f_hi<2.5  resid.var")
for label, row in d["f_hi"].items():
    print(f"{label:>10}  {row['10']:8.1%}  {row['5']:8.1%}  {row['2.5']:8.1%}  {d['resid_var'][label]:9.3e}")
print("calibration (msd2 = mean squared second difference of rho):")
for label, row in d["calib"].items():
    if "msd2_4kpc" in row:
        print(f"{label:>10}  msd2_4kpc={row['msd2_4kpc']:8.2e}  prior_neg={row.get('prior_neg', float('nan')):.4f}")
PYEOF

echo '=== OUTPUTS ==='
find figures/density-spectrum-round1 figures/pt-adjudication-round1 -type f | sort
echo '=== ROUND-1 OSC ADJUDICATION DONE ==='
