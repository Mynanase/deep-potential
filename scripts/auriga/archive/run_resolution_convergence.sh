#!/usr/bin/env bash
# Resolution-convergence study of the particle-truth density spectrum
# (node particle-truth-spectrum-resolution-convergence-6): same particles as
# the frozen truth grids (sha-verified), NGP mass histograms at
# 64/96/128/192^3 over [-75,75]^3 kpc, one spectrum pipeline (512 Sobol
# directions x 128 radial nodes, trilinear interpolation, angular mean
# removed, Hann window, rfft), pre-registered 10% bin-median convergence
# criterion with a 2x-Nyquist(finest) guard, seeded A/B shot-noise check at
# 96^3 and 192^3.  CPU-only compute layer (numpy/scipy/h5py) persists
# npz+json; the plot layer reads persisted arrays only.  No training, no
# models, no GPU.
set -eo pipefail

PY=/localdisk/kosmos/my-deep-potential/.venv-halo/bin/python
GRIDS=data/auriga/halo12_particle_truth_grids.h5
ASSET=/localdisk/kosmos/my-deep-potential/data/auriga/halo12_total_matter_particles_starframe.h5
OUT=runs/orx/resolution-convergence
FIGDIR=figures/resolution-convergence
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== RESOLUTION-CONVERGENCE PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$GRIDS" || { echo "PREFLIGHT FAIL: missing $GRIDS"; exit 2; }
test -f "$ASSET" || { echo "PREFLIGHT FAIL: missing $ASSET"; exit 2; }
"$PY" -c 'import numpy, scipy, h5py, matplotlib' \
  || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
echo "python: $PY"
echo "grids:  $GRIDS"
echo "asset:  $ASSET"

echo '=== FOCUSED TESTS (analytic machinery) ==='
env JAX_PLATFORMS=cpu "$PY" -m pytest tests/test_resolution_convergence.py -q

"$PY" -u scripts/auriga/resolution_convergence.py \
  --grids "$GRIDS" --asset "$ASSET" --output-dir "$OUT"
"$PY" -u scripts/auriga/plot_resolution_convergence.py \
  --input "$OUT/resolution_convergence.npz" \
  --json "$OUT/resolution_convergence.json" --fig-dir "$FIGDIR"

echo '=== OUTPUTS ==='
find "$OUT" "$FIGDIR" -type f | sort
echo '=== RESOLUTION-CONVERGENCE DONE ==='
