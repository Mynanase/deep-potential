#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

: "${INPUT_PATH:?Set INPUT_PATH to halo_12_stars.hdf5.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for the canonical HDF5 file.}"

component="${COMPONENT:-all}"

python -m experiments.prepare_auriga \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --group PartType4 \
    --recompute-components \
    --component "$component" \
    --weight-by-mass \
    --seed 42 \
    --length-unit kpc \
    --velocity-unit km/s \
    --potential-unit '(km/s)^2'
