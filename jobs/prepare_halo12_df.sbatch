#!/usr/bin/env bash
#SBATCH --job-name=halo12-prepare
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${INPUT_PATH:?Set INPUT_PATH to halo_12_stars.hdf5.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH for the canonical HDF5 file.}"

repo_root="${DEEP_POTENTIAL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
component="${COMPONENT:-all}"
conda_env="${CONDA_ENV_NAME:-dp-jax}"

cd "$repo_root"
eval "$(conda shell.bash hook)"
conda activate "$conda_env"

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
