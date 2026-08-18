"""One-off Auriga truth validation; edit the constants below before running."""

from __future__ import annotations

from experiments.paths import resolve_path
from experiments.validation.auriga_truth import evaluate_auriga_truth


EXPERIMENT_DIR = "runs/halo12/static-baseline"
DATA_PATH = "data/auriga/halo12_all_mass.h5"
N_EVAL = 65_536
BATCH_SIZE = 4_096
SEED = 0
TRUTH_POTENTIAL_SCALE = 1.0
TRUTH_ACCELERATION_SCALE = 1.0
RADIAL_BINS = 12
SLICE_PHI_BINS = 6
SLICE_R_BINS = 40
SLICE_Z_BINS = 40
SLICE_MIN_COUNT = 3


def main() -> None:
    experiment_dir = resolve_path(EXPERIMENT_DIR)
    result = evaluate_auriga_truth(
        resolve_path(DATA_PATH),
        experiment_dir / "df",
        experiment_dir / "phi",
        experiment_dir / "validation" / "auriga",
        n_eval=N_EVAL,
        batch_size=BATCH_SIZE,
        seed=SEED,
        truth_potential_scale=TRUTH_POTENTIAL_SCALE,
        truth_acceleration_scale=TRUTH_ACCELERATION_SCALE,
        radial_bins=RADIAL_BINS,
        slice_phi_bins=SLICE_PHI_BINS,
        slice_r_bins=SLICE_R_BINS,
        slice_z_bins=SLICE_Z_BINS,
        slice_min_count=SLICE_MIN_COUNT,
    )
    print(f"Wrote Auriga truth artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
