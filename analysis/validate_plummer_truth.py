"""One-off Plummer truth validation; edit the constants below before running."""

from __future__ import annotations

from experiments.paths import resolve_path
from experiments.validation.plummer import evaluate_plummer_truth


EXPERIMENT_DIR = "runs/plummer_rcut/full-oracle"
R_MIN = 1.0
R_MAX = 10.0
N_EVAL = 65_536
BATCH_SIZE = 4_096
SEED = 0
N_R = 256
SLICE_GRID = 128
SLICE_RMAX = 10.0


def main() -> None:
    experiment_dir = resolve_path(EXPERIMENT_DIR)
    result = evaluate_plummer_truth(
        experiment_dir / "df",
        experiment_dir / "phi",
        experiment_dir / "results" / "data",
        n_eval=N_EVAL,
        batch_size=BATCH_SIZE,
        seed=SEED,
        r_min=R_MIN,
        r_max=R_MAX,
        n_r=N_R,
        slice_grid=SLICE_GRID,
        slice_rmax=SLICE_RMAX,
        artifact_prefix="validation_plummer",
    )
    print(f"Wrote Plummer truth artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
