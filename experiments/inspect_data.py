"""Inspect the shape and basic statistics of a phase-space HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.datasets.phase_space import load_eta_h5

_LABELS = ("x", "y", "z", "vx", "vy", "vz")


def summarize_eta(eta: np.ndarray) -> dict[str, Any]:
    """Return serializable summary statistics for ``(N, 6)`` phase space."""
    eta = np.asarray(eta)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")
    if len(eta) == 0:
        raise ValueError("Cannot summarize an empty dataset.")

    radius = np.linalg.norm(eta[:, :3], axis=1)
    speed = np.linalg.norm(eta[:, 3:], axis=1)
    return {
        "shape": eta.shape,
        "dtype": str(eta.dtype),
        "mean": np.mean(eta, axis=0),
        "std": np.std(eta, axis=0),
        "min": np.min(eta, axis=0),
        "max": np.max(eta, axis=0),
        "radius": np.array(
            [np.min(radius), np.median(radius), np.max(radius)]
        ),
        "speed": np.array(
            [np.min(speed), np.median(speed), np.max(speed)]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5 phase-space dataset."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", default="eta")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Only summarize the first N rows.",
    )
    args = parser.parse_args()

    eta = load_eta_h5(args.data, dataset=args.dataset)
    if args.sample_size is not None:
        if args.sample_size <= 0:
            parser.error("--sample-size must be positive")
        eta = eta[: args.sample_size]

    summary = summarize_eta(eta)
    print(f"shape={summary['shape']}, dtype={summary['dtype']}")
    print(f"{'dim':<4} {'mean':>13} {'std':>13} {'min':>13} {'max':>13}")
    for index, label in enumerate(_LABELS):
        print(
            f"{label:<4} "
            f"{summary['mean'][index]:>13.5g} "
            f"{summary['std'][index]:>13.5g} "
            f"{summary['min'][index]:>13.5g} "
            f"{summary['max'][index]:>13.5g}"
        )

    r_min, r_median, r_max = summary["radius"]
    v_min, v_median, v_max = summary["speed"]
    print(f"r:   min={r_min:.5g}, median={r_median:.5g}, max={r_max:.5g}")
    print(f"|v|: min={v_min:.5g}, median={v_median:.5g}, max={v_max:.5g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
