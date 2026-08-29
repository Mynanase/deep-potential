"""Detect and remove the densest outer spatial clump from an Auriga snapshot.

The operation is reversible: the source HDF5 is never modified, retained rows
carry their original ``source_index``, and the removed rows are persisted in a
separate NPZ artifact and in the output HDF5 ``cleaning`` group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

from experiments.datasets.auriga import (
    AurigaSnapshot,
    load_auriga_snapshot,
    save_auriga_snapshot,
)
from experiments.datasets.phase_space import phase_space_sha256


SELECTION_SCHEMA = "dpjax.outer-spatial-clump-selection.v1"
DEFAULT_INPUT = Path("data/halo_12_stars.hdf5")
DEFAULT_OUTPUT = Path(
    "data/auriga/halo12_all_mass_clean_outer_clump.h5"
)
DEFAULT_MASK_OUTPUT = Path(
    "runs/halo12/outer-clump-removal/outer_clump_mask.npz"
)
DEFAULT_DIAGNOSTIC_OUTPUT = Path(
    "runs/halo12/outer-clump-removal/outer_clump_diagnostic.png"
)


@dataclass(frozen=True)
class OuterClumpSelection:
    """Rows belonging to one density-connected outer spatial clump."""

    source_size: int
    search_indices: np.ndarray
    removed_indices: np.ndarray
    neighbor_counts: np.ndarray
    outer_radius_min: float
    linking_length: float
    min_neighbors: int
    peak_index: int

    @property
    def keep_indices(self) -> np.ndarray:
        keep = np.ones(self.source_size, dtype=bool)
        keep[self.removed_indices] = False
        return np.flatnonzero(keep)

    @property
    def removed_fraction(self) -> float:
        return float(self.removed_indices.size / self.source_size)

    @property
    def removed_neighbor_counts(self) -> np.ndarray:
        local = np.searchsorted(self.search_indices, self.removed_indices)
        return self.neighbor_counts[local]


def detect_densest_outer_clump(
    positions: np.ndarray,
    *,
    outer_radius_min: float = 40.0,
    linking_length: float = 1.0,
    min_neighbors: int = 16,
    workers: int = -1,
) -> OuterClumpSelection:
    """Return the DBSCAN-like component containing the densest outer point.

    Density is the number of particles inside ``linking_length`` (including
    the query particle). Core particles have at least ``min_neighbors``.
    Starting at the globally densest point in the outer search region, core
    particles are linked transitively and their one-link border is included.
    This is the standard density-connectivity rule for a single DBSCAN cluster
    without depending on scikit-learn.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected positions shape (N, 3), got {positions.shape}."
        )
    if positions.shape[0] == 0:
        raise ValueError("positions must contain at least one row.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain NaN or Inf values.")
    if outer_radius_min < 0.0:
        raise ValueError("outer_radius_min must be non-negative.")
    if linking_length <= 0.0:
        raise ValueError("linking_length must be positive.")
    if min_neighbors < 2:
        raise ValueError("min_neighbors must be at least 2.")

    radius = np.linalg.norm(positions, axis=1)
    search_indices = np.flatnonzero(radius >= float(outer_radius_min))
    if search_indices.size < min_neighbors:
        raise ValueError(
            "Outer search region has fewer rows than min_neighbors."
        )

    search_positions = positions[search_indices]
    tree = cKDTree(search_positions)
    neighbor_counts = np.asarray(
        tree.query_ball_point(
            search_positions,
            float(linking_length),
            return_length=True,
            workers=int(workers),
        ),
        dtype=np.int64,
    )
    peak_local = int(np.argmax(neighbor_counts))
    if int(neighbor_counts[peak_local]) < min_neighbors:
        raise ValueError(
            "No outer point satisfies the requested density threshold."
        )

    core = neighbor_counts >= int(min_neighbors)
    visited_core = np.zeros(search_indices.size, dtype=bool)
    member = np.zeros(search_indices.size, dtype=bool)
    queue: deque[int] = deque([peak_local])
    visited_core[peak_local] = True

    while queue:
        current = queue.popleft()
        neighbors = np.asarray(
            tree.query_ball_point(
                search_positions[current],
                float(linking_length),
            ),
            dtype=np.int64,
        )
        member[neighbors] = True
        new_core = neighbors[core[neighbors] & ~visited_core[neighbors]]
        if new_core.size:
            visited_core[new_core] = True
            queue.extend(new_core.tolist())

    removed_indices = search_indices[np.flatnonzero(member)]
    if removed_indices.size == 0:
        raise RuntimeError("Density expansion unexpectedly returned no rows.")

    return OuterClumpSelection(
        source_size=int(positions.shape[0]),
        search_indices=search_indices.astype(np.int64, copy=False),
        removed_indices=removed_indices.astype(np.int64, copy=False),
        neighbor_counts=neighbor_counts,
        outer_radius_min=float(outer_radius_min),
        linking_length=float(linking_length),
        min_neighbors=int(min_neighbors),
        peak_index=int(search_indices[peak_local]),
    )


def _pca_sigma(values: np.ndarray) -> np.ndarray:
    covariance = np.cov(np.asarray(values, dtype=np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    return np.sqrt(np.maximum(eigenvalues, 0.0))


def summarize_selection(
    snapshot: AurigaSnapshot,
    selection: OuterClumpSelection,
) -> dict[str, Any]:
    """Build JSON-serializable spatial, kinematic, and density diagnostics."""
    eta = np.asarray(snapshot.eta, dtype=np.float64)
    position = eta[:, :3]
    velocity = eta[:, 3:]
    removed = selection.removed_indices
    radius = np.linalg.norm(position, axis=1)
    removed_position = position[removed]
    removed_velocity = velocity[removed]

    search_member = np.zeros(selection.search_indices.size, dtype=bool)
    removed_local = np.searchsorted(
        selection.search_indices,
        selection.removed_indices,
    )
    search_member[removed_local] = True
    background_counts = selection.neighbor_counts[~search_member]
    background_max = (
        int(background_counts.max()) if background_counts.size else 0
    )

    position_sigma = _pca_sigma(removed_position)
    velocity_sigma = _pca_sigma(removed_velocity)
    return {
        "schema": SELECTION_SCHEMA,
        "source_size": int(selection.source_size),
        "search_size": int(selection.search_indices.size),
        "removed_count": int(removed.size),
        "kept_count": int(selection.source_size - removed.size),
        "removed_fraction": selection.removed_fraction,
        "outer_radius_min": selection.outer_radius_min,
        "linking_length": selection.linking_length,
        "min_neighbors": selection.min_neighbors,
        "peak_source_row": selection.peak_index,
        "peak_neighbor_count": int(selection.removed_neighbor_counts.max()),
        "background_max_neighbor_count": background_max,
        "position_mean": removed_position.mean(axis=0).tolist(),
        "position_median": np.median(removed_position, axis=0).tolist(),
        "position_pca_sigma": position_sigma.tolist(),
        "position_axis_ratio": (position_sigma / position_sigma[0]).tolist(),
        "velocity_mean": removed_velocity.mean(axis=0).tolist(),
        "velocity_median": np.median(removed_velocity, axis=0).tolist(),
        "velocity_pca_sigma": velocity_sigma.tolist(),
        "radius_quantiles": dict(
            zip(
                ("min", "p05", "median", "p95", "max"),
                np.percentile(radius[removed], [0, 5, 50, 95, 100]).tolist(),
            )
        ),
    }


def _indices_sha256(indices: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def save_selection_artifact(
    path: str | Path,
    *,
    snapshot: AurigaSnapshot,
    selection: OuterClumpSelection,
    source_sha256: str,
) -> Path:
    """Persist the exact removed source rows and their identity contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_index = np.asarray(snapshot.source_index, dtype=np.int64)
    removed_source_index = source_index[selection.removed_indices]
    payload: dict[str, Any] = {
        "schema": np.array(SELECTION_SCHEMA),
        "source_size": np.array(selection.source_size, dtype=np.int64),
        "source_sha256": np.array(source_sha256),
        "outer_radius_min": np.array(
            selection.outer_radius_min,
            dtype=np.float64,
        ),
        "linking_length": np.array(
            selection.linking_length,
            dtype=np.float64,
        ),
        "min_neighbors": np.array(
            selection.min_neighbors,
            dtype=np.int64,
        ),
        "peak_source_row": np.array(selection.peak_index, dtype=np.int64),
        "removed_row_index": selection.removed_indices,
        "removed_source_index": removed_source_index,
        "removed_neighbor_count": selection.removed_neighbor_counts,
    }
    if snapshot.particle_id is not None:
        payload["removed_particle_id"] = np.asarray(snapshot.particle_id)[
            selection.removed_indices
        ]
    np.savez_compressed(path, **payload)
    return path


def write_clean_snapshot(
    path: str | Path,
    *,
    snapshot: AurigaSnapshot,
    selection: OuterClumpSelection,
    source_sha256: str,
    mask_path: str | Path,
    summary: dict[str, Any],
    weight_by_mass: bool = True,
    length_unit: str = "kpc",
    velocity_unit: str = "km/s",
    potential_unit: str = "(km/s)^2",
) -> Path:
    """Write the canonical clean HDF5 plus a self-contained cleaning group."""
    clean = snapshot.subset(selection.keep_indices)
    attrs = dict(clean.attrs)
    attrs.update(
        {
            "cleaning_schema": SELECTION_SCHEMA,
            "cleaning_source_sha256": source_sha256,
            "cleaning_mask_path": str(Path(mask_path)),
            "cleaning_removed_count": int(selection.removed_indices.size),
            "cleaning_removed_fraction": selection.removed_fraction,
            "cleaning_outer_radius_min": selection.outer_radius_min,
            "cleaning_linking_length": selection.linking_length,
            "cleaning_min_neighbors": selection.min_neighbors,
            "cleaning_removed_indices_sha256": _indices_sha256(
                np.asarray(snapshot.source_index, dtype=np.int64)[
                    selection.removed_indices
                ]
            ),
        }
    )
    if weight_by_mass:
        if clean.mass is None:
            raise ValueError(
                "Mass weighting requested, but the source has no mass field."
            )
        mean_mass = float(np.mean(clean.mass, dtype=np.float64))
        if not mean_mass > 0.0:
            raise ValueError("Cannot construct weights from non-positive mass.")
        clean = replace(
            clean,
            tracer_weight=(clean.mass / mean_mass).astype(np.float32),
        )
        attrs["tracer_weight_definition"] = "mass / mean(clean mass)"
    clean = replace(clean, attrs=attrs)

    path = save_auriga_snapshot(
        clean,
        path,
        length_unit=length_unit,
        velocity_unit=velocity_unit,
        potential_unit=potential_unit,
    )
    removed_source_index = np.asarray(snapshot.source_index, dtype=np.int64)[
        selection.removed_indices
    ]
    with h5py.File(path, "a") as handle:
        cleaning = handle.create_group("cleaning")
        cleaning.attrs["schema"] = SELECTION_SCHEMA
        cleaning.attrs["source_sha256"] = source_sha256
        cleaning.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        cleaning.create_dataset(
            "removed_row_index",
            data=selection.removed_indices,
            compression="lzf",
            chunks=True,
        )
        cleaning.create_dataset(
            "removed_source_index",
            data=removed_source_index,
            compression="lzf",
            chunks=True,
        )
        if snapshot.particle_id is not None:
            cleaning.create_dataset(
                "removed_particle_id",
                data=np.asarray(snapshot.particle_id)[selection.removed_indices],
                compression="lzf",
                chunks=True,
            )
    return path


def save_diagnostic_figure(
    path: str | Path,
    *,
    snapshot: AurigaSnapshot,
    selection: OuterClumpSelection,
) -> Path:
    """Save three spatial projections and the local-density separation."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    position = np.asarray(snapshot.eta[:, :3], dtype=np.float64)
    search_position = position[selection.search_indices]
    removed_position = position[selection.removed_indices]
    removed_local = np.searchsorted(
        selection.search_indices,
        selection.removed_indices,
    )
    member = np.zeros(selection.search_indices.size, dtype=bool)
    member[removed_local] = True

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    projections = (
        (0, 1, "x", "y"),
        (0, 2, "x", "z"),
        (1, 2, "y", "z"),
    )
    for axis, (first, second, xlabel, ylabel) in zip(
        axes.flat[:3],
        projections,
    ):
        axis.scatter(
            search_position[:, first],
            search_position[:, second],
            s=1.0,
            c="#5f6368",
            alpha=0.12,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            removed_position[:, first],
            removed_position[:, second],
            s=2.0,
            c="#d95f02",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)

    histogram_axis = axes.flat[3]
    density_log = np.log10(selection.neighbor_counts.astype(np.float64))
    bins = np.linspace(0.0, float(density_log.max()), 50)
    histogram_axis.hist(
        density_log[~member],
        bins=bins,
        color="#5f6368",
        alpha=0.65,
        label="outer background",
    )
    histogram_axis.hist(
        density_log[member],
        bins=bins,
        color="#d95f02",
        alpha=0.65,
        label="removed clump",
    )
    histogram_axis.axvline(
        np.log10(selection.min_neighbors),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="core threshold",
    )
    histogram_axis.set_xlabel(
        f"log10 neighbors within {selection.linking_length:g}"
    )
    histogram_axis.set_ylabel("particle count")
    histogram_axis.set_yscale("log")
    histogram_axis.legend(frameon=False)
    histogram_axis.grid(alpha=0.15)
    fig.suptitle(
        "Halo12 outer density-connected clump: "
        f"{selection.removed_indices.size:,} removed / "
        f"{selection.source_size:,} total"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the densest outer spatial clump and write a reversible "
            "clean canonical Auriga HDF5."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--group", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=DEFAULT_MASK_OUTPUT,
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_OUTPUT,
    )
    parser.add_argument("--outer-radius-min", type=float, default=40.0)
    parser.add_argument("--linking-length", type=float, default=1.0)
    parser.add_argument("--min-neighbors", type=int, default=16)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--no-mass-weights",
        action="store_true",
        help="Do not write tracer_weight = mass / mean(clean mass).",
    )
    parser.add_argument("--length-unit", default="kpc")
    parser.add_argument("--velocity-unit", default="km/s")
    parser.add_argument("--potential-unit", default="(km/s)^2")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived outputs; the source HDF5 is never changed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    derived_paths = (
        args.output,
        args.mask_output,
        args.diagnostic_output,
    )
    existing = [path for path in derived_paths if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "derived output already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    snapshot = load_auriga_snapshot(args.input, group=args.group)
    selection = detect_densest_outer_clump(
        snapshot.eta[:, :3],
        outer_radius_min=args.outer_radius_min,
        linking_length=args.linking_length,
        min_neighbors=args.min_neighbors,
        workers=args.workers,
    )
    source_sha256 = phase_space_sha256(snapshot.eta)
    summary = summarize_selection(snapshot, selection)
    summary["source"] = str(args.input.resolve())
    summary["source_sha256"] = source_sha256

    mask_path = save_selection_artifact(
        args.mask_output,
        snapshot=snapshot,
        selection=selection,
        source_sha256=source_sha256,
    )
    output_path = write_clean_snapshot(
        args.output,
        snapshot=snapshot,
        selection=selection,
        source_sha256=source_sha256,
        mask_path=mask_path,
        summary=summary,
        weight_by_mass=not args.no_mass_weights,
        length_unit=args.length_unit,
        velocity_unit=args.velocity_unit,
        potential_unit=args.potential_unit,
    )
    diagnostic_path = save_diagnostic_figure(
        args.diagnostic_output,
        snapshot=snapshot,
        selection=selection,
    )
    summary.update(
        {
            "output": str(output_path),
            "mask_output": str(mask_path),
            "diagnostic_output": str(diagnostic_path),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
