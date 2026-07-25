"""Prepare an Auriga/Gadget stellar snapshot for Deep Potential."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from dpjax.datasets.auriga import (
    AurigaSnapshot,
    align_snapshot,
    center_snapshot,
    classify_kinematic_components,
    load_auriga_snapshot,
    save_auriga_snapshot,
    select_snapshot,
)


def prepare_auriga(
    input_path: str | Path,
    output_path: str | Path,
    *,
    group: str | None = None,
    align_to: str | Path | None = None,
    align_group: str | None = None,
    atol: float | Sequence[float] | None = None,
    position_center: Sequence[float] | None = None,
    velocity_center: Sequence[float] | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    components: Sequence[int | str] | None = None,
    recompute_components: bool = False,
    component_bins: int = 101,
    weight_by_mass: bool = False,
    max_particles: int | None = None,
    seed: int = 0,
    length_unit: str = "unknown",
    velocity_unit: str = "unknown",
    potential_unit: str = "unknown",
    acceleration_unit: str = "unknown",
) -> tuple[Path, AurigaSnapshot, str | None]:
    """Load, frame-center, align/select, and save the canonical data contract."""
    snapshot = load_auriga_snapshot(input_path, group=group)
    snapshot = center_snapshot(
        snapshot,
        position_center=position_center,
        velocity_center=velocity_center,
    )
    alignment_method = None
    if align_to is not None:
        target = load_auriga_snapshot(align_to, group=align_group)
        snapshot, alignment_method = align_snapshot(
            snapshot,
            target,
            atol=atol,
        )
    if recompute_components:
        snapshot = classify_kinematic_components(
            snapshot,
            bins=component_bins,
        )
    snapshot = select_snapshot(
        snapshot,
        r_min=r_min,
        r_max=r_max,
        components=components,
        max_particles=max_particles,
        seed=seed,
    )
    attrs = dict(snapshot.attrs)
    attrs["prepared_from"] = str(Path(input_path).resolve())
    if align_to is not None:
        attrs["aligned_to"] = str(Path(align_to).resolve())
    if weight_by_mass:
        if snapshot.mass is None:
            raise ValueError(
                "Mass weighting requested, but the input has no mass dataset."
            )
        mean_mass = float(snapshot.mass.mean())
        if not mean_mass > 0:
            raise ValueError("Cannot construct weights from non-positive mean mass.")
        snapshot = replace(
            snapshot,
            tracer_weight=(snapshot.mass / mean_mass).astype("f4"),
        )
        attrs["tracer_weight_definition"] = "mass / mean(selected mass)"
    snapshot = replace(snapshot, attrs=attrs)
    path = save_auriga_snapshot(
        snapshot,
        output_path,
        length_unit=length_unit,
        velocity_unit=velocity_unit,
        potential_unit=potential_unit,
        acceleration_unit=acceleration_unit,
    )
    return path, snapshot, alignment_method


def _parse_atol(values: list[float] | None) -> float | list[float] | None:
    if values is None:
        return None
    if len(values) == 1:
        return values[0]
    if len(values) == 6:
        return values
    raise ValueError("--atol accepts either 1 value or 6 per-coordinate values.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Auriga/Gadget stellar particles to the canonical "
            "Deep Potential eta+truth HDF5 contract."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", default=None)
    parser.add_argument(
        "--align-to",
        type=Path,
        default=None,
        help="Optional eta/mock file whose row order and subset should be used.",
    )
    parser.add_argument("--align-group", default=None)
    parser.add_argument(
        "--atol",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Coordinate-match tolerance when IDs are unavailable: one value "
            "or six values for x y z vx vy vz."
        ),
    )
    parser.add_argument("--position-center", type=float, nargs=3, default=None)
    parser.add_argument("--velocity-center", type=float, nargs=3, default=None)
    parser.add_argument("--r-min", type=float, default=None)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument(
        "--component",
        action="append",
        default=None,
        help=(
            "Kinematic component to retain; repeat to combine components. "
            "Choices: all, cold, warm, hot, counter."
        ),
    )
    parser.add_argument(
        "--recompute-components",
        action="store_true",
        help=(
            "Recompute circularity labels using Potential + 0.5*v^2 before "
            "component selection."
        ),
    )
    parser.add_argument("--component-bins", type=int, default=101)
    parser.add_argument(
        "--weight-by-mass",
        action="store_true",
        help="Write tracer_weight = Masses / mean(Masses) for mass-weighted DF.",
    )
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length-unit", default="unknown")
    parser.add_argument("--velocity-unit", default="unknown")
    parser.add_argument("--potential-unit", default="unknown")
    parser.add_argument("--acceleration-unit", default="unknown")
    args = parser.parse_args()

    try:
        atol = _parse_atol(args.atol)
    except ValueError as exc:
        parser.error(str(exc))

    path, snapshot, alignment_method = prepare_auriga(
        args.input,
        args.output,
        group=args.group,
        align_to=args.align_to,
        align_group=args.align_group,
        atol=atol,
        position_center=args.position_center,
        velocity_center=args.velocity_center,
        r_min=args.r_min,
        r_max=args.r_max,
        components=args.component,
        recompute_components=args.recompute_components,
        component_bins=args.component_bins,
        weight_by_mass=args.weight_by_mass,
        max_particles=args.max_particles,
        seed=args.seed,
        length_unit=args.length_unit,
        velocity_unit=args.velocity_unit,
        potential_unit=args.potential_unit,
        acceleration_unit=args.acceleration_unit,
    )
    summary = {
        "output": str(path),
        "n_particles": snapshot.n_particles,
        "alignment_method": alignment_method,
        "has_particle_id": snapshot.particle_id is not None,
        "has_mass": snapshot.mass is not None,
        "has_potential_truth": snapshot.potential is not None,
        "has_acceleration_truth": snapshot.acceleration is not None,
        "has_tracer_weight": snapshot.tracer_weight is not None,
        "has_component_labels": snapshot.component is not None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
