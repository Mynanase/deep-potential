"""Auriga data adapter, selection, and particle alignment.

The active Deep Potential pipeline consumes a single ``eta`` array with columns
``[x, y, z, vx, vy, vz]``.  Auriga/Gadget exports are less uniform: positions
and velocities may be vector datasets or six scalar datasets, while simulator
truth and particle identifiers live beside them.  This module converts both
layouts into one validated, testable representation without coupling the core
library to a particular Halo12 file name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

AURIGA_SCHEMA = "dpjax.auriga.mock.v1"
ETA_COLUMNS = ("x", "y", "z", "vx", "vy", "vz")

_ID_NAMES = ("particle_id", "ParticleIDs", "ParticleID", "ids")
_MASS_NAMES = ("mass", "Masses")
_POTENTIAL_NAMES = ("potential", "Potential")
_ACCELERATION_NAMES = (
    "acceleration",
    "Acceleration",
    "Accelerations",
    "GravAcceleration",
)
_SOURCE_INDEX_NAMES = ("source_index", "SourceIndex")
_TRACER_WEIGHT_NAMES = ("tracer_weight", "TracerWeight", "weights")
_COMPONENT_NAMES = ("component", "kinematic_label", "KinematicLabel")
_CIRCULARITY_NAMES = ("circularity", "lambda_z", "LambdaZ")

KINEMATIC_COMPONENTS = {
    "cold": 0,
    "cold_disk": 0,
    "warm": 1,
    "warm_disk": 1,
    "hot": 2,
    "hot_bulge": 2,
    "counter": 3,
    "counter_rotating": 3,
}


@dataclass(frozen=True)
class AurigaSnapshot:
    """Aligned phase-space tracers and optional simulator truth fields."""

    eta: np.ndarray
    particle_id: np.ndarray | None = None
    mass: np.ndarray | None = None
    potential: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    source_index: np.ndarray | None = None
    tracer_weight: np.ndarray | None = None
    component: np.ndarray | None = None
    circularity: np.ndarray | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        return int(self.eta.shape[0])

    def validate(self) -> AurigaSnapshot:
        """Validate shapes, finite values, and row-wise field alignment."""
        eta = np.asarray(self.eta)
        if eta.ndim != 2 or eta.shape[1] != 6:
            raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")
        if eta.shape[0] == 0:
            raise ValueError("Auriga snapshot must contain at least one particle.")
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta contains NaN or Inf values.")

        n = eta.shape[0]
        for name in (
            "particle_id",
            "mass",
            "potential",
            "source_index",
            "tracer_weight",
            "component",
            "circularity",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim != 1 or array.shape[0] != n:
                raise ValueError(
                    f"Expected {name} shape ({n},), got {array.shape}."
                )
            if name not in {"particle_id", "source_index"} and not np.all(
                np.isfinite(array)
            ):
                raise ValueError(f"{name} contains NaN or Inf values.")
        if self.mass is not None and np.any(np.asarray(self.mass) <= 0):
            raise ValueError("mass must be strictly positive.")
        if self.tracer_weight is not None and np.any(
            np.asarray(self.tracer_weight) <= 0
        ):
            raise ValueError("tracer_weight must be strictly positive.")

        if self.acceleration is not None:
            acceleration = np.asarray(self.acceleration)
            if acceleration.shape != (n, 3):
                raise ValueError(
                    f"Expected acceleration shape ({n}, 3), "
                    f"got {acceleration.shape}."
                )
            if not np.all(np.isfinite(acceleration)):
                raise ValueError("acceleration contains NaN or Inf values.")
        return self

    def subset(self, indices: np.ndarray | Sequence[int]) -> AurigaSnapshot:
        """Return a row-aligned subset while preserving metadata."""
        indices = np.asarray(indices)
        kwargs: dict[str, Any] = {"eta": np.asarray(self.eta)[indices]}
        for name in (
            "particle_id",
            "mass",
            "potential",
            "acceleration",
            "source_index",
            "tracer_weight",
            "component",
            "circularity",
        ):
            value = getattr(self, name)
            kwargs[name] = None if value is None else np.asarray(value)[indices]
        kwargs["attrs"] = dict(self.attrs)
        return AurigaSnapshot(**kwargs).validate()


def _first_dataset(
    group: h5py.Group,
    names: Sequence[str],
) -> np.ndarray | None:
    for name in names:
        if name in group and isinstance(group[name], h5py.Dataset):
            return np.asarray(group[name])
    return None


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_eta(group: h5py.Group) -> np.ndarray:
    eta = _first_dataset(group, ("eta",))
    if eta is not None:
        return np.asarray(eta, dtype=np.float32)

    coordinates = _first_dataset(group, ("Coordinates", "coordinates"))
    velocities = _first_dataset(group, ("Velocities", "velocities"))
    if coordinates is not None or velocities is not None:
        if coordinates is None or velocities is None:
            raise KeyError(
                "Auriga group must contain both Coordinates and Velocities."
            )
        coordinates = np.asarray(coordinates)
        velocities = np.asarray(velocities)
        if (
            coordinates.ndim != 2
            or velocities.ndim != 2
            or coordinates.shape[1] != 3
            or velocities.shape[1] != 3
            or coordinates.shape[0] != velocities.shape[0]
        ):
            raise ValueError(
                "Expected Coordinates and Velocities shapes (N, 3), got "
                f"{coordinates.shape} and {velocities.shape}."
            )
        return np.concatenate([coordinates, velocities], axis=1).astype(
            np.float32,
            copy=False,
        )

    missing = [name for name in ETA_COLUMNS if name not in group]
    if missing:
        raise KeyError(
            "No supported phase-space layout found. Expected eta, "
            "Coordinates+Velocities, or scalar datasets "
            f"{ETA_COLUMNS}; missing={missing}."
        )
    return np.stack(
        [np.asarray(group[name], dtype=np.float32) for name in ETA_COLUMNS],
        axis=1,
    )


def load_auriga_snapshot(
    path: str | Path,
    *,
    group: str | None = None,
) -> AurigaSnapshot:
    """Load a standardized file or an Auriga/Gadget stellar-particle group.

    Group selection is automatic when ``group`` is omitted: a root-level
    ``eta`` takes precedence, followed by ``PartType4``.
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if group is not None:
            if group not in handle:
                raise KeyError(f"Group {group!r} not found in {path}.")
            selected = handle[group]
            if not isinstance(selected, h5py.Group):
                raise TypeError(f"{group!r} in {path} is not an HDF5 group.")
        elif "eta" in handle:
            selected = handle
        elif "PartType4" in handle:
            selected = handle["PartType4"]
        else:
            selected = handle

        eta = _load_eta(selected)
        attrs = {
            str(key): _decode_attr(value)
            for key, value in handle.attrs.items()
        }
        if "Header" in handle and isinstance(handle["Header"], h5py.Group):
            attrs.update(
                {
                    f"header_{key}": _decode_attr(value)
                    for key, value in handle["Header"].attrs.items()
                }
            )
        attrs.update(
            {
                str(key): _decode_attr(value)
                for key, value in selected.attrs.items()
            }
        )
        attrs.setdefault("source_file", str(path.resolve()))
        attrs.setdefault("source_group", selected.name)

        snapshot = AurigaSnapshot(
            eta=eta,
            particle_id=_first_dataset(selected, _ID_NAMES),
            mass=_first_dataset(selected, _MASS_NAMES),
            potential=_first_dataset(selected, _POTENTIAL_NAMES),
            acceleration=_first_dataset(selected, _ACCELERATION_NAMES),
            source_index=_first_dataset(selected, _SOURCE_INDEX_NAMES),
            tracer_weight=_first_dataset(selected, _TRACER_WEIGHT_NAMES),
            component=_first_dataset(selected, _COMPONENT_NAMES),
            circularity=_first_dataset(selected, _CIRCULARITY_NAMES),
            attrs=attrs,
        )

    source_index = snapshot.source_index
    if source_index is None:
        source_index = np.arange(snapshot.n_particles, dtype=np.int64)
    return replace(
        snapshot,
        eta=np.asarray(snapshot.eta, dtype=np.float32),
        particle_id=(
            None
            if snapshot.particle_id is None
            else np.asarray(snapshot.particle_id)
        ),
        mass=(
            None
            if snapshot.mass is None
            else np.asarray(snapshot.mass, dtype=np.float32).reshape(-1)
        ),
        potential=(
            None
            if snapshot.potential is None
            else np.asarray(snapshot.potential, dtype=np.float32).reshape(-1)
        ),
        acceleration=(
            None
            if snapshot.acceleration is None
            else np.asarray(snapshot.acceleration, dtype=np.float32)
        ),
        source_index=np.asarray(source_index, dtype=np.int64).reshape(-1),
        tracer_weight=(
            None
            if snapshot.tracer_weight is None
            else np.asarray(snapshot.tracer_weight, dtype=np.float32).reshape(-1)
        ),
        component=(
            None
            if snapshot.component is None
            else np.asarray(snapshot.component, dtype=np.int8).reshape(-1)
        ),
        circularity=(
            None
            if snapshot.circularity is None
            else np.asarray(snapshot.circularity, dtype=np.float32).reshape(-1)
        ),
    ).validate()


def center_snapshot(
    snapshot: AurigaSnapshot,
    *,
    position_center: Sequence[float] | None = None,
    velocity_center: Sequence[float] | None = None,
) -> AurigaSnapshot:
    """Translate the snapshot to an explicitly recorded reference frame."""
    eta = np.array(snapshot.eta, dtype=np.float32, copy=True)
    attrs = dict(snapshot.attrs)
    if position_center is not None:
        position_center_array = np.asarray(position_center, dtype=np.float32)
        if position_center_array.shape != (3,):
            raise ValueError("position_center must contain exactly 3 values.")
        eta[:, :3] -= position_center_array
        attrs["position_center"] = position_center_array.tolist()
    if velocity_center is not None:
        velocity_center_array = np.asarray(velocity_center, dtype=np.float32)
        if velocity_center_array.shape != (3,):
            raise ValueError("velocity_center must contain exactly 3 values.")
        eta[:, 3:] -= velocity_center_array
        attrs["velocity_center"] = velocity_center_array.tolist()
    return replace(snapshot, eta=eta, attrs=attrs).validate()


def classify_kinematic_components(
    snapshot: AurigaSnapshot,
    *,
    bins: int = 101,
) -> AurigaSnapshot:
    """Recompute Zhu-style phase-space averaged circularity labels.

    Particles are equal-population ranked in binding energy, angular momentum
    around z, and total angular momentum.  Means within those 3D rank cells
    define the circularity proxy.  The potential may contain an arbitrary
    additive offset because only its rank is used.
    """
    if snapshot.potential is None:
        raise ValueError(
            "Kinematic classification requires a per-particle potential."
        )
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    eta = np.asarray(snapshot.eta, dtype=np.float64)
    potential = np.asarray(snapshot.potential, dtype=np.float64)
    position = eta[:, :3]
    velocity = eta[:, 3:]
    n = eta.shape[0]

    v2 = np.sum(velocity**2, axis=1)
    binding_energy = potential + 0.5 * v2
    angular_momentum = np.cross(position, velocity)
    lz = angular_momentum[:, 2]
    angular_momentum_norm = np.linalg.norm(angular_momentum, axis=1)
    radius = np.linalg.norm(position, axis=1)

    def rank_bin(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n, dtype=np.int64)
        return np.minimum((ranks * int(bins)) // n, int(bins) - 1)

    energy_bin = rank_bin(binding_energy)
    lz_bin = rank_bin(lz)
    angular_momentum_bin = rank_bin(angular_momentum_norm)
    flat_bin = (
        (energy_bin * int(bins) + lz_bin) * int(bins)
        + angular_momentum_bin
    )
    n_cells = int(bins) ** 3
    count = np.bincount(flat_bin, minlength=n_cells)

    def cell_mean(values: np.ndarray) -> np.ndarray:
        total = np.bincount(flat_bin, weights=values, minlength=n_cells)
        means = np.divide(
            total,
            count,
            out=np.zeros_like(total, dtype=np.float64),
            where=count > 0,
        )
        return means[flat_bin]

    radius_mean = cell_mean(radius)
    lz_mean = cell_mean(lz)
    v2_mean = cell_mean(v2)
    denominator = radius_mean * np.sqrt(np.maximum(v2_mean, 0.0))
    circularity = np.divide(
        lz_mean,
        denominator,
        out=np.zeros_like(lz_mean),
        where=denominator > 0,
    )
    disk = (radius_mean > 2.0) & (radius_mean < 20.0)
    if np.sum(circularity[disk]) < 0:
        circularity = -circularity

    component = np.full(n, 2, dtype=np.int8)
    component[circularity > 0.8] = 0
    component[(circularity > 0.25) & (circularity <= 0.8)] = 1
    component[circularity < -0.25] = 3
    attrs = dict(snapshot.attrs)
    attrs.update(
        {
            "component_definition": (
                "ranked (Potential+0.5*v^2, Lz, |L|) circularity"
            ),
            "component_bins_per_dimension": int(bins),
        }
    )
    return replace(
        snapshot,
        component=component,
        circularity=circularity.astype(np.float32),
        attrs=attrs,
    ).validate()


def select_snapshot(
    snapshot: AurigaSnapshot,
    *,
    r_min: float | None = None,
    r_max: float | None = None,
    components: Sequence[int | str] | None = None,
    max_particles: int | None = None,
    seed: int = 0,
) -> AurigaSnapshot:
    """Apply component/radial cuts and optional reproducible downsampling."""
    radius = np.linalg.norm(np.asarray(snapshot.eta)[:, :3], axis=1)
    mask = np.ones(snapshot.n_particles, dtype=bool)
    if r_min is not None:
        if r_min < 0:
            raise ValueError("r_min must be non-negative.")
        mask &= radius >= float(r_min)
    if r_max is not None:
        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        mask &= radius <= float(r_max)
    component_values: list[int] | None = None
    if components is not None:
        if snapshot.component is None:
            raise ValueError(
                "Component selection requested, but no component labels exist."
            )
        component_values = []
        for value in components:
            if isinstance(value, str):
                key = value.strip().lower().replace("-", "_").replace("/", "_")
                if key == "all":
                    component_values = None
                    break
                if key not in KINEMATIC_COMPONENTS:
                    valid = ", ".join(sorted(KINEMATIC_COMPONENTS))
                    raise ValueError(
                        f"Unknown component {value!r}; expected one of {valid}."
                    )
                component_values.append(KINEMATIC_COMPONENTS[key])
            else:
                component_values.append(int(value))
        if component_values is not None:
            if not component_values:
                raise ValueError("components must not be empty.")
            if not set(component_values).issubset({0, 1, 2, 3}):
                raise ValueError("Integer components must be in {0, 1, 2, 3}.")
            mask &= np.isin(snapshot.component, np.unique(component_values))
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Auriga selection removed every particle.")
    if max_particles is not None:
        if max_particles <= 0:
            raise ValueError("max_particles must be positive.")
        if indices.size > max_particles:
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(indices, size=int(max_particles), replace=False)
            )
    attrs = dict(snapshot.attrs)
    attrs.update(
        {
            "selection_r_min": r_min if r_min is not None else "none",
            "selection_r_max": r_max if r_max is not None else "none",
            "selection_components": (
                "all"
                if component_values is None
                else ",".join(str(value) for value in sorted(set(component_values)))
            ),
            "selection_max_particles": (
                max_particles if max_particles is not None else "none"
            ),
            "selection_seed": int(seed),
        }
    )
    return replace(snapshot.subset(indices), attrs=attrs)


def align_indices_by_id(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    """Map target particle IDs to unique source rows in ``O(N log N)``."""
    source_ids = np.asarray(source_ids).reshape(-1)
    target_ids = np.asarray(target_ids).reshape(-1)
    if source_ids.size == 0 or target_ids.size == 0:
        raise ValueError("Particle ID arrays must be non-empty.")

    order = np.argsort(source_ids, kind="mergesort")
    sorted_ids = source_ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("Source particle IDs are not unique.")
    target_sorted = np.sort(target_ids, kind="mergesort")
    if np.any(target_sorted[1:] == target_sorted[:-1]):
        raise ValueError("Target particle IDs are not unique.")

    positions = np.searchsorted(sorted_ids, target_ids)
    valid = positions < sorted_ids.size
    matched = np.zeros(target_ids.shape, dtype=bool)
    matched[valid] = sorted_ids[positions[valid]] == target_ids[valid]
    if not np.all(matched):
        missing = target_ids[~matched][:5].tolist()
        raise ValueError(
            f"{np.count_nonzero(~matched)} target particle IDs are absent "
            f"from the source; first missing IDs={missing}."
        )
    return order[positions]


def _tolerance_vector(
    atol: float | Sequence[float],
) -> np.ndarray:
    values = np.asarray(atol, dtype=np.float64)
    if values.ndim == 0:
        values = np.repeat(values, 6)
    if values.shape != (6,):
        raise ValueError("atol must be a scalar or contain exactly 6 values.")
    if np.any(values <= 0):
        raise ValueError("Every atol value must be positive.")
    return values


def align_indices_by_eta(
    source_eta: np.ndarray,
    target_eta: np.ndarray,
    *,
    atol: float | Sequence[float],
    workers: int = -1,
) -> np.ndarray:
    """Map target phase-space rows to unique source rows within tolerance.

    Each dimension is scaled by its own tolerance before an infinity-norm
    nearest-neighbour query.  A second neighbour inside the tolerance box is
    treated as an ambiguity instead of silently choosing one.
    """
    source_eta = np.asarray(source_eta, dtype=np.float64)
    target_eta = np.asarray(target_eta, dtype=np.float64)
    if source_eta.ndim != 2 or source_eta.shape[1] != 6:
        raise ValueError(
            f"Expected source_eta shape (N, 6), got {source_eta.shape}."
        )
    if target_eta.ndim != 2 or target_eta.shape[1] != 6:
        raise ValueError(
            f"Expected target_eta shape (N, 6), got {target_eta.shape}."
        )
    if source_eta.shape[0] < 2:
        raise ValueError("Coordinate alignment requires at least 2 source rows.")

    tolerance = _tolerance_vector(atol)
    source_scaled = source_eta / tolerance[None, :]
    target_scaled = target_eta / tolerance[None, :]
    tree = cKDTree(source_scaled)
    distance, indices = tree.query(
        target_scaled,
        k=2,
        p=np.inf,
        distance_upper_bound=1.0 + 1.0e-12,
        workers=int(workers),
    )
    missing = ~np.isfinite(distance[:, 0])
    ambiguous = np.isfinite(distance[:, 1])
    if np.any(missing):
        raise ValueError(
            f"{np.count_nonzero(missing)} target eta rows have no source "
            "match within tolerance."
        )
    if np.any(ambiguous):
        raise ValueError(
            f"{np.count_nonzero(ambiguous)} target eta rows have multiple "
            "source matches within tolerance."
        )

    nearest = indices[:, 0].astype(np.int64, copy=False)
    delta = np.abs(source_eta[nearest] - target_eta)
    if not np.all(delta <= tolerance[None, :] * (1.0 + 1.0e-12)):
        raise RuntimeError("KD-tree match failed the per-dimension tolerance check.")
    if np.unique(nearest).size != nearest.size:
        raise ValueError("Multiple target rows map to the same source particle.")
    return nearest


def align_snapshot(
    source: AurigaSnapshot,
    target: AurigaSnapshot,
    *,
    atol: float | Sequence[float] | None = None,
    workers: int = -1,
) -> tuple[AurigaSnapshot, str]:
    """Reorder ``source`` to ``target`` using IDs first, then 6D coordinates."""
    if source.particle_id is not None and target.particle_id is not None:
        indices = align_indices_by_id(source.particle_id, target.particle_id)
        method = "particle_id"
    else:
        if atol is None:
            raise ValueError(
                "Particle IDs are unavailable in one of the files. Provide "
                "a scalar or six-value atol for explicit 6D coordinate matching."
            )
        indices = align_indices_by_eta(
            source.eta,
            target.eta,
            atol=atol,
            workers=workers,
        )
        method = "eta_kdtree"

    aligned = source.subset(indices)
    attrs = dict(aligned.attrs)
    attrs["alignment_method"] = method
    attrs["alignment_target_rows"] = int(target.n_particles)
    aligned = replace(aligned, attrs=attrs)
    if not np.allclose(
        aligned.eta,
        target.eta,
        rtol=0.0,
        atol=0.0 if atol is None else np.asarray(atol),
    ):
        raise ValueError("Aligned source eta does not match target eta.")
    return aligned, method


def _write_attr(handle: h5py.File, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (dict, set)):
        value = str(value)
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    try:
        handle.attrs[key] = value
    except (TypeError, ValueError):
        handle.attrs[key] = str(value)


def save_auriga_snapshot(
    snapshot: AurigaSnapshot,
    path: str | Path,
    *,
    length_unit: str = "unknown",
    velocity_unit: str = "unknown",
    potential_unit: str = "unknown",
    acceleration_unit: str = "unknown",
) -> Path:
    """Write the canonical Auriga mock HDF5 contract."""
    snapshot.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = AURIGA_SCHEMA
        handle.attrs["eta_columns"] = np.asarray(ETA_COLUMNS, dtype="S")
        handle.attrs["length_unit"] = length_unit
        handle.attrs["velocity_unit"] = velocity_unit
        handle.attrs["potential_unit"] = potential_unit
        handle.attrs["acceleration_unit"] = acceleration_unit
        for key, value in snapshot.attrs.items():
            if key not in handle.attrs:
                _write_attr(handle, str(key), value)

        handle.create_dataset(
            "eta",
            data=np.asarray(snapshot.eta, dtype=np.float32),
            compression="lzf",
            chunks=True,
        )
        fields = {
            "particle_id": snapshot.particle_id,
            "mass": snapshot.mass,
            "potential": snapshot.potential,
            "acceleration": snapshot.acceleration,
            "source_index": snapshot.source_index,
            "tracer_weight": snapshot.tracer_weight,
            "component": snapshot.component,
            "circularity": snapshot.circularity,
        }
        for name, value in fields.items():
            if value is not None:
                handle.create_dataset(
                    name,
                    data=np.asarray(value),
                    compression="lzf",
                    chunks=True,
                )
    return path
