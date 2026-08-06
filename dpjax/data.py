from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import h5py
import numpy as np


def load_eta_h5(path: str | Path, dataset: str = "eta") -> np.ndarray:
    """Load `eta` from an HDF5 file.

    Expected shape: (N, 6)
    Expected order: [x, y, z, vx, vy, vz]
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset!r} not found in {str(path)!r}.")
        eta = np.asarray(f[dataset])

    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")

    return eta.astype(np.float32, copy=False)


def load_h5_vector(path: str | Path, dataset: str) -> np.ndarray:
    """Load a row-aligned one-dimensional dataset from HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"Dataset {dataset!r} not found in {str(path)!r}.")
        values = np.asarray(handle[dataset], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(
            f"Expected {dataset!r} shape (N,), got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Dataset {dataset!r} contains NaN or Inf values.")
    return values


def save_eta_h5(
    eta: np.ndarray,
    path: str | Path,
    dataset: str = "eta",
) -> Path:
    """Validate and save phase-space data to a compressed HDF5 dataset."""
    eta = np.asarray(eta, dtype=np.float32)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset, data=eta, compression="lzf", chunks=True)
    return path


# ── Coordinate preprocessing transforms ────────────────────────────────

@dataclass(frozen=True)
class CoordinateTransform:
    """Configurable coordinate preprocessing applied *before* standardization.

    Supports ``none``, ``asinh``, ``log``, and ``power`` transforms on a
    subset of dimensions (e.g. spatial coords ``[0, 1, 2]``).
    """

    type: str  # "none", "asinh", "log", "power"
    dims: np.ndarray  # (D,) int – which dimensions to transform
    params: Dict[str, np.ndarray]  # backend-specific parameters

    def _get_slices(self, data: np.ndarray):
        """Return views of *data* on the transformed dims."""
        return data[:, self.dims]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply forward transform in-place and return *data*."""
        data = data.astype(np.float32, copy=False)
        if self.type == "none" or self.dims.size == 0:
            return data

        x = data[:, self.dims]
        if self.type == "asinh":
            scale = float(self.params["scale"])
            data[:, self.dims] = np.arcsinh(x / scale)
        elif self.type == "log":
            scale = float(self.params["scale"])
            data[:, self.dims] = np.sign(x) * np.log1p(np.abs(x) / scale)
        elif self.type == "power":
            alpha = float(self.params["alpha"])
            data[:, self.dims] = np.sign(x) * np.power(np.abs(x), alpha)
        else:
            raise ValueError(f"Unknown transform type {self.type!r}")
        return data

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse transform in-place and return *data*."""
        data = data.astype(np.float32, copy=False)
        if self.type == "none" or self.dims.size == 0:
            return data

        y = data[:, self.dims]
        if self.type == "asinh":
            scale = float(self.params["scale"])
            data[:, self.dims] = scale * np.sinh(y)
        elif self.type == "log":
            scale = float(self.params["scale"])
            data[:, self.dims] = np.sign(y) * scale * np.expm1(np.abs(y))
        elif self.type == "power":
            alpha = float(self.params["alpha"])
            data[:, self.dims] = np.sign(y) * np.power(np.abs(y), 1.0 / alpha)
        else:
            raise ValueError(f"Unknown transform type {self.type!r}")
        return data

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            schema_version=np.array(2, dtype=np.int32),
            type=self.type,
            dims=self.dims,
            **self.params,
        )

    @staticmethod
    def load_npz(path: str | Path) -> "CoordinateTransform":
        path = Path(path)
        with np.load(path) as data:
            schema_version = (
                int(data["schema_version"])
                if "schema_version" in data.files
                else 1
            )
            if schema_version < 2:
                warnings.warn(
                    f"Ignoring legacy coordinate transform {path}: files "
                    "without schema_version=2 were written by an implementation "
                    "that did not apply NumPy advanced-index transforms.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return CoordinateTransform(
                    type="none",
                    dims=np.array([], dtype=np.int32),
                    params={},
                )

            ttype = str(data["type"])
            dims = np.asarray(data["dims"], dtype=np.int32)
            # Remaining arrays are params
            params = {
                k: np.asarray(data[k])
                for k in data.files
                if k not in ("schema_version", "type", "dims")
            }
        return CoordinateTransform(type=ttype, dims=dims, params=params)

    @staticmethod
    def fit(data: np.ndarray, cfg: Dict[str, Any]) -> "CoordinateTransform":
        """Create a transform from raw *data* and a config dict.

        Expected *cfg* keys:
            - ``type``: ``"none" | "asinh" | "log" | "power"``
            - ``dims``: list of int (default ``[0, 1, 2]``)
            - ``scale``: ``"auto"`` or float (for ``asinh`` / ``log``)
            - ``alpha``: float (for ``power``)
        """
        ttype = str(cfg.get("type", "none")).lower()
        if ttype == "none":
            return CoordinateTransform(type="none", dims=np.array([], dtype=np.int32), params={})

        dims = np.array(cfg.get("dims", [0, 1, 2]), dtype=np.int32)
        if dims.size == 0:
            return CoordinateTransform(type="none", dims=dims, params={})

        # Clamp dims to valid range
        dims = np.clip(dims, 0, data.shape[1] - 1)
        x = data[:, dims]

        params: Dict[str, np.ndarray] = {}

        if ttype in ("asinh", "log"):
            scale_cfg = cfg.get("scale", "auto")
            if str(scale_cfg).lower() == "auto":
                scale = float(np.percentile(np.abs(x), 95.0))
                scale = max(scale, 1e-6)
            else:
                scale = float(scale_cfg)
            params["scale"] = np.array(scale, dtype=np.float32)
        elif ttype == "power":
            alpha = float(cfg.get("alpha", 0.5))
            params["alpha"] = np.array(alpha, dtype=np.float32)
        else:
            raise ValueError(f"Unknown transform type {ttype!r}")

        return CoordinateTransform(type=ttype, dims=dims, params=params)


# ── Standard normalizer ───────────────────────────────────────────────

@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray  # (6,)
    std: np.ndarray  # (6,)

    def transform(self, eta: np.ndarray) -> np.ndarray:
        eta = eta.astype(np.float32, copy=False)
        return (eta - self.mean) / self.std

    def inverse(self, eta_std: np.ndarray) -> np.ndarray:
        eta_std = eta_std.astype(np.float32, copy=False)
        return eta_std * self.std + self.mean

    def inverse_transform(self, eta_std: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for :meth:`inverse`."""
        return self.inverse(eta_std)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load_npz(path: str | Path) -> "Normalizer":
        path = Path(path)
        with np.load(path) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            std = np.asarray(data["std"], dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError(f"Invalid normalizer shapes: mean={mean.shape}, std={std.shape}")
        return Normalizer(mean=mean, std=std)


@dataclass(frozen=True)
class DFDataSelection:
    """Row selection used to train a DF from a source phase-space table.

    All indices refer to rows in the original source dataset.  Persisting this
    contract lets downstream CBE/Phi stages reuse the exact sigma-clipped
    support instead of evaluating DF scores on rows the flow never saw.
    """

    source_size: int
    dataset: str
    source_sha256: str
    clip_sigma: float
    split_seed: int
    support_indices: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray

    def __post_init__(self) -> None:
        source_size = int(self.source_size)
        if source_size < 1:
            raise ValueError("source_size must be positive.")
        dataset = str(self.dataset)
        if not dataset:
            raise ValueError("dataset must be non-empty.")
        source_sha256 = str(self.source_sha256).lower()
        if len(source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest.")

        arrays: dict[str, np.ndarray] = {}
        for name in ("support_indices", "train_indices", "val_indices"):
            values = np.asarray(getattr(self, name), dtype=np.int64)
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if values.size and (
                np.any(values < 0) or np.any(values >= source_size)
            ):
                raise ValueError(
                    f"{name} contains indices outside [0, {source_size})."
                )
            if np.unique(values).size != values.size:
                raise ValueError(f"{name} contains duplicate row indices.")
            arrays[name] = values

        support = arrays["support_indices"]
        train = arrays["train_indices"]
        val = arrays["val_indices"]
        if np.intersect1d(train, val).size:
            raise ValueError("train_indices and val_indices must be disjoint.")
        if not np.array_equal(
            np.sort(np.concatenate([train, val])),
            np.sort(support),
        ):
            raise ValueError(
                "train_indices and val_indices must partition support_indices."
            )

        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "clip_sigma", float(self.clip_sigma))
        object.__setattr__(self, "split_seed", int(self.split_seed))
        for name, values in arrays.items():
            object.__setattr__(self, name, values)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.array(1, dtype=np.int32),
            source_size=np.array(self.source_size, dtype=np.int64),
            dataset=np.array(self.dataset),
            source_sha256=np.array(self.source_sha256),
            clip_sigma=np.array(self.clip_sigma, dtype=np.float64),
            split_seed=np.array(self.split_seed, dtype=np.int64),
            support_indices=self.support_indices,
            train_indices=self.train_indices,
            val_indices=self.val_indices,
        )

    @staticmethod
    def load_npz(path: str | Path) -> "DFDataSelection":
        path = Path(path)
        with np.load(path) as data:
            schema_version = int(data["schema_version"])
            if schema_version != 1:
                raise ValueError(
                    f"Unsupported DF data-selection schema {schema_version} "
                    f"in {path}."
                )
            return DFDataSelection(
                source_size=int(data["source_size"]),
                dataset=str(data["dataset"]),
                source_sha256=str(data["source_sha256"]),
                clip_sigma=float(data["clip_sigma"]),
                split_seed=int(data["split_seed"]),
                support_indices=np.asarray(
                    data["support_indices"],
                    dtype=np.int64,
                ),
                train_indices=np.asarray(data["train_indices"], dtype=np.int64),
                val_indices=np.asarray(data["val_indices"], dtype=np.int64),
            )


def sigma_clip_mask(
    eta: np.ndarray,
    clip_sigma: float,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the row mask used by DF sigma clipping."""
    eta = np.asarray(eta, dtype=np.float32)
    clip_sigma = float(clip_sigma)
    if clip_sigma <= 0.0:
        return np.ones(eta.shape[0], dtype=bool)
    clip_normalizer = fit_normalizer(eta, weights=weights)
    clip_std = np.maximum(clip_normalizer.std, np.float32(1.0e-6))
    return np.all(
        np.abs(eta - clip_normalizer.mean)
        < clip_sigma * clip_std,
        axis=1,
    )


def phase_space_sha256(eta: np.ndarray) -> str:
    """Return a stable digest for the ordered float32 phase-space rows."""
    contiguous = np.ascontiguousarray(eta, dtype=np.float32)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def resolve_run_support_indices(
    run_dir: str | Path,
    eta: np.ndarray,
    *,
    data_config: Dict[str, Any],
    coordinate_transform: CoordinateTransform | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve the source rows on which a DF was trained.

    New runs use the persisted ``data_selection.npz`` contract.  For legacy
    runs, deterministically reconstruct the sigma-clip mask from the saved DF
    preprocessing and config so existing checkpoints remain usable.
    """
    run_dir = Path(run_dir)
    eta = np.asarray(eta, dtype=np.float32)
    selection_path = run_dir / "data_selection.npz"
    if selection_path.exists():
        selection = DFDataSelection.load_npz(selection_path)
        expected_dataset = str(data_config.get("dataset", "eta"))
        if selection.dataset != expected_dataset:
            raise ValueError(
                f"DF data selection targets dataset {selection.dataset!r}, but "
                f"the supplied config requests {expected_dataset!r}."
            )
        if selection.source_size != eta.shape[0]:
            raise ValueError(
                "DF data selection was created for "
                f"{selection.source_size} source rows, but the supplied data "
                f"contains {eta.shape[0]} rows."
            )
        supplied_sha256 = phase_space_sha256(eta)
        if selection.source_sha256 != supplied_sha256:
            raise ValueError(
                "The supplied phase-space rows do not match the ordered data "
                "used to create the DF selection (SHA-256 mismatch)."
            )
        return selection.support_indices, "persisted"

    clip_sigma = float(data_config.get("clip_sigma", 0.0))
    if clip_sigma <= 0.0:
        return np.arange(eta.shape[0], dtype=np.int64), "all_rows"

    transformed = np.array(eta, dtype=np.float32, copy=True)
    if coordinate_transform is not None:
        transformed = coordinate_transform.transform(transformed)
    mask = sigma_clip_mask(
        transformed,
        clip_sigma,
        weights=weights,
    )
    warnings.warn(
        f"Missing {selection_path}; reconstructed the DF sigma-clip support "
        "from its saved preprocessing and config. Retrain the DF to persist "
        "the exact row-selection contract.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.flatnonzero(mask).astype(np.int64, copy=False), "reconstructed"


def load_df_support_eta(
    data_path: str | Path,
    df_run_dir: str | Path,
    df_data_cfg: Dict[str, Any],
    *,
    coordinate_transform: CoordinateTransform | None = None,
) -> tuple[np.ndarray, str, int]:
    """Load the source rows on which a DF was trained (its support).

    Uses the persisted ``data_selection.npz`` contract when available; for
    legacy runs the sigma-clip support is reconstructed from the saved DF
    preprocessing and config.

    Returns ``(support_eta, support_source, source_n)`` where ``support_eta``
    is the raw (physical) phase-space rows restricted to the DF support,
    ``support_source`` is ``"persisted"`` or ``"reconstructed"``, and
    ``source_n`` is the total number of rows in the source dataset.
    """
    data_path = Path(data_path)
    df_run_dir = Path(df_run_dir)
    dataset = str(df_data_cfg.get("dataset", "eta"))
    eta = load_eta_h5(data_path, dataset=dataset)
    support_weights = None
    selection_path = df_run_dir / "data_selection.npz"
    if (
        not selection_path.exists()
        and float(df_data_cfg.get("clip_sigma", 0.0)) > 0.0
        and df_data_cfg.get("weight_dataset")
    ):
        support_weights = load_h5_vector(
            data_path,
            dataset=str(df_data_cfg["weight_dataset"]),
        )
    source_n = int(eta.shape[0])
    support_indices, support_source = resolve_run_support_indices(
        df_run_dir,
        eta,
        data_config=df_data_cfg,
        coordinate_transform=coordinate_transform,
        weights=support_weights,
    )
    return eta[support_indices], support_source, source_n


def load_run_preprocessing(
    run_dir: str | Path,
) -> tuple[Normalizer, CoordinateTransform | None]:
    """Load the normalizer and optional coordinate transform for a DF run."""
    run_dir = Path(run_dir)
    normalizer_path = run_dir / "normalizer.npz"
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Missing {normalizer_path}")

    coordinate_transform_path = run_dir / "coord_transform.npz"
    coordinate_transform = (
        CoordinateTransform.load_npz(coordinate_transform_path)
        if coordinate_transform_path.exists()
        else None
    )
    return Normalizer.load_npz(normalizer_path), coordinate_transform


def require_physics_compatible_transform(
    coordinate_transform: CoordinateTransform | None,
    *,
    operation: str,
) -> None:
    """Reject preprocessing whose physical-gradient chain rule is unsupported."""
    if coordinate_transform is not None and coordinate_transform.type != "none":
        raise ValueError(
            f"{operation} does not yet support nonlinear DF coordinate "
            f"transforms (got {coordinate_transform.type!r}). The physical-gradient "
            "chain rule must be implemented before using this DF run."
        )


def fit_normalizer(
    eta: np.ndarray,
    eps: float = 1.0e-6,
    weights: np.ndarray | None = None,
) -> Normalizer:
    eta = eta.astype(np.float32, copy=False)
    if weights is None:
        mean = np.mean(eta, axis=0, dtype=np.float64)
        variance = np.var(eta, axis=0, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (eta.shape[0],):
            raise ValueError(
                f"Expected weights shape ({eta.shape[0]},), got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("weights must be finite and strictly positive.")
        weight_sum = float(np.sum(weights))
        mean = np.sum(eta * weights[:, None], axis=0, dtype=np.float64) / weight_sum
        variance = (
            np.sum(
                weights[:, None] * (eta - mean[None, :]) ** 2,
                axis=0,
                dtype=np.float64,
            )
            / weight_sum
        )
    mean = mean.astype(np.float32)
    std = np.sqrt(variance).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return Normalizer(mean=mean, std=std)


def preprocess_eta(
    eta: np.ndarray,
    normalizer: Normalizer,
    coordinate_transform: CoordinateTransform | None = None,
) -> np.ndarray:
    """Apply optional coordinate preprocessing followed by standardization."""
    transformed = np.array(eta, dtype=np.float32, copy=True)
    if coordinate_transform is not None:
        transformed = coordinate_transform.transform(transformed)
    return normalizer.transform(transformed)


def inverse_preprocess_eta(
    eta_std: np.ndarray,
    normalizer: Normalizer,
    coordinate_transform: CoordinateTransform | None = None,
) -> np.ndarray:
    """Map standardized model coordinates back to physical phase space."""
    eta = np.array(normalizer.inverse(eta_std), dtype=np.float32, copy=True)
    if coordinate_transform is not None:
        eta = coordinate_transform.inverse(eta)
    return eta


def iter_batches(
    eta: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle: bool = True,
    drop_remainder: bool = True,
    max_batches: Optional[int] = None,
) -> Iterator[np.ndarray]:
    n = eta.shape[0]
    if shuffle:
        idx = rng.permutation(n)
        eta = eta[idx]

    n_full = n // batch_size
    n_batches = n_full if drop_remainder else int(np.ceil(n / batch_size))
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    for i in range(n_batches):
        lo = i * batch_size
        hi = lo + batch_size
        if hi > n:
            if drop_remainder:
                break
            hi = n
        yield eta[lo:hi]
