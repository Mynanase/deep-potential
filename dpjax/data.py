from __future__ import annotations

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
            x[:] = np.arcsinh(x / scale)
        elif self.type == "log":
            scale = float(self.params["scale"])
            x[:] = np.sign(x) * np.log1p(np.abs(x) / scale)
        elif self.type == "power":
            alpha = float(self.params["alpha"])
            x[:] = np.sign(x) * np.power(np.abs(x), alpha)
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
            y[:] = scale * np.sinh(y)
        elif self.type == "log":
            scale = float(self.params["scale"])
            y[:] = np.sign(y) * scale * np.expm1(np.abs(y))
        elif self.type == "power":
            alpha = float(self.params["alpha"])
            y[:] = np.sign(y) * np.power(np.abs(y), 1.0 / alpha)
        else:
            raise ValueError(f"Unknown transform type {self.type!r}")
        return data

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            type=self.type,
            dims=self.dims,
            **self.params,
        )

    @staticmethod
    def load_npz(path: str | Path) -> "CoordinateTransform":
        path = Path(path)
        with np.load(path) as data:
            ttype = str(data["type"])
            dims = np.asarray(data["dims"], dtype=np.int32)
            # Remaining arrays are params
            params = {
                k: np.asarray(data[k])
                for k in data.files
                if k not in ("type", "dims")
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


def fit_normalizer(eta: np.ndarray, eps: float = 1.0e-6) -> Normalizer:
    eta = eta.astype(np.float32, copy=False)
    mean = np.mean(eta, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(eta, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return Normalizer(mean=mean, std=std)


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
