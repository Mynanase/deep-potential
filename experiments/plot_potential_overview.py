from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

from dpjax.data import (
    Normalizer,
    load_eta_h5,
    load_run_preprocessing,
    require_physics_compatible_transform,
)
from dpjax.models.potential import grad_phi_apply, laplacian_phi_apply, load_phi, phi_apply
from dpjax.physics.analytic import plummer_ar, plummer_phi
from dpjax.plotting.diagnostics import plot_potential_density_overview


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_physics_normalizer(df_run_dir: Path) -> Normalizer:
    normalizer, coordinate_transform = load_run_preprocessing(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Potential overview rendering",
    )
    return normalizer


def _ensure_slice(
    df_run_dir: Path,
    phi_run_dir: Path,
    out_dir: Path,
    *,
    rmax: float,
    grid: int,
    batch: int,
    recompute: bool,
) -> Path:
    slice_path = out_dir / "phi_slice_xy.npz"
    if slice_path.exists() and not recompute:
        return slice_path

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "plot_phi_slice.py"),
        "--df-run-dir",
        str(df_run_dir),
        "--phi-run-dir",
        str(phi_run_dir),
        "--out-dir",
        str(out_dir),
        "--z",
        "0.0",
        "--rmax",
        str(float(rmax)),
        "--grid",
        str(int(grid)),
        "--batch",
        str(int(batch)),
    ]
    subprocess.run(cmd, check=True)
    if not slice_path.exists():
        raise FileNotFoundError(f"Expected slice cache was not created: {slice_path}")
    return slice_path


def _load_data_xy(data_path: Optional[str], dataset: str, n_points: int, seed: int) -> Optional[np.ndarray]:
    if data_path is None:
        return None
    eta = np.asarray(load_eta_h5(_resolve_path(data_path), dataset=dataset), dtype=np.float32)
    if eta.shape[0] > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(eta.shape[0], size=n_points, replace=False)
        eta = eta[idx]
    return eta[:, :2]


def _auto_rmax(data_xy: Optional[np.ndarray], fallback: float) -> float:
    if data_xy is None or data_xy.size == 0:
        return float(fallback)
    r_xy = np.sqrt(np.sum(data_xy**2, axis=1))
    return float(max(np.percentile(r_xy, 99.0), 1.0e-6))


def _compute_radial(
    df_run_dir: Path,
    phi_run_dir: Path,
    *,
    r_min: float,
    r_max: float,
    n_r: int,
    r_ref: float,
) -> dict[str, np.ndarray]:
    normalizer = _load_physics_normalizer(df_run_dir)
    phi_model, phi_params, _ = load_phi(phi_run_dir)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)

    r = np.geomspace(float(r_min), float(r_max), num=int(n_r)).astype(np.float32)
    x_phys = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1)
    x_std = (x_phys - mean_x[None, :]) / std_x[None, :]
    x_std_j = jnp.asarray(x_std)

    phi_learned = np.asarray(phi_apply(phi_model, phi_params, x_std_j), dtype=np.float32)
    grad_phi_std = np.asarray(grad_phi_apply(phi_model, phi_params, x_std_j), dtype=np.float32)
    grad_phi_phys = grad_phi_std / std_x[None, :]
    ar_learned = -grad_phi_phys[:, 0]
    lap_phys = np.asarray(laplacian_phi_apply(phi_model, phi_params, x_std_j, std_x=jnp.asarray(std_x)), dtype=np.float32)
    rho_learned = lap_phys / (4.0 * np.pi)

    i_ref = int(np.argmin(np.abs(r - float(r_ref))))
    phi_learned_shift = phi_learned - phi_learned[i_ref]
    return {
        "r": r,
        "phi_learned": phi_learned,
        "phi_learned_shift": phi_learned_shift,
        "rho_learned": rho_learned,
        "ar_learned": ar_learned,
    }


def _plummer_truth(radial: dict[str, np.ndarray], r_ref: float) -> dict[str, np.ndarray]:
    r = radial["r"]
    phi_true = plummer_phi(r)
    phi_ref = float(plummer_phi(np.array([float(r_ref)], dtype=np.float32))[0])
    i_ref = int(np.argmin(np.abs(r - float(r_ref))))
    radial["phi_learned_shift"] = radial["phi_learned"] - radial["phi_learned"][i_ref] + phi_ref
    return {
        "phi_true": phi_true,
        "rho_true": (3.0 / (4.0 * np.pi)) * (1.0 + r**2) ** (-2.5),
        "ar_true": plummer_ar(r),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a generic potential/density overview for Plummer or halo Phi runs.")
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--phi-run-dir", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="eta")
    parser.add_argument("--system", choices=["generic", "plummer", "halo"], default="generic")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--r-min", type=float, default=1.0e-3)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--r-ref", type=float, default=1.0)
    parser.add_argument("--n-r", type=int, default=256)
    parser.add_argument("--rmax-slice", type=float, default=None)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--n-contour-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute-slice", action="store_true")
    args = parser.parse_args()

    df_run_dir = _resolve_path(args.df_run_dir)
    phi_run_dir = _resolve_path(args.phi_run_dir)
    out_dir = _resolve_path(args.out_dir) if args.out_dir else phi_run_dir / "eval" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_xy = _load_data_xy(args.data, args.dataset, int(args.n_contour_points), int(args.seed))
    rmax_slice = float(args.rmax_slice) if args.rmax_slice is not None else _auto_rmax(data_xy, 5.0)
    r_max = float(args.r_max) if args.r_max is not None else rmax_slice

    radial = _compute_radial(
        df_run_dir,
        phi_run_dir,
        r_min=float(args.r_min),
        r_max=r_max,
        n_r=int(args.n_r),
        r_ref=float(args.r_ref),
    )
    truth = _plummer_truth(radial, float(args.r_ref)) if args.system == "plummer" else {}
    slice_path = _ensure_slice(
        df_run_dir,
        phi_run_dir,
        out_dir,
        rmax=rmax_slice,
        grid=int(args.grid),
        batch=int(args.batch),
        recompute=bool(args.recompute_slice),
    )
    slice_data = np.load(slice_path)

    title = "Plummer Potential / Density Overview" if args.system == "plummer" else "Halo Potential / Density Overview"
    filename = "plummer_potential_density_overview" if args.system == "plummer" else "halo_potential_density_overview"
    plot_potential_density_overview(
        radial["r"],
        radial["phi_learned_shift"],
        radial["rho_learned"],
        slice_data["x"],
        slice_data["y"],
        slice_data["phi"],
        slice_data["rho"],
        ar_learned=radial["ar_learned"],
        data_xy=data_xy,
        title=title,
        fig_dir=out_dir,
        fig_fmt=tuple(args.formats),
        dpi=int(args.dpi),
        filename=filename,
        **truth,
    )

    for fmt in args.formats:
        print(f"Wrote {out_dir / f'{filename}.{fmt}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
