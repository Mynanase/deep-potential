from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.data import (
    Normalizer,
    load_run_preprocessing,
    require_physics_compatible_transform,
)
from dpjax.models.potential import load_phi
from dpjax.physics.units import (
    gravitational_constant_for_system,
    summarize_density_sign,
)
from dpjax.plotting.diagnostics import plot_laplacian_density_diagnostics


def _load_physics_normalizer(df_run_dir: Path) -> Normalizer:
    normalizer, coordinate_transform = load_run_preprocessing(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Phi slice rendering",
    )
    return normalizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phi/rho/acc slices (TF-free) for JAX potential.")
    parser.add_argument("--df-run-dir", type=str, required=True)
    parser.add_argument("--phi-run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=5.0)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument(
        "--system",
        choices=["generic", "plummer", "halo"],
        default="generic",
    )
    parser.add_argument("--gravitational-constant", type=float, default=None)

    args = parser.parse_args()

    df_run_dir = Path(args.df_run_dir)
    phi_run_dir = Path(args.phi_run_dir)

    normalizer = _load_physics_normalizer(df_run_dir)
    phi_model, phi_params, _ = load_phi(phi_run_dir)

    out_dir = Path(args.out_dir) if args.out_dir else (phi_run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)
    density_g = gravitational_constant_for_system(
        args.system,
        args.gravitational_constant,
    )

    # Grid in physical coords
    rmax = float(args.rmax)
    grid = int(args.grid)
    xs = np.linspace(-rmax, rmax, grid, dtype=np.float32)
    ys = np.linspace(-rmax, rmax, grid, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    xyz = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(args.z), dtype=np.float32)], axis=-1)
    xyz_std = (xyz - mean_x[None, :]) / std_x[None, :]

    def phi_single(xi: jnp.ndarray) -> jnp.ndarray:
        return phi_model.apply({"params": phi_params}, xi)

    grad_fn = jax.grad(phi_single)
    hess_fn = jax.hessian(phi_single)

    @jax.jit
    def eval_batch(x_std_b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        phi_b = jax.vmap(phi_single)(x_std_b)
        grad_std_b = jax.vmap(grad_fn)(x_std_b)  # dPhi/dx_std
        hess_std_b = jax.vmap(hess_fn)(x_std_b)  # d2Phi/dx_std^2

        # Convert derivatives to physical units
        grad_phys_b = grad_std_b / jnp.asarray(std_x)[None, :]

        # Laplacian in physical coords: sum_i d2Phi/dx_i^2
        diag_std = jnp.stack([hess_std_b[:, 0, 0], hess_std_b[:, 1, 1], hess_std_b[:, 2, 2]], axis=-1)
        lap_phys_b = jnp.sum(diag_std / (jnp.asarray(std_x) ** 2)[None, :], axis=-1)

        acc_mag_b = jnp.linalg.norm(-grad_phys_b, axis=-1)
        rho_b = lap_phys_b / (4.0 * jnp.pi * density_g)

        return phi_b, acc_mag_b, rho_b

    # Batched evaluation
    batch = int(args.batch)
    n = xyz_std.shape[0]
    phi_all: list[np.ndarray] = []
    acc_all: list[np.ndarray] = []
    rho_all: list[np.ndarray] = []

    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        phi_b, acc_b, rho_b = eval_batch(jnp.asarray(xyz_std[sl]))
        phi_all.append(np.asarray(phi_b, dtype=np.float32))
        acc_all.append(np.asarray(acc_b, dtype=np.float32))
        rho_all.append(np.asarray(rho_b, dtype=np.float32))

    phi_img = np.concatenate(phi_all).reshape(X.shape)
    acc_img = np.concatenate(acc_all).reshape(X.shape)
    rho_img = np.concatenate(rho_all).reshape(X.shape)

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import colors

    # Phi (mean-subtracted)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=150)
    phi0 = phi_img - np.nanmean(phi_img)
    vmin, vmax = np.nanpercentile(phi0, [1, 99])
    if vmin * vmax < 0:
        divnorm = colors.TwoSlopeNorm(vcenter=0.0, vmin=float(vmin), vmax=float(vmax))
        im = ax.imshow(phi0, extent=[-rmax, rmax, -rmax, rmax], origin="lower", cmap="seismic", norm=divnorm)
    else:
        im = ax.imshow(phi0, extent=[-rmax, rmax, -rmax, rmax], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\Phi(x,y)$ (mean-subtracted)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_slice_xy.png")
    plt.close(fig)

    # rho (signed if needed)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=150)
    finite_rho = rho_img[np.isfinite(rho_img)]
    density_summary = summarize_density_sign(rho_img)
    if np.any(finite_rho < 0.0):
        rho_scale = max(
            float(np.percentile(np.abs(finite_rho), 99.0)),
            1.0e-12,
        )
        rho_nonzero = np.abs(finite_rho[finite_rho != 0.0])
        rho_linthresh = (
            max(
                float(np.percentile(rho_nonzero, 10.0)),
                rho_scale * 1.0e-4,
                1.0e-12,
            )
            if rho_nonzero.size
            else rho_scale * 1.0e-4
        )
        im = ax.imshow(
            rho_img,
            extent=[-rmax, rmax, -rmax, rmax],
            origin="lower",
            cmap="coolwarm",
            norm=colors.SymLogNorm(
                linthresh=rho_linthresh,
                vmin=-rho_scale,
                vmax=rho_scale,
            ),
        )
    else:
        positive = np.ma.masked_less_equal(rho_img, 0.0)
        positive_values = finite_rho[finite_rho > 0.0]
        if positive_values.size:
            rho_vmin, rho_vmax = np.percentile(
                positive_values,
                [5.0, 99.0],
            )
        else:
            rho_vmin, rho_vmax = 1.0e-12, 1.0e-11
        im = ax.imshow(
            positive,
            extent=[-rmax, rmax, -rmax, rmax],
            origin="lower",
            cmap="magma",
            norm=colors.LogNorm(
                vmin=max(float(rho_vmin), 1.0e-12),
                vmax=max(float(rho_vmax), 1.0e-11),
            ),
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    density_title = (
        r"$\rho_{\rm total}(x,y)=\nabla^2\Phi/(4\pi G)$"
        if args.system == "halo"
        else r"$\rho(x,y)=\nabla^2\Phi/(4\pi G)$"
    )
    ax.set_title(
        density_title
        + f"\nnegative pixels: {density_summary['negative_fraction']:.1%}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_slice_xy.png")
    plt.close(fig)
    (out_dir / "rho_slice_xy_summary.json").write_text(
        json.dumps(density_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_laplacian_density_diagnostics(
        xs,
        ys,
        rho_img,
        density_label=(
            r"$\rho_{\rm total}$"
            if args.system == "halo"
            else r"$\rho$"
        ),
        fig_dir=out_dir,
        fig_fmt=("png",),
        filename="rho_laplacian_diagnostics",
    )

    # |a|
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=150)
    im = ax.imshow(
        acc_img,
        extent=[-rmax, rmax, -rmax, rmax],
        origin="lower",
        cmap="cubehelix",
        norm=colors.LogNorm(vmin=np.nanpercentile(acc_img, 5), vmax=np.nanpercentile(acc_img, 99)),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$|a(x,y)|$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "accmag_slice_xy.png")
    plt.close(fig)

    np.savez(
        out_dir / "phi_slice_xy.npz",
        x=xs,
        y=ys,
        phi=phi_img,
        rho=rho_img,
        acc_mag=acc_img,
        density_gravitational_constant=np.asarray(density_g),
        density_semantics=np.asarray("total_gravitating_density"),
    )

    print(f"Wrote slice plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
