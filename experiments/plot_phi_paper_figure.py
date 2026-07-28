"""Paper-style figure: potential & density comparison (Halo12).

Recreates the figure described in the Deep Potential paper:
  Left panels (3 stacked):
    - Phi_model vs Phi_truth scatter at random positions
    - rho_model vs rho_truth scatter at random positions
    - rho residuals (model - truth) vs truth
  Right panels (2):
    - 2D Phi_model slice at z=0
    - 2D rho_model slice at z=0 with truth density contours
"""
from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dpjax.data import load_run_preprocessing
from dpjax.models.potential import (
    phi_apply,
    laplacian_phi_apply,
    load_phi,
)

# Physical constant: G in (km/s)^2 kpc / Msun
G = 4.302e-6
FOUR_PI_G = 4.0 * np.pi * G


def main():
    data_path = "data/auriga/halo12_all_mass.h5"
    df_run_dir = "runs/halo_12/df_ffjord_v23_mass/seed_43"
    phi_run_dir = "runs/halo_12/phi_mse_v3_seed43"
    out_dir = Path(phi_run_dir) / "plots"
    out_dir.mkdir(exist_ok=True)

    # Load normalizer from DF run
    normalizer, _ = load_run_preprocessing(df_run_dir)
    mean = np.asarray(normalizer.mean)
    std = np.asarray(normalizer.std)
    std_x = std[:3]

    # Load Phi model
    phi_model, phi_params, _ = load_phi(phi_run_dir)

    # Load data
    with h5py.File(data_path, "r") as f:
        eta = f["eta"][:]          # (N, 6) physical units
        mass = f["mass"][:]         # (N,) Msun
        pot_truth = f["potential"][:]  # (N,) (km/s)^2

    pos = eta[:, :3]   # kpc
    N = len(pos)

    # ---- Scatter: sample 5000 random points ----
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=5000, replace=False)
    pos_sample = pos[idx]
    pot_truth_sample = pot_truth[idx]
    mass_sample = mass[idx]

    # Standardize positions for model input
    pos_std = (pos_sample - mean[:3]) / std_x

    # Phi model
    pot_model = np.asarray(phi_apply(phi_model, phi_params, jnp.asarray(pos_std)))
    # Align additive constant
    offset = np.median(pot_truth_sample - pot_model)
    pot_model_aligned = pot_model + offset

    # Laplacian -> density
    lap_model = np.asarray(
        laplacian_phi_apply(
            phi_model, phi_params, jnp.asarray(pos_std), std_x=jnp.asarray(std_x)
        )
    )
    rho_model = lap_model / FOUR_PI_G  # Msun/kpc^3

    # Truth density: 3D histogram of all particles weighted by mass
    print("Computing 3D density histogram...")
    bins = np.arange(-77.5, 78.5, 2.0)  # 2 kpc bins
    hist, edges = np.histogramdd(pos, bins=[bins, bins, bins], weights=mass)
    bin_vol = 2.0 ** 3  # kpc^3
    rho_truth_3d = hist / bin_vol  # Msun/kpc^3

    # Interpolate truth density at sample points
    # Find bin indices for each sample point
    ix = np.clip(np.digitize(pos_sample[:, 0], edges[0]) - 1, 0, hist.shape[0] - 1)
    iy = np.clip(np.digitize(pos_sample[:, 1], edges[1]) - 1, 0, hist.shape[1] - 1)
    iz = np.clip(np.digitize(pos_sample[:, 2], edges[2]) - 1, 0, hist.shape[2] - 1)
    rho_truth_sample = rho_truth_3d[ix, iy, iz]

    print(
        f"rho_model:  min={rho_model.min():.2f} max={rho_model.max():.2f} "
        f"median={np.median(rho_model):.2f}"
    )
    print(
        f"rho_truth:  min={rho_truth_sample.min():.2f} max={rho_truth_sample.max():.2f} "
        f"median={np.median(rho_truth_sample):.2f}"
    )

    # ---- 2D slice at z=0 ----
    print("Computing 2D slices...")
    grid_1d = np.linspace(-75, 75, 200)
    gx, gy = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    gz = np.zeros_like(gx)
    grid_pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    grid_std = (grid_pos - mean[:3]) / std_x

    # Phi on grid
    phi_grid = np.asarray(phi_apply(phi_model, phi_params, jnp.asarray(grid_std)))
    phi_grid = phi_grid.reshape(200, 200)

    # Laplacian on grid (batch to avoid OOM)
    lap_grid = np.zeros(len(grid_std))
    batch = 10000
    for i in range(0, len(grid_std), batch):
        sl = slice(i, min(i + batch, len(grid_std)))
        lap_grid[sl] = np.asarray(
            laplacian_phi_apply(
                phi_model,
                phi_params,
                jnp.asarray(grid_std[sl]),
                std_x=jnp.asarray(std_x),
            )
        )
    rho_grid = (lap_grid / FOUR_PI_G).reshape(200, 200)

    # Truth 2D density: histogram of particles near z=0
    z_mask = np.abs(pos[:, 2]) < 1.0  # ±1 kpc slab
    pos_slab = pos[z_mask]
    mass_slab = mass[z_mask]
    truth_2d, _, _ = np.histogram2d(
        pos_slab[:, 0], pos_slab[:, 1], bins=[grid_1d, grid_1d], weights=mass_slab
    )
    truth_2d = truth_2d / (4.0 * 4.0)  # bin area * z-slab width = 2*2*2 kpc^3

    # ---- Plotting ----
    print("Plotting...")
    fig = plt.figure(figsize=(16, 10))

    # Layout: left column (3 rows, x=r), right column (2 panels side by side)
    ax1 = fig.add_axes([0.06, 0.69, 0.26, 0.27])  # top: Phi vs r
    ax2 = fig.add_axes([0.06, 0.39, 0.26, 0.27])  # middle: rho vs r
    ax3 = fig.add_axes([0.06, 0.08, 0.26, 0.27])  # bottom: rho residual vs r
    ax4 = fig.add_axes([0.40, 0.39, 0.27, 0.55])  # Phi 2D slice
    ax5 = fig.add_axes([0.72, 0.39, 0.27, 0.55])  # rho 2D slice
    cax4 = fig.add_axes([0.40, 0.31, 0.27, 0.02])
    cax5 = fig.add_axes([0.72, 0.31, 0.27, 0.02])

    r_sample = np.sqrt((pos_sample ** 2).sum(axis=1))

    # Radial bins for median profiles
    r_bins = np.linspace(0, 75, 30)
    r_bc = 0.5 * (r_bins[:-1] + r_bins[1:])
    r_idx = np.digitize(r_sample, r_bins)

    # --- Left top: Phi vs r ---
    ax1.scatter(r_sample, pot_truth_sample, s=2, alpha=0.25, color="k", label="truth")
    ax1.scatter(r_sample, pot_model_aligned, s=2, alpha=0.25, color="tab:orange", label="model")
    # Median profiles
    t_med = np.array([np.median(pot_truth_sample[r_idx == i]) if (r_idx == i).sum() > 10 else np.nan for i in range(1, len(r_bins))])
    m_med = np.array([np.median(pot_model_aligned[r_idx == i]) if (r_idx == i).sum() > 10 else np.nan for i in range(1, len(r_bins))])
    ax1.plot(r_bc, t_med, "k-", lw=2)
    ax1.plot(r_bc, m_med, "--", color="tab:orange", lw=2)
    ax1.set_xlabel("r [kpc]")
    ax1.set_ylabel(r"$\Phi$ (km/s)$^2$")
    ax1.set_title("Potential vs radius")
    ax1.legend(fontsize=8, markerscale=4)
    ax1.grid(True, alpha=0.3)

    # --- Left middle: rho vs r ---
    pos_mask = rho_truth_sample > 0
    ax2.scatter(r_sample[pos_mask], rho_truth_sample[pos_mask], s=2, alpha=0.25, color="k", label="truth")
    ax2.scatter(r_sample[pos_mask], np.clip(rho_model[pos_mask], 1e-1, None), s=2, alpha=0.25, color="tab:cyan", label="model")
    # Median profiles (only positive model values)
    m_rho = np.array([np.median(np.clip(rho_model[(r_idx == i) & pos_mask], 1e-1, None)) if ((r_idx == i) & pos_mask).sum() > 10 else np.nan for i in range(1, len(r_bins))])
    t_rho = np.array([np.median(rho_truth_sample[(r_idx == i) & pos_mask]) if ((r_idx == i) & pos_mask).sum() > 10 else np.nan for i in range(1, len(r_bins))])
    ax2.plot(r_bc, t_rho, "k-", lw=2)
    ax2.plot(r_bc, m_rho, "--", color="tab:cyan", lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("r [kpc]")
    ax2.set_ylabel(r"$\rho$ ($M_\odot$/kpc$^3$)")
    ax2.set_title("Density vs radius")
    ax2.legend(fontsize=8, markerscale=4)
    ax2.grid(True, alpha=0.3)

    # --- Left bottom: rho residual vs r ---
    resid = rho_model[pos_mask] - rho_truth_sample[pos_mask]
    ax3.scatter(r_sample[pos_mask], resid, s=2, alpha=0.25, color="tab:red")
    ax3.axhline(0, color="k", ls="--", lw=1)
    # Median residual profile
    med_resid = np.array([np.median(resid[(r_idx == i)[pos_mask]]) if ((r_idx == i) & pos_mask).sum() > 10 else np.nan for i in range(1, len(r_bins))])
    ax3.plot(r_bc, med_resid, "-", color="darkred", lw=2)
    ax3.set_xlabel("r [kpc]")
    ax3.set_ylabel(r"$\rho_{\rm model} - \rho_{\rm truth}$")
    ax3.set_title("Density residuals vs radius")
    ax3.grid(True, alpha=0.3)

    # --- Right middle: 2D Phi slice ---
    im4 = ax4.imshow(
        phi_grid.T,
        origin="lower",
        extent=[-75, 75, -75, 75],
        cmap="RdBu_r",
        aspect="equal",
    )
    ax4.set_xlabel("x [kpc]")
    ax4.set_ylabel("y [kpc]")
    ax4.set_title(r"$\Phi_{\rm model}(x, y, z{=}0)$")
    plt.colorbar(im4, cax=cax4, orientation="horizontal")

    # --- Right right: 2D rho slice with truth contours ---
    # Clip rho for display (can have negative values from Laplacian noise)
    rho_disp = np.clip(rho_grid, 1e-2, None)
    im5 = ax5.imshow(
        rho_disp.T,
        origin="lower",
        extent=[-75, 75, -75, 75],
        cmap="magma",
        aspect="equal",
        norm=plt.matplotlib.colors.LogNorm(vmin=1e-1, vmax=rho_disp.max()),
    )
    # Truth density contours
    truth_disp = truth_2d.T
    truth_disp = np.clip(truth_disp, 1e-10, None)
    contour_levels = np.logspace(
        np.log10(truth_disp[truth_disp > 0].max() * 1e-3),
        np.log10(truth_disp[truth_disp > 0].max()),
        8,
    )
    ax5.contour(
        truth_disp,
        levels=contour_levels,
        extent=[-75, 75, -75, 75],
        colors="cyan",
        linewidths=0.8,
        alpha=0.7,
    )
    ax5.set_xlabel("x [kpc]")
    ax5.set_ylabel("y [kpc]")
    ax5.set_title(r"$\rho_{\rm model}(x, y, z{=}0)$ + truth contours")
    plt.colorbar(im5, cax=cax5, orientation="horizontal")

    plt.suptitle(
        "Halo12: Potential & Density Recovery (Phi MSE-v3, frozen DF seed_43)",
        fontsize=14,
        y=0.98,
    )

    out_path = out_dir / "phi_paper_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Also save a version with potential colorbar
    fig2, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(
        phi_grid.T,
        origin="lower",
        extent=[-75, 75, -75, 75],
        cmap="RdBu_r",
        aspect="equal",
    )
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(r"$\Phi_{\rm model}(x, y, z{=}0)$ (km/s)$^2$")
    plt.colorbar(im, ax=ax, label=r"$\Phi$ (km/s)$^2$")
    plt.tight_layout()
    fig2.savefig(out_dir / "phi_slice_2d.png", dpi=150)
    plt.close(fig2)
    print(f"saved {out_dir / 'phi_slice_2d.png'}")


if __name__ == "__main__":
    main()
