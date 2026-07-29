"""Paper-style Halo12 potential and density-context figure.

The stellar-particle histogram is explicitly kept separate from the total
gravitating density inferred through the Poisson equation.  They are useful
together as morphological context, but are not a model/truth density pair.
"""
from __future__ import annotations

import json

import h5py
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dpjax.data import (
    load_run_preprocessing,
    require_physics_compatible_transform,
)
from dpjax.evaluation import (
    binned_potential_truth_by_phi,
    potential_error_metrics,
)
from dpjax.models.potential import (
    phi_apply,
    laplacian_phi_apply,
    load_phi,
)
from dpjax.physics.units import (
    G_KPC_KMS2_PER_MSUN,
    density_from_laplacian,
    summarize_density_sign,
)
from dpjax.plotting.diagnostics import (
    plot_auriga_potential_comparison,
    plot_laplacian_density_diagnostics,
    plot_potential_rz_by_phi,
)


def main():
    data_path = "data/auriga/halo12_all_mass.h5"
    df_run_dir = "runs/halo_12/df_ffjord_v23_mass/seed_43"
    phi_run_dir = "runs/halo_12/phi_mse_v3_seed43"
    out_dir = Path(phi_run_dir) / "plots"
    out_dir.mkdir(exist_ok=True)

    # Load normalizer from DF run
    normalizer, coordinate_transform = load_run_preprocessing(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Halo12 paper-figure rendering",
    )
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
    # Standardize positions for model input
    pos_std = (pos_sample - mean[:3]) / std_x

    # Phi model
    pot_model = np.asarray(phi_apply(phi_model, phi_params, jnp.asarray(pos_std)))
    # Align additive constant
    offset = np.median(pot_truth_sample - pot_model)
    pot_model_aligned = pot_model + offset
    potential_metrics, _ = potential_error_metrics(
        pot_model,
        pot_truth_sample,
    )
    plot_auriga_potential_comparison(
        pos_sample,
        pot_truth_sample,
        pot_model_aligned,
        metrics=potential_metrics,
        fig_dir=out_dir,
        fig_fmt=("png",),
        filename="phi_truth_comparison",
    )

    # Laplacian -> density
    lap_model = np.asarray(
        laplacian_phi_apply(
            phi_model, phi_params, jnp.asarray(pos_std), std_x=jnp.asarray(std_x)
        )
    )
    rho_total_model = density_from_laplacian(
        lap_model,
        gravitational_constant=G_KPC_KMS2_PER_MSUN,
    )

    # Stellar-tracer density is not the total-density truth.
    print("Computing stellar-tracer 3D density histogram...")
    bins = np.arange(-77.5, 78.5, 2.0)  # 2 kpc bins
    hist, edges = np.histogramdd(pos, bins=[bins, bins, bins], weights=mass)
    bin_vol = 2.0 ** 3  # kpc^3
    rho_star_3d = hist / bin_vol  # Msun/kpc^3

    # Interpolate truth density at sample points
    # Find bin indices for each sample point
    ix = np.clip(np.digitize(pos_sample[:, 0], edges[0]) - 1, 0, hist.shape[0] - 1)
    iy = np.clip(np.digitize(pos_sample[:, 1], edges[1]) - 1, 0, hist.shape[1] - 1)
    iz = np.clip(np.digitize(pos_sample[:, 2], edges[2]) - 1, 0, hist.shape[2] - 1)
    rho_star_sample = rho_star_3d[ix, iy, iz]

    print(
        f"rho_total_model: min={rho_total_model.min():.2f} "
        f"max={rho_total_model.max():.2f} "
        f"median={np.median(rho_total_model):.2f}"
    )
    print(
        f"rho_star: min={rho_star_sample.min():.2f} "
        f"max={rho_star_sample.max():.2f} "
        f"median={np.median(rho_star_sample):.2f}"
    )

    # ---- 2D slice at z=0 ----
    print("Computing 2D slices...")
    grid_edges = np.linspace(-75.0, 75.0, 201)
    grid_1d = 0.5 * (grid_edges[:-1] + grid_edges[1:])
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
    rho_total_grid = density_from_laplacian(
        lap_grid,
        gravitational_constant=G_KPC_KMS2_PER_MSUN,
    ).reshape(200, 200)
    density_summary = summarize_density_sign(rho_total_grid)
    (out_dir / "rho_laplacian_summary.json").write_text(
        json.dumps(density_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_laplacian_density_diagnostics(
        grid_1d,
        grid_1d,
        rho_total_grid.T,
        density_label=r"$\rho_{\rm total}$ [$M_\odot\,\mathrm{kpc}^{-3}$]",
        fig_dir=out_dir,
        fig_fmt=("png",),
        filename="rho_laplacian_diagnostics",
    )

    # Dense model R-z slices at azimuth-bin centers.  Simulator truth exists
    # only at particle positions, so its row is a per-cell median with sparse
    # cells left gray rather than interpolated.
    slice_phi_edges = np.linspace(-np.pi, np.pi, 7)
    slice_radius_edges = np.linspace(0.0, 75.0, 41)
    slice_z_edges = np.linspace(-75.0, 75.0, 41)
    truth_slices = binned_potential_truth_by_phi(
        pos,
        pot_truth,
        phi_edges=slice_phi_edges,
        cylindrical_radius_edges=slice_radius_edges,
        z_edges=slice_z_edges,
        min_cell_count=5,
    )
    slice_phi_centers = 0.5 * (
        slice_phi_edges[:-1] + slice_phi_edges[1:]
    )
    slice_radius_centers = 0.5 * (
        slice_radius_edges[:-1] + slice_radius_edges[1:]
    )
    slice_z_centers = 0.5 * (slice_z_edges[:-1] + slice_z_edges[1:])
    slice_radius_grid, slice_z_grid = np.meshgrid(
        slice_radius_centers,
        slice_z_centers,
        indexing="ij",
    )
    model_slices = np.empty((6, 40, 40), dtype=np.float64)
    for phi_index, phi_center in enumerate(slice_phi_centers):
        slice_positions = np.column_stack(
            [
                slice_radius_grid.ravel() * np.cos(phi_center),
                slice_radius_grid.ravel() * np.sin(phi_center),
                slice_z_grid.ravel(),
            ]
        )
        slice_positions_std = (
            slice_positions - mean[:3]
        ) / std_x
        model_slices[phi_index] = (
            np.asarray(
                phi_apply(
                    phi_model,
                    phi_params,
                    jnp.asarray(slice_positions_std, dtype=jnp.float32),
                )
            ).reshape(slice_radius_grid.shape)
            + offset
        )
    np.savez_compressed(
        out_dir / "phi_rz_by_phi.npz",
        phi_edges=slice_phi_edges,
        cylindrical_radius_edges=slice_radius_edges,
        z_edges=slice_z_edges,
        model_potential=model_slices,
        truth_potential=truth_slices["truth_median"],
        truth_count=truth_slices["count"],
        min_cell_count=truth_slices["min_cell_count"],
        fitted_additive_offset=np.asarray(offset),
    )
    plot_potential_rz_by_phi(
        slice_phi_edges,
        slice_radius_edges,
        slice_z_edges,
        model_slices,
        truth_potential=truth_slices["truth_median"],
        truth_count=truth_slices["count"],
        min_cell_count=5,
        fig_dir=out_dir,
        fig_fmt=("png",),
        filename="phi_rz_by_phi",
    )

    # Stellar-tracer density in a finite z slab.  The exact cell volume uses
    # the histogram edges rather than a hard-coded pixel size.
    z_half_width = 1.0
    z_mask = np.abs(pos[:, 2]) < z_half_width
    pos_slab = pos[z_mask]
    mass_slab = mass[z_mask]
    truth_2d, _, _ = np.histogram2d(
        pos_slab[:, 0],
        pos_slab[:, 1],
        bins=[grid_edges, grid_edges],
        weights=mass_slab,
    )
    dx = np.diff(grid_edges)[:, None]
    dy = np.diff(grid_edges)[None, :]
    stellar_cell_volume = dx * dy * (2.0 * z_half_width)
    rho_star_2d = truth_2d / stellar_cell_volume

    # ---- Plotting ----
    print("Plotting...")
    fig = plt.figure(figsize=(16, 10))

    # Layout: left column (3 rows, x=r), right column (2 panels side by side)
    ax1 = fig.add_axes([0.06, 0.68, 0.26, 0.23])  # top: Phi vs r
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

    # --- Left middle: total gravitating density inferred from Phi ---
    total_positive = rho_total_model > 0
    ax2.scatter(
        r_sample[total_positive],
        rho_total_model[total_positive],
        s=2,
        alpha=0.25,
        color="tab:cyan",
        label=r"$\nabla^2\Phi_{\rm model}/(4\pi G)$",
    )
    total_median = np.array(
        [
            np.median(
                rho_total_model[(r_idx == i) & total_positive]
            )
            if np.count_nonzero((r_idx == i) & total_positive) > 10
            else np.nan
            for i in range(1, len(r_bins))
        ]
    )
    ax2.plot(r_bc, total_median, color="tab:cyan", lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("r [kpc]")
    ax2.set_ylabel(r"$\rho_{\rm total,model}$ ($M_\odot$/kpc$^3$)")
    ax2.set_title("Total gravitating density inferred from model")
    ax2.legend(fontsize=8, markerscale=4)
    ax2.grid(True, alpha=0.3)

    # --- Left bottom: stellar tracer density (context, not total truth) ---
    stellar_positive = rho_star_sample > 0
    ax3.scatter(
        r_sample[stellar_positive],
        rho_star_sample[stellar_positive],
        s=2,
        alpha=0.25,
        color="tab:purple",
        label="stellar particles",
    )
    stellar_median = np.array(
        [
            np.median(rho_star_sample[(r_idx == i) & stellar_positive])
            if np.count_nonzero((r_idx == i) & stellar_positive) > 10
            else np.nan
            for i in range(1, len(r_bins))
        ]
    )
    ax3.plot(r_bc, stellar_median, color="tab:purple", lw=2)
    ax3.set_yscale("log")
    ax3.set_xlabel("r [kpc]")
    ax3.set_ylabel(r"$\rho_\star$ ($M_\odot$/kpc$^3$)")
    ax3.set_title("Stellar-tracer density (not total-density truth)")
    ax3.legend(fontsize=8, markerscale=4)
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

    # --- Right right: signed 2D rho slice with stellar-density contours ---
    # Negative Laplacians are kept visible.  Clipping them to a positive floor
    # would make sign failures look like ordinary low-density black pixels.
    rho_finite = rho_total_grid[np.isfinite(rho_total_grid)]
    rho_scale = max(
        float(np.percentile(np.abs(rho_finite), 99.0)),
        1.0e-12,
    )
    rho_nonzero = np.abs(rho_finite[rho_finite != 0.0])
    rho_linthresh = (
        max(
            float(np.percentile(rho_nonzero, 10.0)),
            rho_scale * 1.0e-4,
            1.0e-12,
        )
        if rho_nonzero.size
        else rho_scale * 1.0e-4
    )
    im5 = ax5.imshow(
        rho_total_grid.T,
        origin="lower",
        extent=[-75, 75, -75, 75],
        cmap="coolwarm",
        aspect="equal",
        norm=plt.matplotlib.colors.SymLogNorm(
            linthresh=rho_linthresh,
            vmin=-rho_scale,
            vmax=rho_scale,
        ),
    )
    # Stellar density contours provide morphology only; they are not truth
    # contours for the total gravitating density.
    truth_disp = np.clip(rho_star_2d.T, 1e-10, None)
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
    ax5.set_title(
        r"$\rho_{\rm total,model}(x,y,z{=}0)$"
        "\n"
        f"signed; negative pixels={density_summary['negative_fraction']:.1%}; "
        "cyan: stellar density"
    )
    plt.colorbar(im5, cax=cax5, orientation="horizontal")

    plt.suptitle(
        "Halo12: Potential Recovery and Density Context "
        "(total model density is not compared to stellar density as truth)",
        fontsize=14,
        y=0.99,
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
