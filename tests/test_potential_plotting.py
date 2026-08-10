from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.plotting.diagnostics import (
    plot_auriga_potential_comparison,
    plot_laplacian_density_diagnostics,
    plot_potential_rz_by_phi,
)


def test_potential_truth_and_slice_plots(tmp_path):
    rng = np.random.default_rng(3)
    positions = rng.normal(size=(300, 3))
    truth = -100.0 / np.sqrt(1.0 + np.sum(positions**2, axis=1))
    model = truth + rng.normal(scale=0.5, size=truth.shape)
    plot_auriga_potential_comparison(
        positions,
        truth,
        model,
        metrics={"normalized_rmse": 0.01, "pearson_r": 0.99},
        radial_bins=8,
        fig_dir=tmp_path,
        fig_fmt=("png",),
        dpi=50,
    )

    phi_edges = np.linspace(-np.pi, np.pi, 4)
    radius_edges = np.linspace(0.0, 3.0, 5)
    z_edges = np.linspace(-2.0, 2.0, 6)
    shape = (3, 4, 5)
    model_slices = rng.normal(loc=-50.0, scale=5.0, size=shape)
    truth_slices = model_slices + rng.normal(scale=1.0, size=shape)
    truth_count = np.full(shape, 5)
    truth_count[0, 0, 0] = 0
    plot_potential_rz_by_phi(
        phi_edges,
        radius_edges,
        z_edges,
        model_slices,
        truth_potential=truth_slices,
        truth_count=truth_count,
        min_cell_count=3,
        fig_dir=tmp_path,
        fig_fmt=("png",),
        dpi=50,
    )

    assert (tmp_path / "potential_truth_comparison.png").exists()
    assert (tmp_path / "potential_rz_by_phi.png").exists()


def test_laplacian_density_plot_exposes_negative_pixels(tmp_path):
    x = np.linspace(-1.0, 1.0, 9)
    y = np.linspace(-1.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y)
    density = np.exp(-(xx**2 + yy**2))
    density[2:4, 3:5] *= -1.0

    plot_laplacian_density_diagnostics(
        x,
        y,
        density,
        fig_dir=tmp_path,
        fig_fmt=("png",),
        dpi=50,
    )

    assert (tmp_path / "laplacian_density_diagnostics.png").exists()
