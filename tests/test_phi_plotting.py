from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.plotting import (
    plot_mass_density_profile,
    plot_mass_density_residual,
    plot_mass_density_slice,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
)


def test_phi_plotting_functions_return_independent_figures_without_io(tmp_path):
    import matplotlib.pyplot as plt

    radius = np.geomspace(0.1, 5.0, 32)
    truth_phi = -1.0 / np.sqrt(1.0 + radius**2)
    model_phi = truth_phi + 0.01 * np.sin(radius)
    truth_density = (1.0 + radius**2) ** -2.5
    model_density = truth_density.copy()
    model_density[7] *= -1.0
    truth_acceleration = -radius * (1.0 + radius**2) ** -1.5

    x = np.linspace(-2.0, 2.0, 17)
    y = np.linspace(-2.0, 2.0, 15)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    truth_slice = (1.0 + xx**2 + yy**2) ** -2.5
    model_slice = truth_slice.copy()
    model_slice[4:7, 5:8] *= -1.0
    mask = xx**2 + yy**2 >= 0.25

    figures = [
        plot_potential_profile(radius, model_phi, truth_potential=truth_phi),
        plot_radial_acceleration_profile(
            radius,
            truth_acceleration * 0.98,
            truth_acceleration=truth_acceleration,
        ),
        plot_mass_density_profile(
            radius,
            model_density,
            truth_density=truth_density,
        ),
        plot_mass_density_residual(radius, model_density, truth_density),
        plot_potential_slice(
            x,
            y,
            -1.0 / np.sqrt(1.0 + xx**2 + yy**2),
            truth_potential=-1.0 / np.sqrt(1.0 + xx**2 + yy**2),
            mask=mask,
        ),
        plot_mass_density_slice(
            x,
            y,
            model_slice,
            truth_density=truth_slice,
            mask=mask,
        ),
    ]

    assert all(figure.axes for figure in figures)
    assert list(tmp_path.iterdir()) == []
    for figure in figures:
        plt.close(figure)


def test_mass_density_slice_keeps_negative_model_values_visible():
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-1.0, 1.0, 5)
    density = np.ones((5, 7))
    density[2, 3] = -0.25

    figure = plot_mass_density_slice(x, y, density)

    norm = figure.axes[0].collections[0].norm
    assert norm.vmin < 0 < norm.vmax
    plt.close(figure)


def test_mass_density_slice_color_range_ignores_unsupported_cells():
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-1.0, 1.0, 5)
    density = np.ones((5, 7))
    density[2, 3] = -0.25
    # Extreme extrapolated speckle in a corner cell that receives no data.
    density[0, 0] = -1.0e6
    rng = np.random.default_rng(3)
    positions = rng.uniform(-0.4, 0.4, size=(500, 2))

    figure = plot_mass_density_slice(
        x,
        y,
        density,
        data_positions=positions,
        min_data_count=5,
    )

    norm = figure.axes[0].collections[0].norm
    # The unsupported corner outlier must not set the color range.
    assert abs(norm.vmax) < 1.0e3
    assert norm.vmin < 0 < norm.vmax
    # The unsupported region is grayed out (an extra contourf collection).
    assert len(figure.axes[0].collections) >= 2
    # Data-support radius reference circles are drawn.
    assert len(figure.axes[0].patches) == 3
    plt.close(figure)


def test_mass_density_slice_without_data_positions_keeps_legacy_behavior():
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-1.0, 1.0, 5)
    density = np.ones((5, 7))
    density[2, 3] = -0.25

    figure = plot_mass_density_slice(x, y, density)

    norm = figure.axes[0].collections[0].norm
    assert norm.vmin < 0 < norm.vmax
    plt.close(figure)
