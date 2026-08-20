"""Repository-specific, composable plotting surface."""

from experiments.plotting.df_diagnostics import (
    plot_cylindrical_rz_density,
    plot_cylindrical_marginals_by_radius,
    plot_density_profile,
    plot_radial_speed_comparison,
    plot_radial_speed_density,
    plot_score_field_rv,
    plot_score_slices_by_radius,
    plot_score_distribution,
    plot_velocity_marginals,
)
from experiments.plotting.phi_diagnostics import (
    plot_mass_density_profile,
    plot_mass_density_residual,
    plot_mass_density_slice,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
)
from experiments.plotting.registry import (
    FigureRegistry,
    FigureWriter,
    render_all,
    render_figure,
    write_figure,
)

__all__ = [
    "plot_cylindrical_rz_density",
    "plot_cylindrical_marginals_by_radius",
    "plot_density_profile",
    "plot_mass_density_profile",
    "plot_mass_density_residual",
    "plot_mass_density_slice",
    "plot_potential_profile",
    "plot_potential_slice",
    "plot_radial_speed_comparison",
    "plot_radial_speed_density",
    "plot_radial_acceleration_profile",
    "plot_score_distribution",
    "plot_score_field_rv",
    "plot_score_slices_by_radius",
    "plot_velocity_marginals",
    "FigureRegistry",
    "FigureWriter",
    "render_all",
    "render_figure",
    "write_figure",
]
