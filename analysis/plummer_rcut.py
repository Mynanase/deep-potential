import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from experiments.diagnostics import (
        load_df_diagnostics,
        load_df_metrics,
        load_df_samples,
        load_phi_diagnostics,
        load_phi_metrics,
        load_validation_diagnostics,
        load_validation_metrics,
    )
    from experiments.paths import resolve_path
    from experiments.plotting import (
        plot_cylindrical_rz_density,
        plot_density_profile,
        plot_mass_density_profile,
        plot_mass_density_residual,
        plot_mass_density_slice,
        plot_potential_profile,
        plot_potential_slice,
        plot_radial_speed_comparison,
        plot_radial_acceleration_profile,
        plot_score_distribution,
        plot_velocity_marginals,
    )
    from experiments.validation.plummer import (
        plummer_phi,
        plummer_rv_ideal_grid,
    )

    return (
        load_df_diagnostics,
        load_df_metrics,
        load_df_samples,
        load_phi_diagnostics,
        load_phi_metrics,
        load_validation_diagnostics,
        load_validation_metrics,
        mo,
        np,
        plot_cylindrical_rz_density,
        plot_density_profile,
        plot_mass_density_profile,
        plot_mass_density_residual,
        plot_mass_density_slice,
        plot_potential_profile,
        plot_potential_slice,
        plot_radial_speed_comparison,
        plot_radial_acceleration_profile,
        plot_score_distribution,
        plot_velocity_marginals,
        plummer_phi,
        plummer_rv_ideal_grid,
        resolve_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Plummer r-cut result analysis

    This app only reads completed experiment artifacts. Choose any two
    experiment directories and compose the diagnostics below.
    """)
    full_dir_input = mo.ui.text(
        value="runs/plummer_rcut/full-baseline",
        label="Full experiment directory",
        full_width=True,
    )
    cut_dir_input = mo.ui.text(
        value="runs/plummer_rcut/cut-baseline",
        label="Cut experiment directory",
        full_width=True,
    )
    mo.vstack([full_dir_input, cut_dir_input])
    return cut_dir_input, full_dir_input


@app.cell
def _(cut_dir_input, full_dir_input, mo, resolve_path):
    _experiment_dirs = {
        "full": resolve_path(full_dir_input.value),
        "cut": resolve_path(cut_dir_input.value),
    }
    eval_dirs = {name: path / "eval" for name, path in _experiment_dirs.items()}
    _missing = [
        str(path / "df_metrics.json")
        for path in eval_dirs.values()
        if not (path / "df_metrics.json").is_file()
    ]
    mo.stop(
        _missing,
        mo.callout(
            "Missing evaluation artifacts:\n\n" + "\n\n".join(_missing),
            kind="warn",
        ),
    )
    return eval_dirs


@app.cell
def _(eval_dirs, load_df_diagnostics, load_df_metrics):
    df_metrics = {
        name: load_df_metrics(eval_dir)
        for name, eval_dir in eval_dirs.items()
    }
    df_diagnostics = {
        name: load_df_diagnostics(eval_dir)
        for name, eval_dir in eval_dirs.items()
    }
    return df_diagnostics, df_metrics


@app.cell
def _(eval_dirs, load_df_samples):
    df_samples = {
        name: load_df_samples(eval_dir)
        for name, eval_dir in eval_dirs.items()
        if (eval_dir / "df_samples.npz").is_file()
    }
    return (df_samples,)


@app.cell
def _(df_metrics, mo):
    _rows = []
    for _name, _metrics in df_metrics.items():
        _density = _metrics.get("density_profile", {})
        _row = {
            "experiment": _name,
            "n_data": _metrics.get("n_data"),
            "density_log10_rmse_dex": _density.get("log10_rmse_dex"),
            "median_fractional_error": _density.get("median_fractional_error"),
        }
        _rows.append(_row)
    mo.vstack([mo.md("## Metrics"), mo.ui.table(_rows)])


@app.cell
def _(df_diagnostics, mo, plot_density_profile):
    _density_tabs = {
        name: plot_density_profile(
            diagnostics,
            title=f"{name}: radial density",
        )
        for name, diagnostics in df_diagnostics.items()
    }
    mo.vstack([mo.md("## Radial density"), mo.ui.tabs(_density_tabs)])


@app.cell
def _(df_diagnostics, mo, plot_cylindrical_rz_density):
    _spatial_tabs = {
        name: plot_cylindrical_rz_density(
            diagnostics,
            title=f"{name}: cylindrical R-z density",
        )
        for name, diagnostics in df_diagnostics.items()
    }
    mo.vstack([mo.md("## Cylindrical density"), mo.ui.tabs(_spatial_tabs)])


@app.cell
def _(df_diagnostics, mo, plot_velocity_marginals):
    _velocity_tabs = {}
    for _coordinate in ("r", "theta", "phi"):
        _velocity_tabs[_coordinate] = mo.ui.tabs(
            {
                name: plot_velocity_marginals(
                    diagnostics,
                    _coordinate,
                    title=f"{name}: conditioned on {_coordinate}",
                )
                for name, diagnostics in df_diagnostics.items()
            }
        )
    mo.vstack([mo.md("## Velocity marginals"), mo.ui.tabs(_velocity_tabs)])


@app.cell
def _(df_diagnostics, mo, plot_score_distribution):
    _score_tabs = {
        name: plot_score_distribution(
            diagnostics,
            title=f"{name}: score distribution",
        )
        for name, diagnostics in df_diagnostics.items()
    }
    mo.vstack([mo.md("## Score distributions"), mo.ui.tabs(_score_tabs)])


@app.cell
def _(
    df_samples,
    mo,
    np,
    plot_radial_speed_comparison,
    plummer_phi,
    plummer_rv_ideal_grid,
):
    _radius = np.linspace(0.0, 5.0, 256)
    _escape_curve = (_radius, np.sqrt(-2.0 * plummer_phi(_radius)))
    _rv_tabs = {}
    for _name, _samples in df_samples.items():
        _radial_selection = (1.0, 10.0) if _name == "cut" else (0.0, 10.0)
        _ideal = plummer_rv_ideal_grid(
            r_lim=(0.0, 5.0),
            v_lim=(0.0, 1.5),
            bins=(48, 48),
            radial_selection=_radial_selection,
        )
        _rv_tabs[_name] = plot_radial_speed_comparison(
            None,
            _samples["model_eta_by_model"][0],
            reference_probability_mass=_ideal["probability_mass"],
            radius_range=(0.0, 5.0),
            speed_range=(0.0, 1.5),
            bins=(48, 48),
            escape_curve=_escape_curve,
            reference_label="Ideal DF",
            model_label="Normalizing flow",
            title=f"{_name}: radial phase-space distribution",
        )
    _output = (
        mo.ui.tabs(_rv_tabs)
        if _rv_tabs
        else mo.callout(
            "No df_samples.npz artifact is available. Re-run the DF evaluation "
            "with the current workflow to enable the radial phase-space plot.",
            kind="info",
        )
    )
    mo.vstack([mo.md("## Radial phase-space comparison"), _output])


@app.cell
def _(
    eval_dirs,
    load_phi_diagnostics,
    load_phi_metrics,
    mo,
    plot_mass_density_profile,
    plot_mass_density_slice,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
):
    _phi_tabs = {}
    for _name, _eval_dir in eval_dirs.items():
        if not (_eval_dir / "phi_diagnostics.npz").is_file():
            continue
        _metrics = load_phi_metrics(_eval_dir)
        _diagnostics = load_phi_diagnostics(_eval_dir)
        _figures = {
            "potential profile": plot_potential_profile(
                _diagnostics["radial_r"],
                _diagnostics["radial_phi_shifted"],
            ),
            "acceleration profile": plot_radial_acceleration_profile(
                _diagnostics["radial_r"],
                _diagnostics["radial_acceleration"],
            ),
            "density profile": plot_mass_density_profile(
                _diagnostics["radial_r"],
                _diagnostics["radial_density"],
            ),
        }
        if "slice_phi" in _diagnostics:
            _figures["potential slice"] = plot_potential_slice(
                _diagnostics["slice_x"],
                _diagnostics["slice_y"],
                _diagnostics["slice_phi"],
            )
            _figures["density slice"] = plot_mass_density_slice(
                _diagnostics["slice_x"],
                _diagnostics["slice_y"],
                _diagnostics["slice_density"],
            )
        _phi_tabs[_name] = mo.vstack(
            [mo.json(_metrics), mo.ui.tabs(_figures)]
        )
    _output = (
        mo.ui.tabs(_phi_tabs)
        if _phi_tabs
        else mo.callout("No Phi evaluation artifacts are available.", kind="info")
    )
    mo.vstack([mo.md("## Phi diagnostics"), _output])


@app.cell
def _(
    eval_dirs,
    load_validation_diagnostics,
    load_validation_metrics,
    mo,
    plot_mass_density_profile,
    plot_mass_density_residual,
    plot_mass_density_slice,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
):
    _validation_tabs = {}
    for _name, _eval_dir in eval_dirs.items():
        _validation_dir = _eval_dir.parent / "validation" / "plummer"
        if not (_validation_dir / "diagnostics.npz").is_file():
            continue
        _metrics = load_validation_metrics(_validation_dir)
        _diagnostics = load_validation_diagnostics(_validation_dir)
        _radius = _diagnostics["radial_r"]
        _mask = _diagnostics["slice_support_mask"]
        _figures = {
            "potential profile": plot_potential_profile(
                _radius,
                _diagnostics["radial_model_potential"],
                truth_potential=_diagnostics["radial_truth_potential"],
            ),
            "acceleration profile": plot_radial_acceleration_profile(
                _radius,
                _diagnostics["radial_model_acceleration"],
                truth_acceleration=_diagnostics["radial_truth_acceleration"],
            ),
            "density profile": plot_mass_density_profile(
                _radius,
                _diagnostics["radial_model_density"],
                truth_density=_diagnostics["radial_truth_density"],
            ),
            "density residual": plot_mass_density_residual(
                _radius,
                _diagnostics["radial_model_density"],
                _diagnostics["radial_truth_density"],
            ),
            "potential slice": plot_potential_slice(
                _diagnostics["slice_x"],
                _diagnostics["slice_y"],
                _diagnostics["slice_model_potential"],
                truth_potential=_diagnostics["slice_truth_potential"],
                mask=_mask,
            ),
            "density slice": plot_mass_density_slice(
                _diagnostics["slice_x"],
                _diagnostics["slice_y"],
                _diagnostics["slice_model_density"],
                truth_density=_diagnostics["slice_truth_density"],
                mask=_mask,
            ),
        }
        _validation_tabs[_name] = mo.vstack(
            [mo.json(_metrics), mo.ui.tabs(_figures)]
        )
    _output = (
        mo.ui.tabs(_validation_tabs)
        if _validation_tabs
        else mo.callout(
            "No separately generated Plummer truth artifacts are available.",
            kind="info",
        )
    )
    mo.vstack([mo.md("## Optional analytic-truth validation"), _output])


if __name__ == "__main__":
    app.run()
