import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import yaml

    from experiments.diagnostics import (
        load_df_diagnostics,
        load_df_metrics,
        load_metrics,
        load_phi_diagnostics,
        load_phi_metrics,
        load_validation_diagnostics,
        load_validation_metrics,
        plot_training_metrics,
    )
    from experiments.paths import resolve_path
    from experiments.plotting import (
        plot_cylindrical_rz_density,
        plot_density_profile,
        plot_mass_density_profile,
        plot_mass_density_slice,
        plot_potential_profile,
        plot_potential_slice,
        plot_radial_acceleration_profile,
        plot_score_distribution,
        plot_velocity_marginals,
    )

    return (
        load_df_diagnostics,
        load_df_metrics,
        load_metrics,
        load_phi_diagnostics,
        load_phi_metrics,
        load_validation_diagnostics,
        load_validation_metrics,
        mo,
        np,
        plot_cylindrical_rz_density,
        plot_density_profile,
        plot_mass_density_profile,
        plot_mass_density_slice,
        plot_potential_profile,
        plot_potential_slice,
        plot_radial_acceleration_profile,
        plot_score_distribution,
        plot_training_metrics,
        plot_velocity_marginals,
        resolve_path,
        yaml,
    )


@app.cell
def _(mo):
    mo.md("# Halo12 result analysis")
    run_dir_input = mo.ui.text(
        value="runs/halo12/static-baseline",
        label="Experiment directory",
        full_width=True,
    )
    run_dir_input
    return (run_dir_input,)


@app.cell
def _(mo, resolve_path, run_dir_input, yaml):
    run_dir = resolve_path(run_dir_input.value)
    snapshot_path = run_dir / "run.yaml"
    mo.stop(
        not snapshot_path.is_file(),
        mo.callout(f"Missing run snapshot: `{snapshot_path}`", kind="warn"),
    )
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    mo.json(snapshot)
    return run_dir


@app.cell
def _(load_metrics, mo, plot_training_metrics, run_dir):
    _tabs = {}
    for _stage in ("df", "phi"):
        _path = run_dir / _stage / "metrics.csv"
        if _path.is_file():
            _figures = plot_training_metrics(load_metrics(_path), title=_stage)
            for _name, _figure in _figures.items():
                _tabs[f"{_stage}: {_name}"] = _figure
    _output = mo.ui.tabs(_tabs) if _tabs else mo.md("No training metrics yet.")
    mo.vstack([mo.md("## Training"), _output])


@app.cell
def _(
    load_df_diagnostics,
    load_df_metrics,
    mo,
    plot_cylindrical_rz_density,
    plot_density_profile,
    plot_score_distribution,
    plot_velocity_marginals,
    run_dir,
):
    _eval_dir = run_dir / "eval"
    mo.stop(
        not (_eval_dir / "df_diagnostics.npz").is_file(),
        mo.callout("No DF evaluation artifacts yet.", kind="info"),
    )
    _metrics = load_df_metrics(_eval_dir)
    _diagnostics = load_df_diagnostics(_eval_dir)
    _tabs = {
        "density": plot_density_profile(_diagnostics),
        "R-z density": plot_cylindrical_rz_density(_diagnostics),
        "velocity by r": plot_velocity_marginals(_diagnostics, "r"),
        "velocity by theta": plot_velocity_marginals(_diagnostics, "theta"),
        "velocity by phi": plot_velocity_marginals(_diagnostics, "phi"),
        "score": plot_score_distribution(_diagnostics),
    }
    mo.vstack([mo.md("## DF diagnostics"), mo.json(_metrics), mo.ui.tabs(_tabs)])


@app.cell
def _(
    load_phi_diagnostics,
    load_phi_metrics,
    mo,
    plot_mass_density_profile,
    plot_mass_density_slice,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
    run_dir,
):
    _eval_dir = run_dir / "eval"
    mo.stop(
        not (_eval_dir / "phi_diagnostics.npz").is_file(),
        mo.callout("No Phi evaluation artifacts yet.", kind="info"),
    )
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
    mo.vstack(
        [mo.md("## Phi diagnostics"), mo.json(_metrics), mo.ui.tabs(_figures)]
    )


@app.cell
def _(
    load_validation_diagnostics,
    load_validation_metrics,
    mo,
    np,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
    run_dir,
):
    _validation_dir = run_dir / "validation" / "auriga"
    mo.stop(
        not (_validation_dir / "diagnostics.npz").is_file(),
        mo.callout(
            "No separately generated Auriga truth artifacts are available.",
            kind="info",
        ),
    )
    _metrics = load_validation_metrics(_validation_dir)
    _diagnostics = load_validation_diagnostics(_validation_dir)
    _figures = {}
    _radius = _diagnostics["sample_radius"]
    if "sample_truth_potential" in _diagnostics:
        _figures["potential samples"] = plot_potential_profile(
            _radius,
            _diagnostics["sample_model_potential"],
            truth_potential=_diagnostics["sample_truth_potential"],
            model_kwargs={"linestyle": "none", "marker": ".", "alpha": 0.1},
            truth_kwargs={"linestyle": "none", "marker": ".", "alpha": 0.1},
        )
        if "slice_model_potential" in _diagnostics:
            _radius_edges = _diagnostics["slice_radius_edges"]
            _z_edges = _diagnostics["slice_z_edges"]
            _radius_centers = 0.5 * (_radius_edges[:-1] + _radius_edges[1:])
            _z_centers = 0.5 * (_z_edges[:-1] + _z_edges[1:])
            for _index in range(_diagnostics["slice_model_potential"].shape[0]):
                _figures[f"potential slice {_index}"] = plot_potential_slice(
                    _radius_centers,
                    _z_centers,
                    _diagnostics["slice_model_potential"][_index].T,
                    truth_potential=_diagnostics["slice_truth_potential"][_index].T,
                    mask=_diagnostics["slice_truth_mask"][_index].T,
                    x_label="R",
                    y_label="z",
                )
    if "sample_truth_acceleration" in _diagnostics:
        _position = _diagnostics["sample_position"]
        _safe_radius = np.maximum(_radius, 1.0e-12)
        _radial_direction = _position / _safe_radius[:, None]
        _model_radial = np.sum(
            _diagnostics["sample_model_acceleration"] * _radial_direction,
            axis=1,
        )
        _truth_radial = np.sum(
            _diagnostics["sample_truth_acceleration"] * _radial_direction,
            axis=1,
        )
        _figures["radial acceleration samples"] = (
            plot_radial_acceleration_profile(
                _radius,
                _model_radial,
                truth_acceleration=_truth_radial,
                model_kwargs={
                    "linestyle": "none",
                    "marker": ".",
                    "alpha": 0.1,
                },
                truth_kwargs={
                    "linestyle": "none",
                    "marker": ".",
                    "alpha": 0.1,
                },
            )
        )
    mo.vstack(
        [
            mo.md("## Optional simulator-truth validation"),
            mo.json(_metrics),
            mo.ui.tabs(_figures),
        ]
    )


if __name__ == "__main__":
    app.run()
