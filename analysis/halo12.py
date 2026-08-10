import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import yaml

    from experiments.diagnostics import (
        load_df_evaluation,
        load_metrics,
        load_phi_evaluation,
        plot_density_profile,
        plot_radial_curves,
        plot_training_metrics,
    )
    from experiments.paths import resolve_path

    return (
        Path,
        load_df_evaluation,
        load_metrics,
        load_phi_evaluation,
        mo,
        plot_density_profile,
        plot_radial_curves,
        plot_training_metrics,
        resolve_path,
        yaml,
    )


@app.cell
def _(mo):
    mo.md("""
    # Halo12 result analysis

    This notebook reads completed run artifacts. Training and expensive
    evaluation remain independent background jobs.
    """)


@app.cell
def _(mo):
    DEFAULT_RUN_DIR = "runs/halo12_static_v1"
    run_dir_input = mo.ui.text(
        value=DEFAULT_RUN_DIR,
        label="Run directory",
        full_width=True,
    )
    run_dir_input  # noqa: B018 - final expression is rendered by Marimo
    return (run_dir_input,)


@app.cell
def _(mo, resolve_path, run_dir_input, yaml):
    run_path = resolve_path(run_dir_input.value)
    snapshot_path = run_path / "run.yaml"
    mo.stop(
        not snapshot_path.exists(),
        mo.callout(f"Missing run snapshot: `{snapshot_path}`", kind="warn"),
    )
    run_snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    trial_names = list(run_snapshot.get("trials", {}))
    mo.stop(not trial_names, mo.callout("The run has no trials.", kind="warn"))
    trial_picker = mo.ui.dropdown(
        options=trial_names,
        value=trial_names[0],
        label="Trial",
    )
    mo.hstack([trial_picker, mo.json(run_snapshot)], widths=[1, 3])
    return run_path, trial_picker


@app.cell
def _(load_metrics, mo, plot_training_metrics, run_path, trial_picker):
    trial_path = run_path / trial_picker.value
    training_tabs = {}
    for stage_name in ("df", "phi"):
        metrics_path = trial_path / stage_name / "metrics.csv"
        if metrics_path.exists():
            metrics = load_metrics(metrics_path)
            figures = plot_training_metrics(
                metrics,
                title=f"{trial_picker.value}: {stage_name}",
            )
            for figure_name, figure in figures.items():
                training_tabs[f"{stage_name}: {figure_name}"] = figure
    training_output = (
        mo.ui.tabs(training_tabs)
        if training_tabs
        else mo.callout("No training metrics are available yet.", kind="info")
    )
    mo.vstack([mo.md("## Training"), training_output])
    return (trial_path,)


@app.cell
def _(load_df_evaluation, mo, plot_density_profile, run_path, trial_path):
    trial_df_eval = trial_path / "eval" / "df"
    summary_df_eval = run_path / "summary" / "df"
    df_evaluation = load_df_evaluation(
        trial_df_eval if trial_df_eval.exists() else summary_df_eval
    )
    df_diagnostics = df_evaluation.get("diagnostics")
    density_figure = (
        plot_density_profile(df_diagnostics)
        if df_diagnostics is not None
        else None
    )
    df_output = (
        mo.vstack(
            [
                mo.json(df_evaluation.get("metrics", {})),
                density_figure,
            ]
        )
        if df_evaluation
        else mo.callout("No DF evaluation is available yet.", kind="info")
    )
    mo.vstack([mo.md("## DF diagnostics"), df_output])


@app.cell
def _(load_phi_evaluation, mo, plot_radial_curves, trial_path):
    phi_evaluation = load_phi_evaluation(trial_path / "eval" / "phi")
    radial = phi_evaluation.get("radial")
    radial_tabs = (
        plot_radial_curves(radial) if radial is not None else {}
    )
    radial_output = (
        mo.ui.tabs(radial_tabs)
        if radial_tabs
        else mo.callout("No Phi radial evaluation is available yet.", kind="info")
    )
    stats_output = (
        mo.json(phi_evaluation["stats"])
        if "stats" in phi_evaluation
        else mo.md("No `eval_stats.json` yet.")
    )
    mo.vstack(
        [
            mo.md("## Phi diagnostics"),
            stats_output,
            radial_output,
        ]
    )


@app.cell
def _(Path, mo, trial_path):
    image_paths = sorted(
        path
        for root in (
            trial_path / "plots",
            trial_path / "eval",
            trial_path / "validation",
        )
        if root.exists()
        for path in root.rglob("*.png")
    )
    gallery = {
        str(path.relative_to(trial_path)): mo.image(src=Path(path))
        for path in image_paths
    }
    gallery_output = (
        mo.ui.tabs(gallery)
        if gallery
        else mo.callout("No saved diagnostic figures are available yet.", kind="info")
    )
    mo.vstack([mo.md("## Saved diagnostic figures"), gallery_output])


if __name__ == "__main__":
    app.run()
