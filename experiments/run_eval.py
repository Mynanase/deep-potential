"""Evaluate selected trials from one run-level YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from dpjax.plotting import plot_auriga_df_ensemble
from dpjax.workflows.config import (
    RunSpec,
    TrialSpec,
    load_run_spec,
    prepare_run,
    write_evaluation_config,
)
from dpjax.workflows.evaluation.auriga_df import evaluate_auriga_df
from dpjax.workflows.evaluation.auriga_truth import run_eval_auriga_truth
from dpjax.workflows.evaluation.df import run_eval_df
from dpjax.workflows.evaluation.phi import run_eval_phi


def _enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", True))


def _fig_formats(config: dict[str, Any]) -> tuple[str, ...]:
    value = config.get("fig_formats", ["png"])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("evaluation.*.fig_formats must be a list of strings.")
    return tuple(value)


def _require_stage(path: Path, label: str) -> None:
    if not (path / "ckpt").exists():
        raise FileNotFoundError(f"Missing trained {label} stage: {path}")


def _run_halo_df(
    spec: RunSpec,
    trials: tuple[TrialSpec, ...],
    config: dict[str, Any],
) -> None:
    run_dirs = [spec.layout(trial).df_dir for trial in trials]
    for run_dir in run_dirs:
        _require_stage(run_dir, "DF")

    if len(trials) == 1:
        layout = spec.layout(trials[0])
        output_dir = layout.eval_dir / "df"
        plots_dir = layout.plots_dir / "df"
    else:
        output_dir = spec.summary_dir / "df"
        plots_dir = spec.summary_dir / "plots" / "df"
    write_evaluation_config(output_dir, config)

    theta_edges = np.arccos(
        np.linspace(1.0, -1.0, int(config.get("n_theta_bins", 6)) + 1)
    )
    phi_edges = np.linspace(
        -np.pi,
        np.pi,
        int(config.get("n_phi_bins", 8)) + 1,
    )
    evaluate_auriga_df(
        spec.data_path,
        run_dirs,
        output_dir,
        n_samples_per_model=int(config.get("n_samples_per_model", 262_144)),
        n_score_points=int(config.get("n_score_points", 32_768)),
        score_batch_size=int(config.get("score_batch_size", 1_024)),
        seed=int(config.get("seed", 42)),
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        n_velocity_bins=int(config.get("n_velocity_bins", 64)),
        spatial_r_edges=np.linspace(
            0.0,
            float(config.get("spatial_r_max", 75.0)),
            int(config.get("spatial_r_bins", 48)) + 1,
        ),
        spatial_z_edges=np.linspace(
            -float(config.get("spatial_z_max", 75.0)),
            float(config.get("spatial_z_max", 75.0)),
            int(config.get("spatial_z_bins", 48)) + 1,
        ),
        spatial_min_cell_count=int(config.get("spatial_min_cell_count", 5)),
    )
    plot_auriga_df_ensemble(
        metrics_json=output_dir / "auriga_df_metrics.json",
        diagnostics_npz=output_dir / "auriga_df_diagnostics.npz",
        fig_dir=plots_dir,
        run_dirs=run_dirs,
        fig_fmt=_fig_formats(config),
        dpi=int(config.get("dpi", 150)),
    )


def _run_generic_df(
    spec: RunSpec,
    trial: TrialSpec,
    config: dict[str, Any],
    *,
    system: str,
) -> None:
    layout = spec.layout(trial)
    _require_stage(layout.df_dir, "DF")
    output_dir = layout.eval_dir / "df"
    plots_dir = layout.plots_dir / "df"
    write_evaluation_config(output_dir, config)
    run_eval_df(
        spec.data_path,
        layout.df_dir,
        out_dir=output_dir,
        plots_dir=plots_dir,
        dataset=spec.dataset,
        coordsys=str(config.get("coordsys", "cart")),
        n_samples=int(config.get("n_samples", 262_144)),
        seed=int(config.get("seed", 0)),
        dim1=str(config.get("dim1", "x")),
        dim2=str(config.get("dim2", "y")),
        logscale=bool(config.get("logscale", False)),
        plummer_diag=bool(config.get("plummer_diag", system == "plummer")),
        n_diag_points=int(config.get("n_diag_points", 16_384)),
    )


def _run_phi(
    spec: RunSpec,
    trial: TrialSpec,
    config: dict[str, Any],
    *,
    system: str,
) -> None:
    layout = spec.layout(trial)
    _require_stage(layout.df_dir, "DF")
    _require_stage(layout.phi_dir, "Phi")
    output_dir = layout.eval_dir / "phi"
    plots_dir = layout.plots_dir / "phi"
    write_evaluation_config(output_dir, config)
    run_eval_phi(
        spec.data_path,
        layout.df_dir,
        layout.phi_dir,
        out_dir=output_dir,
        plots_dir=plots_dir,
        n_eval=int(config.get("n_eval", 32_768)),
        batch_size=int(config.get("batch_size", 4_096)),
        seed=int(config.get("seed", 0)),
        r_min=float(config.get("r_min", 1.0e-3)),
        r_max=float(config.get("r_max", 10.0)),
        n_r=int(config.get("n_r", 256)),
        r_ref=float(config.get("r_ref", 1.0)),
        system=system,
        plot_overview=bool(config.get("plot_overview", True)),
        slice_grid=int(config.get("slice_grid", 128)),
        slice_rmax=config.get("slice_rmax"),
        fig_fmt=_fig_formats(config),
        dpi=int(config.get("dpi", 180)),
        gravitational_constant=config.get("gravitational_constant"),
    )


def _run_truth(
    spec: RunSpec,
    trial: TrialSpec,
    config: dict[str, Any],
) -> None:
    layout = spec.layout(trial)
    output_dir = layout.eval_dir / "truth"
    plots_dir = layout.plots_dir / "truth"
    write_evaluation_config(output_dir, config)
    n_eval = config.get("n_eval", 65_536)
    run_eval_auriga_truth(
        spec.data_path,
        layout.df_dir,
        layout.phi_dir,
        out_dir=output_dir,
        plots_dir=plots_dir,
        n_eval=None if n_eval is None else int(n_eval),
        batch_size=int(config.get("batch_size", 4_096)),
        seed=int(config.get("seed", 0)),
        truth_potential_scale=float(config.get("truth_potential_scale", 1.0)),
        truth_acceleration_scale=float(config.get("truth_acceleration_scale", 1.0)),
        radial_bins=int(config.get("radial_bins", 12)),
        plot_truth=bool(config.get("plot_truth", True)),
        slice_phi_bins=int(config.get("slice_phi_bins", 6)),
        slice_r_bins=int(config.get("slice_r_bins", 40)),
        slice_z_bins=int(config.get("slice_z_bins", 40)),
        slice_min_count=int(config.get("slice_min_count", 3)),
        slice_r_max=config.get("slice_r_max"),
        slice_z_max=config.get("slice_z_max"),
        fig_fmt=_fig_formats(config),
        dpi=int(config.get("dpi", 180)),
    )


def run(config_path: str | Path) -> None:
    spec = load_run_spec(config_path)
    prepare_run(spec)
    trials = spec.selected_trials()
    evaluation = spec.evaluation
    system = str(evaluation.get("system", "generic")).lower()
    if system not in {"generic", "plummer", "halo"}:
        raise ValueError("evaluation.system must be generic, plummer, or halo.")

    df_config = dict(evaluation.get("df", {}))
    phi_config = dict(evaluation.get("phi", {}))
    truth_config = dict(evaluation.get("truth", {"enabled": False}))

    if _enabled(df_config):
        if system == "halo":
            _run_halo_df(spec, trials, df_config)
        else:
            for trial in trials:
                _run_generic_df(spec, trial, df_config, system=system)

    if _enabled(phi_config):
        for trial in trials:
            _run_phi(spec, trial, phi_config, system=system)

    if system == "halo" and _enabled(truth_config):
        for trial in trials:
            _run_truth(spec, trial, truth_config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trials declared by a run-level YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
