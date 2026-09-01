"""Evaluate the trained stages for one concrete experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.workflows.checkpoints import require_checkpoint
from experiments.workflows.config import (
    RunSpec,
    load_run_spec,
    prepare_run,
    write_evaluation_config,
)
from experiments.workflows.evaluation.df import (
    DEFAULT_RADIAL_EDGES,
    evaluate_df_diagnostics,
)
from experiments.workflows.evaluation.phi import run_eval_phi


def _enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", True))


def _run_df(
    spec: RunSpec,
    config: dict[str, Any],
) -> None:
    require_checkpoint(spec.df_dir, "DF")
    write_evaluation_config(spec.eval_dir, "df", config)

    theta_edges = np.arccos(
        np.linspace(1.0, -1.0, int(config.get("n_theta_bins", 6)) + 1)
    )
    phi_edges = np.linspace(
        -np.pi,
        np.pi,
        int(config.get("n_phi_bins", 8)) + 1,
    )
    evaluate_df_diagnostics(
        spec.data_path,
        [spec.df_dir],
        spec.eval_dir,
        n_samples_per_model=int(config.get("n_samples_per_model", 262_144)),
        n_score_points=int(config.get("n_score_points", 32_768)),
        score_batch_size=int(config.get("score_batch_size", 1_024)),
        seed=int(config.get("seed", 42)),
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        n_velocity_bins=int(config.get("n_velocity_bins", 64)),
        radial_edges=np.asarray(
            config.get("radial_edges", DEFAULT_RADIAL_EDGES),
            dtype=np.float64,
        ),
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
        n_cylindrical_r_bins=int(config.get("n_cylindrical_r_bins", 8)),
        n_cylindrical_component_bins=int(
            config.get("n_cylindrical_component_bins", 64)
        ),
        score_field_r_bins=int(config.get("score_field_r_bins", 32)),
        score_field_v_bins=int(config.get("score_field_v_bins", 32)),
        score_field_min_effective_count=float(
            config.get("score_field_min_effective_count", 20)
        ),
        radial_speed_r_bins=int(config.get("radial_speed_r_bins", 64)),
        radial_speed_v_bins=int(config.get("radial_speed_v_bins", 64)),
    )


def _run_phi(
    spec: RunSpec,
    config: dict[str, Any],
    *,
    system: str,
) -> None:
    df_run_dir = spec.phi_df_dir
    require_checkpoint(df_run_dir, "DF")
    require_checkpoint(spec.phi_dir, "Phi")
    write_evaluation_config(spec.eval_dir, "phi", config)
    run_eval_phi(
        spec.data_path,
        df_run_dir,
        spec.phi_dir,
        out_dir=spec.eval_dir,
        n_eval=int(config.get("n_eval", 32_768)),
        batch_size=int(config.get("batch_size", 4_096)),
        seed=int(config.get("seed", 0)),
        r_min=float(config.get("r_min", 1.0e-3)),
        r_max=float(config.get("r_max", 10.0)),
        n_r=int(config.get("n_r", 256)),
        r_ref=float(config.get("r_ref", 1.0)),
        system=system,
        compute_slice=bool(config.get("compute_slice", True)),
        slice_grid=int(config.get("slice_grid", 128)),
        slice_rmax=config.get("slice_rmax"),
        gravitational_constant=config.get("gravitational_constant"),
    )


def run(config_path: str | Path) -> None:
    spec = load_run_spec(config_path)
    prepare_run(spec)
    evaluation = spec.evaluation
    system = str(evaluation.get("system", "generic")).lower()
    if system not in {"generic", "plummer", "halo"}:
        raise ValueError("evaluation.system must be generic, plummer, or halo.")

    df_config = dict(evaluation.get("df", {}))
    phi_config = dict(evaluation.get("phi", {}))

    if _enabled(df_config):
        _run_df(spec, df_config)

    if _enabled(phi_config):
        _run_phi(spec, phi_config, system=system)


def run_stage(config_path: str | Path, stage: str) -> None:
    """Evaluate exactly one explicitly selected stage."""
    if stage not in {"df", "phi"}:
        raise ValueError("stage must be 'df' or 'phi'.")
    spec = load_run_spec(config_path)
    evaluation = spec.evaluation
    config = dict(evaluation.get(stage, {}))
    if not _enabled(config):
        raise ValueError(f"evaluation.{stage}.enabled must be true for eval_{stage}.")

    system = str(evaluation.get("system", "generic")).lower()
    if system not in {"generic", "plummer", "halo"}:
        raise ValueError("evaluation.system must be generic, plummer, or halo.")
    prepare_run(spec)
    if stage == "df":
        _run_df(spec, config)
    else:
        _run_phi(spec, config, system=system)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate stages declared by an experiment YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
