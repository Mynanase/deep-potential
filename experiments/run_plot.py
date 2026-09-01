"""Render persisted diagnostics into official experiment figures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.diagnostics.artifact_paths import resolve_artifact
from experiments.plotting.registry import FigureWriter, render_all, render_many
from experiments.workflows.config import load_run_spec, prepare_run

SECTIONS = ("training", "df", "phi", "validation")
DF_FIGURES = (
    "training_df",
    "training_df_score_stats",
    "df_density_profile",
    "df_cylindrical_rz_density",
    "df_velocity_marginals_by_r",
    "df_velocity_marginals_by_theta",
    "df_velocity_marginals_by_phi",
    "df_score_distribution",
    "df_cylindrical_marginals_by_R",
    "df_score_field_rv",
    "df_score_slices_by_R",
    "df_radial_speed_density",
)
PHI_FIGURES = (
    "training_phi",
    "training_phi_residual_stats",
    "phi_potential_profile",
    "phi_acceleration_profile",
    "phi_density_profile",
)
PHI_SLICE_FIGURES = (
    "phi_potential_slice",
    "phi_density_slice",
)


def _write_report(spec, outputs: dict[str, tuple[Path, ...]]) -> None:
    evaluation_data = spec.result_data_dir
    if not any(evaluation_data.glob("*.json")) and not any(
        evaluation_data.glob("*.npz")
    ):
        legacy_evaluation = spec.output_dir / "eval"
        if legacy_evaluation.is_dir():
            evaluation_data = legacy_evaluation
    lines = [
        f"# Experiment report: {spec.name}",
        "",
        f"- Case: `{spec.case}`",
        f"- Run: `{spec.output_dir}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Evaluation data: `{evaluation_data}`",
        "",
        "## Figures",
        "",
    ]
    recorded_outputs: dict[str, list[Path]] = {
        name: list(paths) for name, paths in outputs.items()
    }
    if spec.manifest_path.is_file():
        manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
        recorded_outputs = {
            name: [spec.output_dir / item["path"] for item in entry["outputs"]]
            for name, entry in manifest.get("figures", {}).items()
        }
    if recorded_outputs:
        for name, paths in sorted(recorded_outputs.items()):
            rendered = ", ".join(
                f"`{path.relative_to(spec.output_dir)}`" for path in paths
            )
            lines.append(f"- {name}: {rendered}")
    else:
        lines.append("No renderable saved diagnostics were found.")
    spec.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    *,
    sections: list[str] | tuple[str, ...] | None = None,
) -> dict[str, tuple[Path, ...]]:
    """Render and write every available figure in the requested sections."""
    if sections is not None:
        invalid = sorted(set(sections) - set(SECTIONS))
        if invalid:
            raise ValueError(f"Unknown plot section(s): {', '.join(invalid)}")
    spec = load_run_spec(config_path)
    prepare_run(spec)
    figures = render_all(spec, sections)
    writer = FigureWriter(
        spec,
        dpi=int(spec.plots.get("dpi", 200)),
    )
    outputs = {
        name: writer.write(figure, name, target="official")
        for name, figure in figures.items()
    }
    _write_report(spec, outputs)
    return outputs


def _stage_figure_names(spec, stage: str) -> tuple[str, ...]:
    if stage == "df":
        return DF_FIGURES
    if stage == "phi":
        phi_evaluation = dict(spec.evaluation.get("phi", {}))
        if bool(phi_evaluation.get("compute_slice", True)):
            return PHI_FIGURES + PHI_SLICE_FIGURES
        return PHI_FIGURES
    raise ValueError("stage must be 'df' or 'phi'.")


def _require_stage_plot_inputs(spec, stage: str) -> None:
    metrics_path = getattr(spec, f"{stage}_dir") / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {stage.upper()} training metrics: {metrics_path}")
    resolve_artifact(spec.output_dir, f"{stage}_diagnostics.npz")


def run_stage(
    config_path: str | Path,
    stage: str,
) -> dict[str, tuple[Path, ...]]:
    """Strictly render one stage's training and evaluation figures."""
    spec = load_run_spec(config_path)
    prepare_run(spec)
    names = _stage_figure_names(spec, stage)
    _require_stage_plot_inputs(spec, stage)
    figures = render_many(spec, names)
    writer = FigureWriter(
        spec,
        dpi=int(spec.plots.get("dpi", 200)),
    )
    outputs: dict[str, tuple[Path, ...]] = {}
    try:
        for name, figure in figures.items():
            outputs[name] = writer.write(figure, name, target="official")
    finally:
        import matplotlib.pyplot as plt

        for figure in figures.values():
            plt.close(figure)
    _write_report(spec, outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write publication figures from saved experiment diagnostics."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=SECTIONS,
        metavar="SECTION",
        help="Render only selected sections (training, df, phi, validation).",
    )
    args = parser.parse_args()
    outputs = run(args.config, sections=args.only)
    for name, paths in outputs.items():
        print(f"{name}: {', '.join(str(path) for path in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
