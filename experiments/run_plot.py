"""Render persisted diagnostics into official experiment figures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.plotting.registry import FigureWriter, render_all
from experiments.workflows.config import load_run_spec, prepare_run

SECTIONS = ("training", "df", "phi", "validation")


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
