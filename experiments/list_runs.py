"""List repository experiment runs from their immutable snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.paths import RUNS_DIR
from experiments.workflows.checkpoints import has_checkpoint


def collect_runs(runs_dir: str | Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Collect current state dynamically; no separate run database is used."""
    runs_dir = Path(runs_dir)
    records: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return records
    for snapshot_path in sorted(runs_dir.rglob("run.yaml")):
        run_dir = snapshot_path.parent
        raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        result_data = run_dir / "results" / "data"
        legacy_eval = run_dir / "eval"
        has_new_evaluation = result_data.is_dir() and any(
            result_data.glob("*_metrics.json")
        )
        has_legacy_evaluation = legacy_eval.is_dir() and any(
            legacy_eval.glob("*_metrics.json")
        )
        legacy_plots = run_dir / "plots"
        result_path = run_dir / "results" if (run_dir / "results").exists() else legacy_plots
        records.append(
            {
                "name": raw.get("name", run_dir.name),
                "case": raw.get("case"),
                "git_commit": raw.get("_meta", {}).get("git_commit"),
                "df": has_checkpoint(run_dir / "df" / "ckpt"),
                "phi": has_checkpoint(run_dir / "phi" / "ckpt"),
                "eval": has_new_evaluation or has_legacy_evaluation,
                "plot": (run_dir / "results" / "manifest.json").is_file()
                or (legacy_plots.is_dir() and any(legacy_plots.iterdir())),
                "result_path": str(result_path),
                "run_path": str(run_dir),
            }
        )
    return records


def _status(value: bool) -> str:
    return "yes" if value else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="List experiment runs and artifact state.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args()
    records = collect_runs(args.runs_dir)
    if args.json:
        print(json.dumps(records, indent=2))
        return 0
    columns = ("name", "case", "git", "DF", "Phi", "eval", "plot", "results")
    print("  ".join(columns))
    for record in records:
        commit = record["git_commit"] or "-"
        print(
            "  ".join(
                [
                    str(record["name"]),
                    str(record["case"] or "-"),
                    str(commit)[:10],
                    _status(record["df"]),
                    _status(record["phi"]),
                    _status(record["eval"]),
                    _status(record["plot"]),
                    record["result_path"],
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
