"""Train all selected DF trials from one run-level YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.workflows.config import (
    load_run_spec,
    prepare_run,
    validate_stage_start,
)
from experiments.workflows.logging import ExperimentLogger
from experiments.workflows.training.df import run_df_training


def run(config_path: str | Path) -> None:
    spec = load_run_spec(config_path)
    prepare_run(spec)
    backend = str(spec.logging.get("backend", "csv"))
    project = str(spec.logging.get("project", "deep-potential"))
    mode = spec.logging.get("mode")
    entity = spec.logging.get("entity")

    for trial in spec.selected_trials():
        layout = spec.layout(trial)
        config = spec.resolve_stage_config(trial, "df")
        validate_stage_start(layout.df_dir, config, resume=spec.resume)
        run_name = f"{spec.name}-{trial.name}-df"
        print(f"[run_df] trial={trial.name} output={layout.df_dir}")
        with ExperimentLogger(
            layout.df_dir,
            project=project,
            run_name=run_name,
            backend=backend,
            config=config,
            mode=None if mode is None else str(mode),
            entity=None if entity is None else str(entity),
        ) as logger:
            run_df_training(
                config,
                spec.data_path,
                layout.df_dir,
                resume=spec.resume,
                logger=logger,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train DF trials declared by a run-level YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
