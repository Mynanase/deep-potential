"""Train the DF stage for one concrete experiment."""

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

    config = spec.resolve_stage_config("df")
    validate_stage_start(spec.df_dir, config, resume=spec.resume)
    run_name = f"{spec.name}-df"
    print(f"[run_df] experiment={spec.name} output={spec.df_dir}")
    with ExperimentLogger(
        spec.df_dir,
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
            spec.df_dir,
            resume=spec.resume,
            logger=logger,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the DF stage declared by an experiment YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
