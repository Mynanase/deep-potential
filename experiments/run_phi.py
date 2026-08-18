"""Train the Phi stage for one concrete experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.workflows.config import (
    load_run_spec,
    prepare_run,
    validate_stage_start,
)
from experiments.workflows.logging import ExperimentLogger
from experiments.workflows.training.phi import run_phi_training


def run(config_path: str | Path) -> None:
    spec = load_run_spec(config_path)
    prepare_run(spec)
    backend = str(spec.logging.get("backend", "csv"))
    project = str(spec.logging.get("project", "deep-potential"))
    mode = spec.logging.get("mode")
    entity = spec.logging.get("entity")

    df_run_dir = spec.phi_df_dir
    if not (df_run_dir / "ckpt").exists():
        raise FileNotFoundError(f"Missing completed DF stage: {df_run_dir}")
    config = spec.resolve_stage_config("phi")
    validate_stage_start(spec.phi_dir, config, resume=spec.resume)
    if spec.resume and spec.phi.init_params is not None:
        raise ValueError("resume and phi.init_params are mutually exclusive.")
    run_name = f"{spec.name}-phi"
    print(f"[run_phi] experiment={spec.name} output={spec.phi_dir}")
    with ExperimentLogger(
        spec.phi_dir,
        project=project,
        run_name=run_name,
        backend=backend,
        config=config,
        mode=None if mode is None else str(mode),
        entity=None if entity is None else str(entity),
    ) as logger:
        run_phi_training(
            config,
            spec.data_path,
            df_run_dir,
            spec.phi_dir,
            resume=spec.resume,
            init_params_dir=spec.phi.init_params,
            logger=logger,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the Phi stage declared by an experiment YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
