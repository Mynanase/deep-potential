"""Train all selected Phi trials from one run-level YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpjax.workflows.config import (
    load_run_spec,
    prepare_run,
    validate_stage_start,
)
from dpjax.workflows.logging import ExperimentLogger
from dpjax.workflows.training.phi import run_phi_training


def run(config_path: str | Path) -> None:
    spec = load_run_spec(config_path)
    prepare_run(spec)
    backend = str(spec.logging.get("backend", "csv"))
    project = str(spec.logging.get("project", "deep-potential"))

    for trial in spec.selected_trials():
        layout = spec.layout(trial)
        if not (layout.df_dir / "ckpt").exists():
            raise FileNotFoundError(
                f"Missing completed DF stage for {trial.name}: {layout.df_dir}"
            )
        config = trial.phi.resolve_config(dataset=spec.dataset)
        validate_stage_start(layout.phi_dir, config, resume=spec.resume)
        if spec.resume and trial.phi.init_params is not None:
            raise ValueError(
                f"{trial.name}: resume and phi.init_params are mutually exclusive."
            )
        run_name = f"{spec.name}-{trial.name}-phi"
        print(f"[run_phi] trial={trial.name} output={layout.phi_dir}")
        with ExperimentLogger(
            layout.phi_dir,
            project=project,
            run_name=run_name,
            backend=backend,
            config=config,
        ) as logger:
            run_phi_training(
                config,
                spec.data_path,
                layout.df_dir,
                layout.phi_dir,
                resume=spec.resume,
                init_params_dir=trial.phi.init_params,
                logger=logger,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Phi trials declared by a run-level YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
