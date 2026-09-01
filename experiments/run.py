"""Compose isolated DF and Phi experiment stages into daily workflows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experiments.paths import PROJECT_ROOT
from experiments.runtime import configure_runtime_environment
from experiments.workflows.checkpoints import require_checkpoint
from experiments.workflows.config import RunSpec, load_run_spec

WORKFLOWS = ("df", "phi", "all")
WORKFLOW_MODULES = {
    "df": (
        "experiments.run_df",
        "experiments.eval_df",
        "experiments.plot_df",
    ),
    "phi": (
        "experiments.run_phi",
        "experiments.eval_phi",
        "experiments.plot_phi",
    ),
    "all": (
        "experiments.run_df",
        "experiments.eval_df",
        "experiments.plot_df",
        "experiments.run_phi",
        "experiments.eval_phi",
        "experiments.plot_phi",
    ),
}


def _evaluation_enabled(spec: RunSpec, stage: str) -> bool:
    config = dict(spec.evaluation.get(stage, {}))
    return bool(config.get("enabled", True))


def _validate_workflow(spec: RunSpec, workflow: str) -> None:
    stages = ("df", "phi") if workflow == "all" else (workflow,)
    disabled = [stage for stage in stages if not _evaluation_enabled(spec, stage)]
    if disabled:
        names = ", ".join(f"evaluation.{stage}.enabled" for stage in disabled)
        raise ValueError(f"{names} must be true for the {workflow} workflow.")
    if workflow == "all" and spec.phi_df_dir != spec.df_dir:
        raise ValueError(
            "The all workflow requires Phi to use this run's DF. "
            "Use the phi workflow for a config with phi.df_run."
        )
    if workflow == "phi":
        require_checkpoint(spec.phi_df_dir, "DF")


def _run_module(module: str, config_path: Path) -> None:
    child_env = dict(os.environ)
    configure_runtime_environment(child_env)
    subprocess.run(
        [sys.executable, "-u", "-m", module, str(config_path)],
        cwd=PROJECT_ROOT,
        env=child_env,
        check=True,
    )


def run(config_path: str | Path, *, workflow: str) -> None:
    """Run one fail-fast workflow with every expensive stage isolated."""
    if workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of: {', '.join(WORKFLOWS)}")
    spec = load_run_spec(config_path)
    _validate_workflow(spec, workflow)
    for module in WORKFLOW_MODULES[workflow]:
        _run_module(module, spec.source_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an isolated DF, Phi, or complete experiment workflow."
    )
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config, workflow=args.workflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
