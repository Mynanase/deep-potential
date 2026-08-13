"""Launch one experiment stage as a detached, logged worker process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import PROJECT_ROOT
from experiments.runtime import configure_runtime_environment
from experiments.workflows.config import load_run_spec, prepare_run


STAGE_MODULES = {
    "df": "experiments.run_df",
    "phi": "experiments.run_phi",
    "eval": "experiments.run_eval",
}


@dataclass(frozen=True)
class LaunchResult:
    stage: str
    pid: int
    log_path: Path
    pid_path: Path


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if _process_is_running(pid) else None


def launch(stage: str, config_path: str | Path) -> LaunchResult:
    """Detach one stage, capture its console output, and persist its PID."""
    if stage not in STAGE_MODULES:
        raise ValueError(f"stage must be one of: {', '.join(STAGE_MODULES)}")

    spec = load_run_spec(config_path)
    prepare_run(spec)
    log_path = spec.logs_dir / f"{stage}.log"
    pid_path = spec.logs_dir / f"{stage}.pid"

    active_pid = _active_pid(pid_path)
    if active_pid is not None:
        raise RuntimeError(
            f"{stage} is already running for {spec.name} with PID {active_pid}. "
            f"See {log_path}."
        )

    command = [
        sys.executable,
        "-u",
        "-m",
        STAGE_MODULES[stage],
        str(spec.source_path),
    ]
    child_env = dict(os.environ)
    configure_runtime_environment(child_env)
    started_at = datetime.now(timezone.utc).isoformat()

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"\n[launch] started_at={started_at}\n")
        log_file.write(f"[launch] stage={stage} run={spec.name}\n")
        log_file.write(f"[launch] config={spec.source_path}\n")
        log_file.write(
            "[launch] CUDA_VISIBLE_DEVICES="
            f"{child_env.get('CUDA_VISIBLE_DEVICES', 'all visible devices')}\n"
        )
        log_file.write(
            "[launch] XLA_PYTHON_CLIENT_PREALLOCATE="
            f"{child_env['XLA_PYTHON_CLIENT_PREALLOCATE']}\n"
        )
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return LaunchResult(
        stage=stage,
        pid=process.pid,
        log_path=log_path,
        pid_path=pid_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a detached experiment stage with run-scoped logs."
    )
    parser.add_argument("stage", choices=tuple(STAGE_MODULES))
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    result = launch(args.stage, args.config)
    print(
        f"Launched {result.stage} with PID {result.pid}. "
        f"Log: {result.log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
