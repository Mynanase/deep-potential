"""Orbax checkpoint adapters for experiment workflows."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any


def checkpoint_steps(ckpt_dir: str | Path) -> tuple[int, ...]:
    """Return the completed numeric Orbax steps under *ckpt_dir*."""
    path = Path(ckpt_dir).expanduser()
    if not path.is_dir():
        return ()

    def is_complete_step(child: Path) -> bool:
        if not child.is_dir() or not child.name.isdigit():
            return False
        metadata_path = child / "_CHECKPOINT_METADATA"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(metadata, dict):
            return False
        committed_at = metadata.get("commit_timestamp_nsecs")
        return isinstance(committed_at, int) and committed_at > 0

    return tuple(
        sorted(
            int(child.name)
            for child in path.iterdir()
            if is_complete_step(child)
        )
    )


def has_checkpoint(ckpt_dir: str | Path) -> bool:
    """Return whether *ckpt_dir* contains a completed numeric step."""
    return bool(checkpoint_steps(ckpt_dir))


def require_checkpoint(stage_dir: str | Path, label: str) -> Path:
    """Require a restorable checkpoint for one trained experiment stage."""
    stage_path = Path(stage_dir).expanduser()
    ckpt_dir = stage_path / "ckpt"
    if not has_checkpoint(ckpt_dir):
        raise FileNotFoundError(
            f"Missing completed {label} checkpoint under {ckpt_dir}."
        )
    return ckpt_dir


def create_manager(ckpt_dir: str | Path, *, max_to_keep: int = 3) -> Any:
    import orbax.checkpoint as ocp

    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # If a previous run crashed mid-save, Orbax can leave temporary directories
    # like "123.orbax-checkpoint-tmp" which are not restorable checkpoints.
    # Clean them up so a new run can proceed and `latest_step()` behaves.
    for child in ckpt_dir.iterdir():
        if child.is_dir() and child.name.endswith(".orbax-checkpoint-tmp"):
            shutil.rmtree(child, ignore_errors=True)

    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    # New API (Orbax >= 0.10): use item_handlers, do not pass `checkpointers`.
    return ocp.CheckpointManager(
        str(ckpt_dir),
        item_handlers=ocp.PyTreeCheckpointHandler(),
        options=options,
    )


def save(manager: Any, step: int, item: Any) -> None:
    manager.save(step, item)


def finalize(manager: Any) -> None:
    """Flush/close background checkpoint workers (best-effort).

    Orbax may use async workers; if the process exits immediately after a save,
    you can see noisy shutdown errors like "cannot schedule new futures after shutdown".
    Calling this at the end of a script avoids that.
    """

    wait = getattr(manager, "wait_until_finished", None)
    if callable(wait):
        wait()

    close = getattr(manager, "close", None)
    if callable(close):
        close()


def restore_latest(manager: Any) -> Any:
    steps = checkpoint_steps(manager.directory)
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {manager.directory!r}.")
    return restore_step(manager, steps[-1])


def restore_step(manager: Any, step: int) -> Any:
    # Orbax may warn about missing sharding info on restore; it's harmless for
    # single-host evaluation/training and just adds noise to logs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Sharding info not provided when restoring\..*",
            category=UserWarning,
        )
        try:
            return manager.restore(step)
        except ValueError as e:
            msg = str(e)
            if "sharding passed to deserialization" in msg and "Got None" in msg:
                raise RuntimeError(
                    "Orbax restore failed due to missing/invalid sharding. This often happens when JAX "
                    "cannot initialize the intended platform (e.g. GPU OOM at startup) or when running "
                    "with a different platform than the one available.\n"
                    "Fix: ensure GPU is available; experiment entry points already default "
                    "`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Force CPU via "
                    "`JAX_PLATFORM_NAME=cpu` only for a quick smoke run."
                ) from e
            raise
