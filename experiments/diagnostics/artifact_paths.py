"""Resolve result artifacts across the current and legacy run layouts."""

from __future__ import annotations

from pathlib import Path


def artifact_candidates(location: str | Path, filename: str) -> tuple[Path, ...]:
    """Return candidates for a flat artifact from any supported location."""
    root = Path(location).expanduser()
    candidates = [
        root / filename,
        root / "data" / filename,
        root / "results" / "data" / filename,
        root / "eval" / filename,
    ]
    return tuple(dict.fromkeys(candidates))


def resolve_artifact(location: str | Path, filename: str) -> Path:
    """Resolve one artifact, preferring the new layout over legacy paths."""
    candidates = artifact_candidates(location, filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing result artifact {filename!r}; searched:\n  - {searched}"
    )


def resolve_figure_artifact(location: str | Path, filename: str) -> Path:
    """Resolve an official figure from the new or legacy plot layout."""
    root = Path(location).expanduser()
    candidates = (
        root / "results" / "figures" / filename,
        root / "figures" / filename,
        root / "plots" / filename,
        root / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing figure artifact {filename!r}; searched:\n  - {searched}"
    )
