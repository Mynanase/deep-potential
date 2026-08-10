"""Run-level configuration, artifact layout, and overwrite protection."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.paths import PROJECT_ROOT, resolve_path
from experiments.workflows.model_config import (
    load_model_config,
    merge_config,
    validate_model_config,
)

RUN_SCHEMA = "dpjax.run.v1"


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return dict(value)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class StageSpec:
    """A model stage backed by an existing model-level YAML configuration."""

    kind: str
    model_path: Path
    seed: int | None
    model_overrides: dict[str, Any]
    init_params: Path | None = None

    def resolve_config(
        self,
        *,
        data_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Combine a reusable model recipe with this run's operations."""
        config = load_model_config(self.model_path, expected_kind=self.kind)
        config = merge_config(config, self.model_overrides)
        config = validate_model_config(config, expected_kind=self.kind)
        config = merge_config(config, {"data": dict(data_config)})
        config = merge_config(config, {"train": dict(runtime_config)})
        if self.seed is not None:
            config = merge_config(config, {"seed": self.seed})
        return config


@dataclass(frozen=True)
class TrialSpec:
    name: str
    df: StageSpec
    phi: StageSpec


@dataclass(frozen=True)
class TrialLayout:
    root: Path

    @property
    def df_dir(self) -> Path:
        return self.root / "df"

    @property
    def phi_dir(self) -> Path:
        return self.root / "phi"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def validation_dir(self) -> Path:
        return self.root / "validation"


@dataclass(frozen=True)
class RunSpec:
    source_path: Path
    name: str
    output_dir: Path
    data_path: Path
    dataset: str
    data_config: dict[str, Any]
    trials: tuple[TrialSpec, ...]
    logging: dict[str, Any]
    execution: dict[str, Any]
    evaluation: dict[str, Any]
    plots: dict[str, Any]
    validation: dict[str, Any]

    @property
    def snapshot_path(self) -> Path:
        return self.output_dir / "run.yaml"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def summary_dir(self) -> Path:
        return self.output_dir / "summary"

    def layout(self, trial: TrialSpec | str) -> TrialLayout:
        name = trial.name if isinstance(trial, TrialSpec) else trial
        return TrialLayout(self.output_dir / name)

    @property
    def resume(self) -> bool:
        return bool(self.execution.get("resume", False))

    def selected_trials(self) -> tuple[TrialSpec, ...]:
        selected = self.execution.get("trials")
        if selected is None:
            return self.trials
        if not isinstance(selected, list) or not all(
            isinstance(item, str) for item in selected
        ):
            raise ValueError("execution.trials must be a list of trial names.")
        by_name = {trial.name: trial for trial in self.trials}
        unknown = [name for name in selected if name not in by_name]
        if unknown:
            raise ValueError(
                "execution.trials contains unknown trial(s): " + ", ".join(unknown)
            )
        return tuple(by_name[name] for name in selected)

    def resolved_data_config(self) -> dict[str, Any]:
        """Translate the run-facing data layout into the training API layout."""
        config = dict(self.data_config)
        split = _require_mapping(config.pop("split", {}), "data.split")
        preprocessing = _require_mapping(
            config.pop("preprocessing", {}),
            "data.preprocessing",
        )
        if "validation_fraction" in split:
            config["val_frac"] = float(split["validation_fraction"])
        if "seed" in split:
            config["split_seed"] = int(split["seed"])
        config.update(preprocessing)
        config["dataset"] = self.dataset
        return config

    def stage_runtime_config(self, stage_name: str) -> dict[str, Any]:
        """Return run-only values in the legacy training-function shape."""
        stages = _require_mapping(
            self.execution.get("stages", {}),
            "execution.stages",
        )
        raw = _require_mapping(
            stages.get(stage_name, {}),
            f"execution.stages.{stage_name}",
        )
        key_map = {
            "multi_gpu": "multi_gpu",
            "log_every": "log_every",
            "checkpoint_every": "ckpt_every",
            "checkpoints_to_keep": "max_to_keep",
        }
        unknown = sorted(set(raw) - set(key_map))
        if unknown:
            raise ValueError(
                f"Unknown execution.stages.{stage_name} field(s): "
                + ", ".join(unknown)
            )
        return {key_map[key]: value for key, value in raw.items()}

    def resolve_stage_config(
        self,
        trial: TrialSpec,
        stage_name: str,
    ) -> dict[str, Any]:
        """Resolve one DF/Phi config for execution and artifact snapshots."""
        if stage_name not in {"df", "phi"}:
            raise ValueError("stage_name must be 'df' or 'phi'.")
        stage = trial.df if stage_name == "df" else trial.phi
        return stage.resolve_config(
            data_config=self.resolved_data_config(),
            runtime_config=self.stage_runtime_config(stage_name),
        )

    def training_snapshot(self) -> dict[str, Any]:
        """Return the immutable, resolved portion of the run configuration."""
        trials: dict[str, Any] = {}
        for trial in self.trials:
            trial_data: dict[str, Any] = {}
            for stage_name, stage in (("df", trial.df), ("phi", trial.phi)):
                stage_data: dict[str, Any] = {
                    "source_model": _portable_path(stage.model_path),
                    "resolved_config": self.resolve_stage_config(
                        trial,
                        stage_name,
                    ),
                }
                if stage.init_params is not None:
                    stage_data["init_params"] = _portable_path(stage.init_params)
                trial_data[stage_name] = stage_data
            trials[trial.name] = trial_data
        return {
            "schema": RUN_SCHEMA,
            "name": self.name,
            "output_dir": _portable_path(self.output_dir),
            "data": {
                "path": _portable_path(self.data_path),
                **self.data_config,
            },
            "trials": trials,
        }


def _load_stage(raw: Any, field: str, *, kind: str) -> StageSpec:
    stage = _require_mapping(raw, field)
    allowed = {"model", "seed", "model_overrides", "init_params"}
    unknown = sorted(set(stage) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {field} field(s): {', '.join(unknown)}. "
            "Use 'model' and optional 'model_overrides'."
        )
    model_path = resolve_path(_require_string(stage.get("model"), f"{field}.model"))
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {field}.model: {model_path}")
    load_model_config(model_path, expected_kind=kind)
    seed_raw = stage.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    model_overrides = _require_mapping(
        stage.get("model_overrides", {}),
        f"{field}.model_overrides",
    )
    init_params_raw = stage.get("init_params")
    init_params = (
        None
        if init_params_raw is None
        else resolve_path(_require_string(init_params_raw, f"{field}.init_params"))
    )
    return StageSpec(
        kind=kind,
        model_path=model_path,
        seed=seed,
        model_overrides=model_overrides,
        init_params=init_params,
    )


def load_run_spec(path: str | Path) -> RunSpec:
    """Load and validate a run-level YAML configuration."""
    source_path = resolve_path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Run config not found: {source_path}")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    raw = _require_mapping(raw, "run config")

    if raw.get("schema") != RUN_SCHEMA:
        raise ValueError(f"run config schema must be {RUN_SCHEMA!r}.")
    allowed = {
        "schema",
        "name",
        "output_dir",
        "data",
        "trials",
        "logging",
        "execution",
        "evaluation",
        "plots",
        "validation",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("Unknown run config field(s): " + ", ".join(unknown))

    name = _require_string(raw.get("name"), "name")
    output_dir = resolve_path(_require_string(raw.get("output_dir"), "output_dir"))
    data = _require_mapping(raw.get("data"), "data")
    data_path = resolve_path(_require_string(data.get("path"), "data.path"))
    dataset = _require_string(data.get("dataset", "eta"), "data.dataset")
    data_config = dict(data)
    data_config.pop("path", None)
    _require_mapping(data_config.get("split", {}), "data.split")
    _require_mapping(data_config.get("preprocessing", {}), "data.preprocessing")

    trials_raw = _require_mapping(raw.get("trials"), "trials")
    if not trials_raw:
        raise ValueError("trials must contain at least one trial.")
    trials: list[TrialSpec] = []
    for trial_name, trial_raw in trials_raw.items():
        trial_name = _require_string(trial_name, "trial name")
        trial = _require_mapping(trial_raw, f"trials.{trial_name}")
        trials.append(
            TrialSpec(
                name=trial_name,
                df=_load_stage(
                    trial.get("df"),
                    f"trials.{trial_name}.df",
                    kind="df",
                ),
                phi=_load_stage(
                    trial.get("phi"),
                    f"trials.{trial_name}.phi",
                    kind="phi",
                ),
            )
        )

    return RunSpec(
        source_path=source_path,
        name=name,
        output_dir=output_dir,
        data_path=data_path,
        dataset=dataset,
        data_config=data_config,
        trials=tuple(trials),
        logging=_require_mapping(raw.get("logging", {}), "logging"),
        execution=_require_mapping(raw.get("execution", {}), "execution"),
        evaluation=_require_mapping(raw.get("evaluation", {}), "evaluation"),
        plots=_require_mapping(raw.get("plots", {}), "plots"),
        validation=_require_mapping(raw.get("validation", {}), "validation"),
    )


def prepare_run(spec: RunSpec) -> Path:
    """Create the run root and persist/validate its immutable config snapshot."""
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.logs_dir.mkdir(parents=True, exist_ok=True)
    payload = spec.training_snapshot()
    canonical = yaml.safe_dump(payload, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if spec.snapshot_path.exists():
        existing = yaml.safe_load(spec.snapshot_path.read_text(encoding="utf-8")) or {}
        existing_digest = existing.get("_meta", {}).get("spec_sha256")
        if existing_digest != digest:
            raise ValueError(
                f"{spec.snapshot_path} belongs to a different resolved run "
                "configuration. Use a new name/output_dir instead of mixing "
                "artifacts from different configurations."
            )
        return spec.snapshot_path

    payload["_meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": _portable_path(spec.source_path),
        "spec_sha256": digest,
        "git_commit": _git_commit(),
    }
    spec.snapshot_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return spec.snapshot_path


def validate_stage_start(
    stage_dir: Path,
    resolved_config: dict[str, Any],
    *,
    resume: bool,
) -> None:
    """Reject accidental overwrite or unsafe resume of a training stage."""
    config_path = stage_dir / "config.yaml"
    ckpt_dir = stage_dir / "ckpt"
    has_checkpoint = ckpt_dir.exists() and any(
        child.is_dir() and child.name.isdigit() for child in ckpt_dir.iterdir()
    )

    if resume and not has_checkpoint:
        raise FileNotFoundError(
            f"resume=true but no checkpoint exists under {ckpt_dir}."
        )
    if has_checkpoint and not resume:
        raise FileExistsError(
            f"{stage_dir} already contains checkpoints. Set execution.resume "
            "to true or choose a new output_dir."
        )
    if config_path.exists():
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if saved != resolved_config:
            raise ValueError(
                f"Resolved config does not match existing {config_path}; "
                "refusing to mix incompatible stage artifacts."
            )


def write_evaluation_config(path: Path, config: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
