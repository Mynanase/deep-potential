from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.workflows.config import (
    load_run_spec,
    prepare_run,
    validate_stage_start,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_run_config(tmp_path: Path) -> Path:
    df_config = tmp_path / "df.yaml"
    phi_config = tmp_path / "phi.yaml"
    df_config.write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.model.v1",
                "kind": "df",
                "normalizer": {"eps": 1.0e-6},
                "flow": {"type": "ffjord", "dim": 6},
                "train": {"optimizer": "radam", "epochs": 10},
            }
        ),
        encoding="utf-8",
    )
    phi_config.write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.model.v1",
                "kind": "phi",
                "potential": {"hidden_sizes": [16, 16]},
                "train": {"optimizer": "adam", "epochs": 20},
            }
        ),
        encoding="utf-8",
    )
    run_config = tmp_path / "run-source.yaml"
    run_config.write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.run.v1",
                "name": "test_run",
                "output_dir": str(tmp_path / "artifacts"),
                "data": {
                    "path": str(tmp_path / "data.h5"),
                    "dataset": "eta",
                    "split": {"validation_fraction": 0.1, "seed": 11},
                    "preprocessing": {"clip_sigma": 4.5},
                },
                "trials": {
                    "trial_00": {
                        "df": {
                            "model": str(df_config),
                            "seed": 42,
                            "model_overrides": {"train": {"epochs": 2}},
                        },
                        "phi": {"model": str(phi_config), "seed": 7},
                    }
                },
                "logging": {"backend": "csv"},
                "execution": {
                    "resume": False,
                    "stages": {
                        "df": {
                            "multi_gpu": False,
                            "log_every": 5,
                            "checkpoint_every": 10,
                            "checkpoints_to_keep": 2,
                        },
                        "phi": {"multi_gpu": False, "log_every": 7},
                    },
                },
                "evaluation": {"system": "generic"},
                "plots": {"formats": ["png"], "dpi": 120},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_config


def test_run_spec_resolves_stage_configs_and_layout(tmp_path):
    spec = load_run_spec(_write_run_config(tmp_path))

    assert spec.name == "test_run"
    assert spec.selected_trials()[0].name == "trial_00"
    assert spec.layout("trial_00").df_dir == (
        tmp_path / "artifacts" / "trial_00" / "df"
    )
    df_config = spec.resolve_stage_config(spec.trials[0], "df")
    assert df_config["seed"] == 42
    assert df_config["data"]["dataset"] == "eta"
    assert df_config["data"]["val_frac"] == 0.1
    assert df_config["data"]["split_seed"] == 11
    assert df_config["data"]["clip_sigma"] == 4.5
    assert df_config["train"]["epochs"] == 2
    assert df_config["train"]["multi_gpu"] is False
    assert df_config["train"]["log_every"] == 5
    assert df_config["train"]["ckpt_every"] == 10
    assert df_config["train"]["max_to_keep"] == 2


def test_prepare_run_snapshots_resolved_config_and_rejects_changes(tmp_path):
    source_path = _write_run_config(tmp_path)
    spec = load_run_spec(source_path)
    snapshot_path = prepare_run(spec)

    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["schema"] == "dpjax.run.v1"
    assert snapshot["trials"]["trial_00"]["df"]["resolved_config"]["seed"] == 42
    assert snapshot["_meta"]["spec_sha256"]
    assert prepare_run(spec) == snapshot_path

    df_path = Path(snapshot["trials"]["trial_00"]["df"]["source_model"])
    if not df_path.is_absolute():
        # Temporary paths are absolute; this branch documents portable repo paths.
        pytest.fail("temporary source config unexpectedly became relative")
    changed = yaml.safe_load(df_path.read_text(encoding="utf-8"))
    changed["train"]["optimizer"] = "adam"
    df_path.write_text(yaml.safe_dump(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="different resolved run configuration"):
        prepare_run(load_run_spec(source_path))


def test_validate_stage_start_prevents_overwrite_and_unsafe_resume(tmp_path):
    stage_dir = tmp_path / "df"
    ckpt_dir = stage_dir / "ckpt" / "12"
    ckpt_dir.mkdir(parents=True)
    config = {"seed": 3, "train": {"epochs": 2}}
    (stage_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_stage_start(stage_dir, config, resume=False)
    validate_stage_start(stage_dir, config, resume=True)
    with pytest.raises(ValueError, match="does not match"):
        validate_stage_start(stage_dir, {"seed": 4}, resume=True)

    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        validate_stage_start(tmp_path / "empty", config, resume=True)


def test_run_df_entrypoint_derives_paths_from_run_config(tmp_path, monkeypatch):
    import experiments.run_df as entrypoint

    source_path = _write_run_config(tmp_path)
    captured = {}

    class DummyLogger:
        def __init__(self, run_dir, **kwargs):
            captured["logger_dir"] = run_dir
            captured["logger_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_training(config, data_path, run_dir, **kwargs):
        captured["config"] = config
        captured["data_path"] = data_path
        captured["run_dir"] = run_dir
        captured["training_kwargs"] = kwargs

    monkeypatch.setattr(entrypoint, "ExperimentLogger", DummyLogger)
    monkeypatch.setattr(entrypoint, "run_df_training", fake_training)

    entrypoint.run(source_path)

    expected = tmp_path / "artifacts" / "trial_00" / "df"
    assert captured["run_dir"] == expected
    assert captured["logger_dir"] == expected
    assert captured["config"]["seed"] == 42
    assert captured["training_kwargs"]["resume"] is False


def test_run_phi_entrypoint_uses_df_from_the_same_trial(tmp_path, monkeypatch):
    import experiments.run_phi as entrypoint

    source_path = _write_run_config(tmp_path)
    run_root = tmp_path / "artifacts" / "trial_00"
    (run_root / "df" / "ckpt" / "1").mkdir(parents=True)
    captured = {}

    class DummyLogger:
        def __init__(self, run_dir, **kwargs):
            captured["logger_dir"] = run_dir

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_training(config, data_path, df_run_dir, run_dir, **kwargs):
        captured["config"] = config
        captured["df_run_dir"] = df_run_dir
        captured["run_dir"] = run_dir
        captured["training_kwargs"] = kwargs

    monkeypatch.setattr(entrypoint, "ExperimentLogger", DummyLogger)
    monkeypatch.setattr(entrypoint, "run_phi_training", fake_training)

    entrypoint.run(source_path)

    assert captured["df_run_dir"] == run_root / "df"
    assert captured["run_dir"] == run_root / "phi"
    assert captured["logger_dir"] == run_root / "phi"
    assert captured["config"]["seed"] == 7


@pytest.mark.parametrize(
    "path",
    sorted((PROJECT_ROOT / "configs" / "runs").glob("*.yaml")),
)
def test_checked_in_run_configs_resolve_model_recipes(path):
    spec = load_run_spec(path)
    for trial in spec.trials:
        assert spec.resolve_stage_config(trial, "df")["kind"] == "df"
        assert spec.resolve_stage_config(trial, "phi")["kind"] == "phi"
