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
                "schema": "dpjax.run.v2",
                "name": "test_run",
                "case": "test",
                "output_dir": str(tmp_path / "artifacts"),
                "data": {
                    "path": str(tmp_path / "data.h5"),
                    "dataset": "eta",
                    "split": {"validation_fraction": 0.1, "seed": 11},
                    "preprocessing": {"clip_sigma": 4.5},
                },
                "df": {
                    "model": str(df_config),
                    "seed": 42,
                    "model_overrides": {"train": {"epochs": 2}},
                },
                "phi": {"model": str(phi_config), "seed": 7},
                "logging": {
                    "backend": "wandb",
                    "project": "test-project",
                    "mode": "offline",
                    "entity": "test-team",
                },
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
    assert spec.case == "test"
    assert spec.df_dir == tmp_path / "artifacts" / "df"
    assert spec.phi_df_dir == spec.df_dir
    assert spec.eval_dir == tmp_path / "artifacts" / "eval"
    df_config = spec.resolve_stage_config("df")
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
    assert snapshot["schema"] == "dpjax.run.v2"
    assert snapshot["case"] == "test"
    assert snapshot["df"]["resolved_config"]["seed"] == 42
    assert snapshot["_meta"]["spec_sha256"]
    for directory in ("logs", "df", "phi", "eval", "plots"):
        assert (tmp_path / "artifacts" / directory).is_dir()
    assert not (tmp_path / "artifacts" / "validation").exists()
    assert prepare_run(spec) == snapshot_path

    df_path = Path(snapshot["df"]["source_model"])
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

    expected = tmp_path / "artifacts" / "df"
    assert captured["run_dir"] == expected
    assert captured["logger_dir"] == expected
    assert captured["config"]["seed"] == 42
    assert captured["training_kwargs"]["resume"] is False
    assert captured["logger_kwargs"]["project"] == "test-project"
    assert captured["logger_kwargs"]["mode"] == "offline"
    assert captured["logger_kwargs"]["entity"] == "test-team"


def test_run_phi_entrypoint_uses_df_from_the_same_experiment(tmp_path, monkeypatch):
    import experiments.run_phi as entrypoint

    source_path = _write_run_config(tmp_path)
    run_root = tmp_path / "artifacts"
    (run_root / "df" / "ckpt" / "1").mkdir(parents=True)
    captured = {}

    class DummyLogger:
        def __init__(self, run_dir, **kwargs):
            captured["logger_dir"] = run_dir
            captured["logger_kwargs"] = kwargs

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
    assert captured["logger_kwargs"]["mode"] == "offline"
    assert captured["logger_kwargs"]["entity"] == "test-team"


def test_run_eval_uses_one_flat_evaluation_directory(tmp_path, monkeypatch):
    import experiments.run_eval as entrypoint

    source_path = _write_run_config(tmp_path)
    run_root = tmp_path / "artifacts"
    (run_root / "df" / "ckpt").mkdir(parents=True)
    (run_root / "phi" / "ckpt").mkdir(parents=True)
    captured = {}

    def fake_df(data_path, run_dirs, output_dir, **kwargs):
        captured["df_run_dirs"] = run_dirs
        captured["df_output_dir"] = output_dir

    def fake_phi(data_path, df_dir, phi_dir, **kwargs):
        captured["phi_df_dir"] = df_dir
        captured["phi_dir"] = phi_dir
        captured["phi_output_dir"] = kwargs["out_dir"]

    monkeypatch.setattr(entrypoint, "evaluate_df_diagnostics", fake_df)
    monkeypatch.setattr(entrypoint, "run_eval_phi", fake_phi)

    entrypoint.run(source_path)

    eval_dir = run_root / "eval"
    assert captured["df_run_dirs"] == [run_root / "df"]
    assert captured["df_output_dir"] == eval_dir
    assert captured["phi_df_dir"] == run_root / "df"
    assert captured["phi_dir"] == run_root / "phi"
    assert captured["phi_output_dir"] == eval_dir
    assert (eval_dir / "df_config.yaml").is_file()
    assert (eval_dir / "phi_config.yaml").is_file()
    assert not (eval_dir / "df").exists()
    assert not (eval_dir / "phi").exists()


def test_run_config_rejects_truth_validation_sections(tmp_path):
    source_path = _write_run_config(tmp_path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["validation"] = {"auriga_truth": {"enabled": True}}
    source_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown run config field.*validation"):
        load_run_spec(source_path)


@pytest.mark.parametrize(
    "path",
    sorted((PROJECT_ROOT / "configs" / "runs").glob("*.yaml")),
)
def test_checked_in_run_configs_resolve_model_recipes(path):
    spec = load_run_spec(path)
    assert spec.resolve_stage_config("df")["kind"] == "df"
    assert spec.resolve_stage_config("phi")["kind"] == "phi"


def test_full_plummer_oracle_only_changes_phi_score_source():
    flow_spec = load_run_spec(
        PROJECT_ROOT / "configs" / "runs" / "plummer_rcut_full.yaml"
    )
    oracle_spec = load_run_spec(
        PROJECT_ROOT / "configs" / "runs" / "plummer_full_oracle.yaml"
    )

    assert flow_spec.data_path == oracle_spec.data_path
    assert oracle_spec.phi_df_dir == (
        PROJECT_ROOT / "runs" / "plummer_rcut" / "full-baseline" / "df"
    )
    assert flow_spec.resolve_stage_config("df") == oracle_spec.resolve_stage_config(
        "df"
    )

    flow_phi = flow_spec.resolve_stage_config("phi")
    oracle_phi = oracle_spec.resolve_stage_config("phi")
    assert "score" not in flow_phi
    assert oracle_phi["score"] == {"source": "plummer_analytic"}
    oracle_without_score = dict(oracle_phi)
    oracle_without_score.pop("score")
    assert oracle_without_score == flow_phi
