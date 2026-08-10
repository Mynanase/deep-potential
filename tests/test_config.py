from __future__ import annotations

from pathlib import Path

import pytest

from experiments.workflows.model_config import (
    load_model_config,
    merge_config,
    validate_model_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_merge_config_is_recursive_and_non_mutating():
    base = {
        "flow": {"type": "ffjord", "hidden": 32},
        "train": {"epochs": 10},
    }

    merged = merge_config(
        base,
        {"flow": {"hidden": 64}, "train": {"batch_size": 128}},
    )

    assert merged == {
        "flow": {"type": "ffjord", "hidden": 64},
        "train": {"epochs": 10, "batch_size": 128},
    }
    assert base["flow"]["hidden"] == 32


def test_load_config_resolves_repository_config():
    config = load_model_config(
        "configs/models/df/halo12_ffjord_wide_v1.yaml",
        expected_kind="df",
    )

    assert config["schema"] == "dpjax.model.v1"
    assert config["kind"] == "df"
    assert config["flow"]["type"] == "ffjord"
    assert config["train"]["optimizer"] == "radam"
    assert "data" not in config


def test_model_config_rejects_run_level_fields():
    with pytest.raises(ValueError, match="Run-level field"):
        validate_model_config(
            {
                "schema": "dpjax.model.v1",
                "kind": "df",
                "data": {"dataset": "eta"},
                "flow": {"type": "ffjord"},
                "train": {"optimizer": "radam"},
            }
        )


def test_model_config_rejects_runtime_train_fields():
    with pytest.raises(ValueError, match="Run-time train field"):
        validate_model_config(
            {
                "schema": "dpjax.model.v1",
                "kind": "phi",
                "potential": {"hidden_sizes": [16, 16]},
                "train": {"optimizer": "adam", "multi_gpu": True},
            }
        )


@pytest.mark.parametrize(
    "path",
    sorted((PROJECT_ROOT / "configs" / "models").rglob("*.yaml")),
)
def test_checked_in_model_configs_obey_schema(path):
    config = load_model_config(path)
    assert config["schema"] == "dpjax.model.v1"
