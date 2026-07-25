from __future__ import annotations

import pytest

from experiments._cli import load_experiment_config


def test_load_experiment_config_applies_json_override():
    config = load_experiment_config(
        "configs/df_halo12_ffjord_v21.yaml",
        '{"train": {"epochs": 2}, "seed": 9}',
    )

    assert config["train"]["epochs"] == 2
    assert config["train"]["batch_size"] == 8192
    assert config["seed"] == 9


def test_load_experiment_config_requires_json_object():
    with pytest.raises(ValueError, match="JSON object"):
        load_experiment_config(
            "configs/df_halo12_ffjord_v21.yaml",
            "[1, 2, 3]",
        )
