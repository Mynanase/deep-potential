from __future__ import annotations

from dpjax.config import load_config, merge_config


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
    config = load_config("configs/df_halo12_ffjord_v21.yaml")

    assert config["flow"]["type"] == "ffjord"
    assert config["data"]["transform"]["type"] == "power"
