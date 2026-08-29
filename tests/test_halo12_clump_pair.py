from __future__ import annotations

import numpy as np
import pytest

from experiments.validation.halo12_clump_pair import (
    _resolved_pair_configs,
    validate_source_partition,
)
from experiments.workflows.config import load_run_spec


def test_halo12_pair_configs_are_identical_except_data_and_output():
    raw = load_run_spec("configs/runs/halo12_raw_no_clip_v1.yaml")
    clean = load_run_spec(
        "configs/runs/halo12_clean_outer_clump_v1.yaml"
    )

    resolved = _resolved_pair_configs(raw, clean)

    assert resolved["df"]["data"]["clip_sigma"] == 0.0
    assert resolved["df"] == raw.resolve_stage_config("df")
    assert resolved["phi"] == raw.resolve_stage_config("phi")


def test_validate_source_partition_returns_ordered_keep_rows():
    keep_rows = validate_source_partition(
        np.array([10, 11, 12, 13, 14]),
        np.array([10, 12, 14]),
        np.array([11, 13]),
    )

    np.testing.assert_array_equal(keep_rows, [0, 2, 4])


@pytest.mark.parametrize(
    ("clean", "removed", "match"),
    [
        (np.array([10, 11, 14]), np.array([11, 13]), "overlap"),
        (np.array([10, 14]), np.array([11, 13]), "complement"),
    ],
)
def test_validate_source_partition_rejects_invalid_pair(clean, removed, match):
    with pytest.raises(ValueError, match=match):
        validate_source_partition(
            np.array([10, 11, 12, 13, 14]),
            clean,
            removed,
        )
