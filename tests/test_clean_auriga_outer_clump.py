from __future__ import annotations

import h5py
import numpy as np
import pytest

from experiments.datasets.clean_outer_clump import (
    SELECTION_SCHEMA,
    detect_densest_outer_clump,
    save_selection_artifact,
    summarize_selection,
    write_clean_snapshot,
)
from experiments.datasets.auriga import AurigaSnapshot, load_auriga_snapshot
from experiments.datasets.phase_space import phase_space_sha256


def _synthetic_snapshot(seed: int = 7) -> tuple[AurigaSnapshot, np.ndarray]:
    rng = np.random.default_rng(seed)
    background = rng.uniform(-20.0, 20.0, size=(600, 3))
    cluster = rng.normal(
        loc=np.array([12.0, -8.0, 5.0]),
        scale=0.12,
        size=(80, 3),
    )
    position = np.concatenate([background, cluster], axis=0)
    velocity = rng.normal(size=position.shape)
    eta = np.concatenate([position, velocity], axis=1).astype(np.float32)
    cluster_rows = np.arange(background.shape[0], eta.shape[0])
    snapshot = AurigaSnapshot(
        eta=eta,
        particle_id=np.arange(10_000, 10_000 + eta.shape[0]),
        mass=np.linspace(1.0, 2.0, eta.shape[0], dtype=np.float32),
        potential=-np.linalg.norm(position, axis=1).astype(np.float32),
        source_index=np.arange(eta.shape[0], dtype=np.int64),
    ).validate()
    return snapshot, cluster_rows


def test_detect_densest_outer_clump_recovers_dense_component():
    snapshot, expected = _synthetic_snapshot()

    selection = detect_densest_outer_clump(
        snapshot.eta[:, :3],
        outer_radius_min=5.0,
        linking_length=0.5,
        min_neighbors=10,
        workers=1,
    )

    np.testing.assert_array_equal(selection.removed_indices, expected)
    assert selection.peak_index in expected
    assert selection.keep_indices.size == snapshot.n_particles - expected.size
    assert selection.removed_fraction == pytest.approx(
        expected.size / snapshot.n_particles
    )


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("outer_radius_min", -1.0, "non-negative"),
        ("linking_length", 0.0, "positive"),
        ("min_neighbors", 1, "at least 2"),
    ],
)
def test_detect_densest_outer_clump_validates_parameters(
    keyword,
    value,
    match,
):
    positions = np.ones((20, 3))
    kwargs = {
        "outer_radius_min": 0.0,
        "linking_length": 1.0,
        "min_neighbors": 2,
        keyword: value,
    }
    with pytest.raises(ValueError, match=match):
        detect_densest_outer_clump(positions, **kwargs)


def test_clean_snapshot_and_mask_preserve_source_identity(tmp_path):
    snapshot, expected = _synthetic_snapshot()
    selection = detect_densest_outer_clump(
        snapshot.eta[:, :3],
        outer_radius_min=5.0,
        linking_length=0.5,
        min_neighbors=10,
        workers=1,
    )
    source_hash = phase_space_sha256(snapshot.eta)
    mask_path = save_selection_artifact(
        tmp_path / "mask.npz",
        snapshot=snapshot,
        selection=selection,
        source_sha256=source_hash,
    )
    summary = summarize_selection(snapshot, selection)
    output = write_clean_snapshot(
        tmp_path / "clean.h5",
        snapshot=snapshot,
        selection=selection,
        source_sha256=source_hash,
        mask_path=mask_path,
        summary=summary,
    )

    restored = load_auriga_snapshot(output)
    expected_keep = np.setdiff1d(
        np.arange(snapshot.n_particles),
        expected,
    )
    np.testing.assert_array_equal(restored.source_index, expected_keep)
    np.testing.assert_array_equal(
        restored.particle_id,
        snapshot.particle_id[expected_keep],
    )
    assert restored.tracer_weight.mean() == pytest.approx(1.0, abs=1.0e-6)
    assert restored.attrs["cleaning_schema"] == SELECTION_SCHEMA

    with np.load(mask_path) as artifact:
        assert str(artifact["schema"]) == SELECTION_SCHEMA
        assert str(artifact["source_sha256"]) == source_hash
        np.testing.assert_array_equal(
            artifact["removed_source_index"],
            expected,
        )
        np.testing.assert_array_equal(
            artifact["removed_particle_id"],
            snapshot.particle_id[expected],
        )

    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(
            handle["cleaning/removed_row_index"],
            expected,
        )
        np.testing.assert_array_equal(
            handle["cleaning/removed_source_index"],
            expected,
        )
        assert handle["cleaning"].attrs["source_sha256"] == source_hash
