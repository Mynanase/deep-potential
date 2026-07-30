from __future__ import annotations

import h5py
import numpy as np
import pytest

from dpjax.data import (
    CoordinateTransform,
    DFDataSelection,
    fit_normalizer,
    inverse_preprocess_eta,
    iter_batches,
    load_eta_h5,
    load_h5_vector,
    load_run_preprocessing,
    phase_space_sha256,
    preprocess_eta,
    require_physics_compatible_transform,
    resolve_run_support_indices,
    save_eta_h5,
    sigma_clip_mask,
)


def test_h5_round_trip(tmp_path):
    eta = np.arange(60, dtype=np.float32).reshape(10, 6)
    path = save_eta_h5(eta, tmp_path / "nested" / "eta.h5")

    np.testing.assert_array_equal(load_eta_h5(path), eta)


def test_load_eta_h5_validates_dataset_and_shape(tmp_path):
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("wrong_shape", data=np.zeros((3, 5)))

    with pytest.raises(KeyError, match="eta"):
        load_eta_h5(path)
    with pytest.raises(ValueError, match=r"Expected eta shape \(N, 6\)"):
        load_eta_h5(path, dataset="wrong_shape")


def test_load_h5_vector_and_weighted_normalizer(tmp_path):
    path = tmp_path / "weighted.h5"
    eta = np.array([[0.0] * 6, [10.0] * 6], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("eta", data=eta)
        handle.create_dataset("tracer_weight", data=[1.0, 3.0])

    weights = load_h5_vector(path, "tracer_weight")
    normalizer = fit_normalizer(eta, weights=weights)

    np.testing.assert_allclose(normalizer.mean, np.full(6, 7.5))
    np.testing.assert_allclose(
        normalizer.std,
        np.full(6, np.sqrt(18.75)),
    )


@pytest.mark.parametrize(
    ("transform_type", "params"),
    [
        ("asinh", {"scale": np.array(2.0, dtype=np.float32)}),
        ("log", {"scale": np.array(2.0, dtype=np.float32)}),
        ("power", {"alpha": np.array(0.5, dtype=np.float32)}),
    ],
)
def test_coordinate_transform_round_trip(transform_type, params):
    eta = np.array(
        [[-4.0, -1.0, 0.0, 1.0, 4.0, 9.0]],
        dtype=np.float32,
    )
    transform = CoordinateTransform(
        type=transform_type,
        dims=np.array([0, 2, 4], dtype=np.int32),
        params=params,
    )

    transformed = transform.transform(eta.copy())
    assert not np.array_equal(transformed[:, transform.dims], eta[:, transform.dims])
    np.testing.assert_allclose(
        transform.inverse(transformed.copy()),
        eta,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_preprocessing_round_trip_does_not_mutate_input():
    eta = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=np.float32,
    )
    original = eta.copy()
    transform = CoordinateTransform(
        type="asinh",
        dims=np.array([0, 1, 2], dtype=np.int32),
        params={"scale": np.array(1.5, dtype=np.float32)},
    )
    transformed = transform.transform(eta.copy())
    normalizer = fit_normalizer(transformed)

    eta_std = preprocess_eta(eta, normalizer, transform)
    restored = inverse_preprocess_eta(eta_std, normalizer, transform)

    np.testing.assert_array_equal(eta, original)
    np.testing.assert_allclose(restored, eta, rtol=1.0e-5, atol=1.0e-6)


def test_coordinate_transform_file_round_trip(tmp_path):
    transform = CoordinateTransform(
        type="power",
        dims=np.array([0, 2, 4], dtype=np.int32),
        params={"alpha": np.array(0.5, dtype=np.float32)},
    )
    path = tmp_path / "coord_transform.npz"

    transform.save_npz(path)
    restored = CoordinateTransform.load_npz(path)

    assert restored.type == "power"
    np.testing.assert_array_equal(restored.dims, [0, 2, 4])
    np.testing.assert_allclose(restored.params["alpha"], 0.5)


def test_legacy_coordinate_transform_file_is_safely_disabled(tmp_path):
    path = tmp_path / "legacy_coord_transform.npz"
    np.savez(
        path,
        type="power",
        dims=np.array([0, 1, 2], dtype=np.int32),
        alpha=np.array(0.5, dtype=np.float32),
    )

    with pytest.warns(RuntimeWarning, match="Ignoring legacy"):
        restored = CoordinateTransform.load_npz(path)

    assert restored.type == "none"
    assert restored.dims.size == 0


def test_load_run_preprocessing_and_physics_guard(tmp_path):
    normalizer = fit_normalizer(np.arange(60, dtype=np.float32).reshape(10, 6))
    normalizer.save_npz(tmp_path / "normalizer.npz")

    loaded_normalizer, coordinate_transform = load_run_preprocessing(tmp_path)
    np.testing.assert_array_equal(loaded_normalizer.mean, normalizer.mean)
    assert coordinate_transform is None
    require_physics_compatible_transform(
        coordinate_transform,
        operation="test operation",
    )

    nonlinear = CoordinateTransform(
        type="power",
        dims=np.array([0, 1, 2], dtype=np.int32),
        params={"alpha": np.array(0.5, dtype=np.float32)},
    )
    nonlinear.save_npz(tmp_path / "coord_transform.npz")
    _, coordinate_transform = load_run_preprocessing(tmp_path)

    with pytest.raises(ValueError, match="test operation.*power"):
        require_physics_compatible_transform(
            coordinate_transform,
            operation="test operation",
        )


def test_iter_batches_handles_partial_final_batch():
    eta = np.arange(30, dtype=np.float32).reshape(5, 6)
    batches = list(
        iter_batches(
            eta,
            batch_size=2,
            rng=np.random.default_rng(0),
            shuffle=False,
            drop_remainder=False,
        )
    )

    assert [len(batch) for batch in batches] == [2, 2, 1]
    np.testing.assert_array_equal(np.concatenate(batches), eta)


def test_df_data_selection_round_trip_preserves_source_indices(tmp_path):
    eta = np.arange(30, dtype=np.float32).reshape(5, 6)
    selection = DFDataSelection(
        source_size=5,
        dataset="eta",
        source_sha256=phase_space_sha256(eta),
        clip_sigma=4.5,
        split_seed=1042,
        support_indices=np.array([0, 2, 3, 4]),
        train_indices=np.array([2, 4]),
        val_indices=np.array([0, 3]),
    )
    path = tmp_path / "data_selection.npz"

    selection.save_npz(path)
    restored = DFDataSelection.load_npz(path)

    assert restored.source_size == 5
    assert restored.dataset == "eta"
    assert restored.source_sha256 == phase_space_sha256(eta)
    assert restored.clip_sigma == 4.5
    assert restored.split_seed == 1042
    np.testing.assert_array_equal(restored.support_indices, [0, 2, 3, 4])
    np.testing.assert_array_equal(restored.train_indices, [2, 4])
    np.testing.assert_array_equal(restored.val_indices, [0, 3])


def test_resolve_run_support_indices_validates_data_identity(tmp_path):
    eta = np.arange(24, dtype=np.float32).reshape(4, 6)
    DFDataSelection(
        source_size=4,
        dataset="eta",
        source_sha256=phase_space_sha256(eta),
        clip_sigma=4.5,
        split_seed=7,
        support_indices=np.array([0, 2]),
        train_indices=np.array([2]),
        val_indices=np.array([0]),
    ).save_npz(tmp_path / "data_selection.npz")

    indices, source = resolve_run_support_indices(
        tmp_path,
        eta,
        data_config={"dataset": "eta", "clip_sigma": 4.5},
    )
    np.testing.assert_array_equal(indices, [0, 2])
    assert source == "persisted"

    with pytest.raises(ValueError, match="targets dataset"):
        resolve_run_support_indices(
            tmp_path,
            eta,
            data_config={"dataset": "other", "clip_sigma": 4.5},
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_run_support_indices(
            tmp_path,
            eta[[1, 0, 2, 3]],
            data_config={"dataset": "eta", "clip_sigma": 4.5},
        )


def test_resolve_run_support_indices_reconstructs_legacy_clip(tmp_path):
    eta = np.zeros((8, 6), dtype=np.float32)
    eta[-1] = 100.0
    expected = np.flatnonzero(sigma_clip_mask(eta, 2.0))

    with pytest.warns(RuntimeWarning, match="reconstructed"):
        indices, source = resolve_run_support_indices(
            tmp_path,
            eta,
            data_config={"dataset": "eta", "clip_sigma": 2.0},
        )

    np.testing.assert_array_equal(indices, expected)
    assert source == "reconstructed"
