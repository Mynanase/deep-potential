from __future__ import annotations

import numpy as np
import pytest

from experiments.datasets.plummer import (
    PlummerSphere,
    plummer_df,
    sample_plummer,
    split_train_test,
)
from experiments.validation.plummer import plummer_rv_ideal_grid


def test_sample_plummer_is_reproducible_and_bound():
    first = sample_plummer(
        2_048,
        max_dist=2.0,
        rng=np.random.default_rng(7),
    )
    second = sample_plummer(
        2_048,
        max_dist=2.0,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2_048, 6)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()

    radius = np.linalg.norm(first[:, :3], axis=1)
    speed_squared = np.sum(first[:, 3:] ** 2, axis=1)
    relative_energy = PlummerSphere.psi(radius) - 0.5 * speed_squared
    assert np.max(radius) <= 2.0
    assert np.min(relative_energy) >= 0.0


def test_plummer_df_is_positive_for_samples():
    eta = sample_plummer(128, rng=np.random.default_rng(4))
    density = plummer_df(eta)

    assert density.shape == (128,)
    assert np.all(density > 0.0)


def test_plummer_rv_grid_is_normalized_for_radial_selections():
    full = plummer_rv_ideal_grid(
        r_lim=(0.0, 10.0),
        v_lim=(0.0, 1.5),
        bins=(64, 64),
        radial_selection=(0.0, 10.0),
    )
    cut = plummer_rv_ideal_grid(
        r_lim=(0.0, 10.0),
        v_lim=(0.0, 1.5),
        bins=(64, 64),
        radial_selection=(1.0, 10.0),
    )

    np.testing.assert_allclose(
        np.sum(full["probability_mass"]),
        1.0,
        rtol=2e-3,
    )
    np.testing.assert_allclose(
        np.sum(cut["probability_mass"]),
        1.0,
        rtol=2e-3,
    )
    assert np.all(cut["probability_mass"][:6] == 0.0)


@pytest.mark.parametrize(
    ("n", "max_dist"),
    [(0, None), (-1, None), (4, 0.0), (4, -1.0)],
)
def test_sample_plummer_validates_arguments(n, max_dist):
    with pytest.raises(ValueError):
        sample_plummer(n, max_dist=max_dist)


def test_split_train_test_is_reproducible_and_non_overlapping():
    eta = np.arange(120, dtype=np.float32).reshape(20, 6)
    train, test = split_train_test(
        eta,
        test_n=5,
        rng=np.random.default_rng(3),
    )

    assert train.shape == (15, 6)
    assert test.shape == (5, 6)
    assert set(train[:, 0]).isdisjoint(set(test[:, 0]))
    assert set(np.concatenate((train[:, 0], test[:, 0]))) == set(eta[:, 0])
