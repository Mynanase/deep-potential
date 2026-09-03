from __future__ import annotations

import sys

import h5py
import numpy as np

from experiments.diagnostics.evaluation import input_velocity_diagnostics
from experiments.plotting.df_diagnostics import plot_input_velocity_distributions

CONDITIONING_EDGES = {
    "r": np.array([0.0, 1.0, 2.0, 3.0]),
    "theta": np.array([0.0, np.pi / 2.0, np.pi]),
    "phi": np.array([-np.pi, 0.0, np.pi]),
}


def _isotropic_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = rng.uniform(0.05, 2.95, size=n)
    return directions * radius[:, None]


def _radial_velocity_field(
    positions: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    return unit * amplitudes[:, None]


def test_input_velocity_moments_match_direct_computation():
    rng = np.random.default_rng(7)
    n = 4_000
    positions = _isotropic_positions(rng, n)
    velocities = rng.normal(size=(n, 3)) * np.array([1.0, 2.0, 0.5])
    eta = np.concatenate([positions, velocities], axis=1)
    weights = rng.uniform(0.5, 2.0, size=n)

    diagnostics = input_velocity_diagnostics(
        eta,
        weights=weights,
        conditioning_edges=CONDITIONING_EDGES,
        n_velocity_bins=24,
    )

    radius = np.linalg.norm(positions, axis=1)
    x, y, z = positions.T
    vx, vy, vz = velocities.T
    spherical_radius = np.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / spherical_radius
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = np.arctan2(y, x)
    v_r = (
        vx * sin_theta * np.cos(phi)
        + vy * sin_theta * np.sin(phi)
        + vz * cos_theta
    )

    for bin_index, (left, right) in enumerate(
        zip(CONDITIONING_EDGES["r"][:-1], CONDITIONING_EDGES["r"][1:])
    ):
        mask = (radius >= left) & (radius < right)
        expected_mean = np.sum(weights[mask] * v_r[mask]) / np.sum(weights[mask])
        np.testing.assert_allclose(
            diagnostics["input_r_mean"][bin_index, 0],
            expected_mean,
            rtol=1.0e-10,
        )
        count = int(np.count_nonzero(mask))
        assert diagnostics["input_r_count"][bin_index] == count
        assert (
            diagnostics["input_r_effective_count"][bin_index]
            <= count
        )

    edges = diagnostics["input_velocity_edges"]
    for velocity_index in range(3):
        widths = np.diff(edges[velocity_index])
        probability = np.sum(
            diagnostics["input_r_hist"][:, velocity_index, :] * widths[None, :],
            axis=1,
        )
        occupied = diagnostics["input_r_count"] > 0
        np.testing.assert_allclose(probability[occupied], 1.0, atol=1.0e-12)
        assert np.isnan(probability[~occupied]).all()

    total_counts = {
        coordinate: int(
            np.sum(diagnostics[f"input_{coordinate}_count"])
        )
        for coordinate in ("r", "theta", "phi")
    }
    assert total_counts == {"r": n, "theta": n, "phi": n}


def test_input_velocity_shape_stats_detect_non_gaussian_radial_field():
    rng = np.random.default_rng(11)
    n = 20_000
    positions = _isotropic_positions(rng, n)
    # Shifted exponential radial speeds: strongly right-skewed, heavy tail.
    amplitudes = -np.log(rng.uniform(size=n)) + 0.5
    eta = np.concatenate(
        [positions, _radial_velocity_field(positions, amplitudes)], axis=1
    )

    diagnostics = input_velocity_diagnostics(
        eta,
        conditioning_edges=CONDITIONING_EDGES,
        n_velocity_bins=32,
    )

    for bin_index in range(CONDITIONING_EDGES["r"].size - 1):
        skew = diagnostics["input_r_skewness"][bin_index, 0]
        kurtosis = diagnostics["input_r_excess_kurtosis"][bin_index, 0]
        assert skew > 1.0
        assert kurtosis > 1.0
        # Tangential components vanish analytically for a radial field;
        # numerically they only carry floating-point noise.
        assert diagnostics["input_r_std"][bin_index, 1] < 1.0e-9
        assert diagnostics["input_r_std"][bin_index, 2] < 1.0e-9


def test_input_velocity_shape_stats_gaussian_radial_field_is_gaussian():
    rng = np.random.default_rng(23)
    n = 60_000
    positions = _isotropic_positions(rng, n)
    amplitudes = rng.normal(0.0, 1.0, size=n)
    eta = np.concatenate(
        [positions, _radial_velocity_field(positions, amplitudes)], axis=1
    )

    diagnostics = input_velocity_diagnostics(
        eta,
        conditioning_edges=CONDITIONING_EDGES,
        n_velocity_bins=48,
    )

    occupied = diagnostics["input_r_count"] > 0
    skew = diagnostics["input_r_skewness"][occupied, 0]
    kurtosis = diagnostics["input_r_excess_kurtosis"][occupied, 0]
    assert np.all(np.abs(skew) < 0.3)
    assert np.all(np.abs(kurtosis) < 0.6)


def test_input_velocity_empty_bins_are_nan():
    rng = np.random.default_rng(31)
    n = 200
    # All particles inside r < 1 so the outer r bins stay empty.
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * rng.uniform(0.05, 0.9, size=n)[:, None]
    eta = np.concatenate([positions, rng.normal(size=(n, 3))], axis=1)

    diagnostics = input_velocity_diagnostics(
        eta,
        conditioning_edges=CONDITIONING_EDGES,
        n_velocity_bins=16,
    )

    assert list(diagnostics["input_r_count"]) == [n, 0, 0]
    assert np.isnan(diagnostics["input_r_hist"][1]).all()
    assert np.isnan(diagnostics["input_r_mean"][1]).all()
    assert np.isnan(diagnostics["input_r_skewness"][1]).all()
    assert np.isnan(diagnostics["input_r_excess_kurtosis"][1]).all()


def test_plot_input_velocity_distributions_smoke():
    rng = np.random.default_rng(5)
    n = 1_000
    positions = _isotropic_positions(rng, n)
    eta = np.concatenate([positions, rng.normal(size=(n, 3))], axis=1)
    diagnostics = input_velocity_diagnostics(
        eta,
        conditioning_edges=CONDITIONING_EDGES,
        n_velocity_bins=16,
    )
    for coordinate in ("r", "theta", "phi"):
        figure = plot_input_velocity_distributions(
            diagnostics,
            coordinate,
            dpi=40,
        )
        assert figure.axes
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_input_data_check_cli_end_to_end(tmp_path, monkeypatch):
    from experiments.validation import input_data_check

    rng = np.random.default_rng(17)
    n = 500
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * rng.uniform(0.05, 2.9, size=n)[:, None]
    eta = np.concatenate([positions, rng.normal(size=(n, 3))], axis=1)

    data_path = tmp_path / "snapshot.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("eta", data=eta.astype(np.float32))
        handle.create_dataset("tracer_weight", data=np.ones(n))

    output_dir = tmp_path / "validation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "input_data_check",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--radial-edges",
            "0,1,3",
            "--n-theta-bins",
            "2",
            "--n-phi-bins",
            "4",
            "--n-velocity-bins",
            "16",
        ],
    )
    assert input_data_check.main() == 0

    assert (output_dir / "input_velocity_diagnostics.npz").is_file()
    assert (output_dir / "input_velocity_summary.json").is_file()
    for name in input_data_check.FIGURE_NAMES.values():
        assert (output_dir / name).is_file()
    with np.load(output_dir / "input_velocity_diagnostics.npz") as data:
        assert data["input_r_hist"].shape == (2, 3, 16)
