from __future__ import annotations

import json
import sys

import h5py
import numpy as np

from experiments.diagnostics.evaluation import (
    input_velocity_diagnostics,
    input_velocity_diagnostics_joint,
)
from experiments.plotting.df_diagnostics import (
    plot_input_velocity_distributions,
    plot_input_velocity_joint_summary,
    plot_input_velocity_joint_wedge,
)

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


JOINT_R_EDGES = np.geomspace(0.5, 75.0, 5)


def _sample_joint_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = np.sqrt(rng.uniform(0.5**2, 75.0**2, size=n))
    return directions * radius[:, None]


def test_joint_binning_counts_and_signal_localization():
    rng = np.random.default_rng(41)
    # Deterministic localization probe: an azimuthal unit-velocity signal
    # (v_phi = 40 + Exp(20), right-skewed) injected ONLY into the disk class
    # (|cos| < 0.3) x phi sector 2 ([0, pi/2)); every other cell gets exactly
    # zero velocity (std=0 -> NaN shape stats).
    candidates = _sample_joint_positions(rng, 60_000)
    radius = np.linalg.norm(candidates, axis=1)
    abs_cos = np.abs(candidates[:, 2]) / radius
    azimuth = np.arctan2(candidates[:, 1], candidates[:, 0])
    in_signal_cell = (
        (abs_cos < 0.3) & (azimuth >= 0.0) & (azimuth < np.pi / 2.0)
    )
    signal = candidates[in_signal_cell]
    filler = candidates[~in_signal_cell]

    phi0 = np.arctan2(signal[:, 1], signal[:, 0])
    amplitude = 40.0 + rng.exponential(20.0, size=signal.shape[0])
    signal_velocity = np.stack(
        [
            -np.sin(phi0) * amplitude,
            np.cos(phi0) * amplitude,
            np.zeros(signal.shape[0]),
        ],
        axis=1,
    )
    eta = np.concatenate(
        [
            np.concatenate([filler, signal], axis=0),
            np.concatenate(
                [np.zeros((filler.shape[0], 3)), signal_velocity], axis=0
            ),
        ],
        axis=1,
    )

    diagnostics = input_velocity_diagnostics_joint(
        eta,
        r_edges=JOINT_R_EDGES,
        n_velocity_bins=32,
    )

    assert int(np.sum(diagnostics["joint_count"])) == eta.shape[0]
    skew = diagnostics["joint_skewness"]
    mean = diagnostics["joint_mean"]
    counts = diagnostics["joint_count"]
    for r_index in range(JOINT_R_EDGES.size - 1):
        if counts[r_index, 0, 2] < 100:
            # Sparse inner bins: shape stats are noise there, which is
            # exactly why wedge figures mask on N_eff.
            assert np.isnan(skew[r_index, 0, 2, 2]) or counts[r_index, 0, 2] < 100
            continue
        # Signal cell: pure shifted exponential in v_phi (skew = 2, mean = 60).
        assert skew[r_index, 0, 2, 2] > 1.5
        np.testing.assert_allclose(mean[r_index, 0, 2, 2], 60.0, rtol=0.05)
        # Same radial/theta cell, neighboring phi sector: exact zeros.
        assert np.isnan(skew[r_index, 0, 3, 2])
        # Polar class, same sector: exact zeros.
        assert np.isnan(skew[r_index, 2, 2, 2])


def test_joint_wedge_plots_mask_and_render():
    rng = np.random.default_rng(43)
    eta = np.concatenate(
        [
            _sample_joint_positions(rng, 3_000),
            rng.normal(0.0, 10.0, size=(3_000, 3)),
        ],
        axis=1,
    )
    diagnostics = input_velocity_diagnostics_joint(
        eta,
        r_edges=JOINT_R_EDGES,
        n_velocity_bins=16,
    )
    import matplotlib.pyplot as plt

    wedge = plot_input_velocity_joint_wedge(
        diagnostics,
        theta_index=0,
        phi_index=2,
        min_effective_count=0.0,
        dpi=40,
    )
    assert wedge.axes
    plt.close(wedge)
    # Aggressive threshold masks every cell: figure still renders.
    masked = plot_input_velocity_joint_wedge(
        diagnostics,
        theta_index=0,
        phi_index=2,
        min_effective_count=1.0e12,
        dpi=40,
    )
    assert masked.axes
    plt.close(masked)
    summary = plot_input_velocity_joint_summary(
        diagnostics,
        velocity_index=2,
        metric="skewness",
        min_effective_count=0.0,
        dpi=40,
    )
    assert summary.axes
    plt.close(summary)


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
            "--joint",
            "--joint-r-bins",
            "3",
            "--joint-r-min",
            "0.5",
            "--joint-r-max",
            "75.0",
        ],
    )
    assert input_data_check.main() == 0

    assert (output_dir / "input_velocity_diagnostics.npz").is_file()
    assert (output_dir / "input_velocity_summary.json").is_file()
    for name in input_data_check.FIGURE_NAMES.values():
        assert (output_dir / name).is_file()
    with np.load(output_dir / "input_velocity_diagnostics.npz") as data:
        assert data["input_r_hist"].shape == (2, 3, 16)

    joint_dir = output_dir / "joint"
    assert (joint_dir / "input_velocity_joint.npz").is_file()
    with np.load(joint_dir / "input_velocity_joint.npz") as data:
        assert data["joint_count"].shape == (3, 3, 4)
        # Joint mode intentionally covers r >= joint-r-min (default 0.5 kpc);
        # particles below that are outside its scope.
        n_in_range = int(np.sum(np.linalg.norm(positions, axis=1) >= 0.5))
        assert int(np.sum(data["joint_count"])) == n_in_range
    theta_names = ("disk", "intermediate", "polar")
    for theta_index, theta_name in enumerate(theta_names):
        for phi_index in range(4):
            assert (joint_dir / f"wedge_{theta_name}_phi{phi_index}.png").is_file()
    assert (joint_dir / "wedge_summary_v_phi_skewness.png").is_file()
    assert (joint_dir / "wedge_summary_v_phi_excess_kurtosis.png").is_file()
    pdf_path = output_dir / "input_velocity_validation.pdf"
    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert pdf_path.read_bytes()[:5] == b"%PDF-"
    summary = json.loads(
        (output_dir / "input_velocity_summary.json").read_text()
    )
    assert set(summary["joint"]["wedges"]) == {
        f"{name}_phi{index}" for name in theta_names for index in range(4)
    }
