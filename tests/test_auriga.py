from __future__ import annotations

import h5py
import numpy as np
import pytest

from dpjax.datasets.auriga import (
    AURIGA_SCHEMA,
    AurigaSnapshot,
    align_indices_by_eta,
    align_indices_by_id,
    align_snapshot,
    center_snapshot,
    classify_kinematic_components,
    load_auriga_snapshot,
    save_auriga_snapshot,
    select_snapshot,
)
from dpjax.evaluation import (
    acceleration_error_metrics,
    cartesian_to_spherical_phase_space,
    conditional_velocity_diagnostics,
    cylindrical_rz_density_by_phi,
    density_profile_metrics,
    potential_error_metrics,
    radial_acceleration_profile,
    score_ensemble_metrics,
    spherical_density_profile,
    stein_score_metrics,
)
from experiments.prepare_auriga import prepare_auriga


def _eta() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 20.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 30.0],
            [4.0, 0.0, 0.0, 40.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_load_vector_gadget_layout(tmp_path):
    eta = _eta()
    path = tmp_path / "snapshot.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("PartType4")
        group.create_dataset("Coordinates", data=eta[:, :3])
        group.create_dataset("Velocities", data=eta[:, 3:])
        group.create_dataset("ParticleIDs", data=[101, 102, 103, 104])
        group.create_dataset("Masses", data=[1.0, 1.1, 1.2, 1.3])
        group.create_dataset("Potential", data=[-4.0, -3.0, -2.0, -1.0])
        group.create_dataset(
            "Acceleration",
            data=-eta[:, :3],
        )
        group.create_dataset("kinematic_label", data=[0, 1, 2, 3])
        group.create_dataset("lambda_z", data=[0.9, 0.5, 0.0, -0.5])

    snapshot = load_auriga_snapshot(path)

    np.testing.assert_array_equal(snapshot.eta, eta)
    np.testing.assert_array_equal(snapshot.particle_id, [101, 102, 103, 104])
    np.testing.assert_allclose(snapshot.acceleration, -eta[:, :3])
    np.testing.assert_array_equal(snapshot.component, [0, 1, 2, 3])
    np.testing.assert_allclose(snapshot.circularity, [0.9, 0.5, 0.0, -0.5])
    np.testing.assert_array_equal(snapshot.source_index, np.arange(4))
    assert snapshot.attrs["source_group"] == "/PartType4"


def test_load_scalar_layout_and_standard_round_trip(tmp_path):
    eta = _eta()
    raw_path = tmp_path / "scalar.hdf5"
    with h5py.File(raw_path, "w") as handle:
        group = handle.create_group("PartType4")
        for index, name in enumerate(("x", "y", "z", "vx", "vy", "vz")):
            group.create_dataset(name, data=eta[:, index])
        group.create_dataset("ParticleIDs", data=[11, 12, 13, 14])

    snapshot = load_auriga_snapshot(raw_path)
    output = save_auriga_snapshot(
        snapshot,
        tmp_path / "prepared.h5",
        length_unit="kpc",
        velocity_unit="km/s",
        potential_unit="(km/s)^2",
    )
    restored = load_auriga_snapshot(output)

    np.testing.assert_array_equal(restored.eta, eta)
    np.testing.assert_array_equal(restored.particle_id, [11, 12, 13, 14])
    assert restored.attrs["schema"] == AURIGA_SCHEMA
    assert restored.attrs["length_unit"] == "kpc"


def test_particle_id_alignment_reorders_all_truth_fields():
    source = AurigaSnapshot(
        eta=_eta(),
        particle_id=np.array([40, 10, 30, 20]),
        mass=np.array([4.0, 1.0, 3.0, 2.0]),
        potential=np.array([-4.0, -1.0, -3.0, -2.0]),
        source_index=np.arange(4),
    ).validate()
    target = AurigaSnapshot(
        eta=source.eta[[1, 3, 2]],
        particle_id=np.array([10, 20, 30]),
    ).validate()

    indices = align_indices_by_id(source.particle_id, target.particle_id)
    np.testing.assert_array_equal(indices, [1, 3, 2])

    aligned, method = align_snapshot(source, target)
    assert method == "particle_id"
    np.testing.assert_array_equal(aligned.eta, target.eta)
    np.testing.assert_array_equal(aligned.mass, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(aligned.source_index, [1, 3, 2])


def test_id_alignment_rejects_duplicates_and_missing():
    with pytest.raises(ValueError, match="Source particle IDs are not unique"):
        align_indices_by_id(np.array([1, 1, 2]), np.array([1, 2]))
    with pytest.raises(ValueError, match="absent"):
        align_indices_by_id(np.array([1, 2, 3]), np.array([1, 4]))


def test_eta_alignment_uses_per_dimension_tolerance():
    source = _eta().astype(np.float64)
    target = source[[2, 0]].copy()
    target[:, :3] += 5.0e-5
    target[:, 3:] += 5.0e-3

    indices = align_indices_by_eta(
        source,
        target,
        atol=[1.0e-4] * 3 + [1.0e-2] * 3,
        workers=1,
    )

    np.testing.assert_array_equal(indices, [2, 0])


def test_eta_alignment_rejects_missing_and_ambiguous_rows():
    source = _eta().astype(np.float64)
    with pytest.raises(ValueError, match="no source match"):
        align_indices_by_eta(source, source[[0]] + 10.0, atol=1.0e-3)

    duplicated = np.concatenate(
        [source, source[[0]] + 1.0e-6],
        axis=0,
    )
    with pytest.raises(ValueError, match="multiple source matches"):
        align_indices_by_eta(
            duplicated,
            source[[0]],
            atol=1.0e-4,
            workers=1,
        )


def test_center_select_and_prepare_preserve_source_mapping(tmp_path):
    eta = _eta()
    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as handle:
        group = handle.create_group("PartType4")
        group.create_dataset("Coordinates", data=eta[:, :3])
        group.create_dataset("Velocities", data=eta[:, 3:])
        group.create_dataset("ParticleIDs", data=[1, 2, 3, 4])
        group.create_dataset("Masses", data=[1.0, 2.0, 3.0, 4.0])
        group.create_dataset("Potential", data=[-4.0, -3.0, -2.0, -1.0])
        group.create_dataset("kinematic_label", data=[0, 1, 2, 3])

    snapshot = load_auriga_snapshot(raw)
    centered = center_snapshot(
        snapshot,
        position_center=[1.0, 0.0, 0.0],
        velocity_center=[10.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(centered.eta[0], np.zeros(6))
    selected = select_snapshot(snapshot, r_min=1.5, r_max=3.5)
    np.testing.assert_array_equal(selected.source_index, [1, 2])
    selected_components = select_snapshot(
        snapshot,
        components=["cold", "hot"],
    )
    np.testing.assert_array_equal(selected_components.source_index, [0, 2])

    output, prepared, method = prepare_auriga(
        raw,
        tmp_path / "prepared.h5",
        r_max=3.0,
        components=["warm", "hot"],
        weight_by_mass=True,
        length_unit="kpc",
        velocity_unit="km/s",
    )
    assert method is None
    assert prepared.n_particles == 2
    restored = load_auriga_snapshot(output)
    np.testing.assert_array_equal(restored.source_index, [1, 2])
    np.testing.assert_array_equal(restored.component, [1, 2])
    np.testing.assert_allclose(restored.tracer_weight, [0.8, 1.2])


def test_component_selection_requires_labels():
    snapshot = AurigaSnapshot(eta=_eta()).validate()
    with pytest.raises(ValueError, match="no component labels"):
        select_snapshot(snapshot, components=["cold"])


def test_recompute_kinematic_components_uses_potential():
    eta = np.tile(_eta(), (3, 1))
    snapshot = AurigaSnapshot(
        eta=eta,
        potential=np.linspace(-100.0, -1.0, eta.shape[0]),
    ).validate()

    classified = classify_kinematic_components(snapshot, bins=2)

    assert classified.component.shape == (eta.shape[0],)
    assert classified.circularity.shape == (eta.shape[0],)
    assert np.all(np.isfinite(classified.circularity))
    assert "Potential+0.5*v^2" in classified.attrs["component_definition"]


def test_potential_metrics_remove_only_additive_offset():
    truth = np.array([-4.0, -2.0, 1.0, 5.0])
    predicted = truth - 17.5

    metrics, aligned = potential_error_metrics(predicted, truth)

    np.testing.assert_allclose(aligned, truth)
    assert metrics["fitted_additive_offset"] == pytest.approx(17.5)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)


def test_acceleration_metrics_and_radial_profile():
    positions = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    truth = -positions
    predicted = truth * 1.1

    metrics = acceleration_error_metrics(predicted, truth)
    profile = radial_acceleration_profile(
        positions,
        predicted,
        truth,
        n_bins=2,
    )

    assert metrics["vector_relative_l2"] == pytest.approx(0.1)
    assert metrics["median_relative_error"] == pytest.approx(0.1)
    assert metrics["median_cosine_similarity"] == pytest.approx(1.0)
    assert sum(row["n"] for row in profile["bins"]) == 3


def test_density_and_score_diagnostics():
    positions = np.array(
        [[0.25, 0.0, 0.0], [0.75, 0.0, 0.0], [1.5, 0.0, 0.0]]
    )
    profile = spherical_density_profile(
        positions,
        weights=np.array([1.0, 2.0, 3.0]),
        edges=np.array([0.0, 0.5, 1.0, 2.0]),
    )
    np.testing.assert_allclose(
        profile["shell_probability"],
        [1 / 6, 2 / 6, 3 / 6],
    )
    density_metrics = density_profile_metrics(
        profile["density"] * 1.1,
        profile["density"],
    )
    assert density_metrics["median_fractional_error"] == pytest.approx(0.1)

    rng = np.random.default_rng(4)
    eta = rng.normal(size=(1000, 6))
    exact_score = -eta
    scores = np.stack([exact_score, exact_score * 1.01], axis=0)
    ensemble = score_ensemble_metrics(scores)
    stein = stein_score_metrics(eta, exact_score)

    assert ensemble["finite_point_fraction"] == pytest.approx(1.0)
    assert ensemble["pairwise_cosine_median"] == pytest.approx(1.0)
    assert stein["mean_score_l2"] < 0.2


def test_cartesian_to_spherical_phase_space_velocity_basis():
    eta = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 2.0, 4.0, 5.0, 6.0],
        ]
    )

    spherical = cartesian_to_spherical_phase_space(eta)

    np.testing.assert_allclose(
        spherical[0],
        [1.0, np.pi / 2.0, 0.0, 1.0, -3.0, 2.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        spherical[1],
        [2.0, 0.0, 0.0, 6.0, 4.0, 5.0],
        atol=1.0e-12,
    )


def test_conditional_velocity_histograms_marginalize_other_coordinates():
    rng = np.random.default_rng(12)
    eta = rng.normal(size=(256, 6))
    eta[:, :3] += np.array([2.0, 0.5, -0.25])
    diagnostics = conditional_velocity_diagnostics(
        eta,
        eta[None, ...],
        reference_weights=np.ones(eta.shape[0]),
        conditioning_edges={
            "r": np.array([0.0, 10.0]),
            "theta": np.array([0.0, np.pi]),
            "phi": np.array([-np.pi, np.pi]),
        },
        n_velocity_bins=16,
    )

    for coordinate in ("r", "theta", "phi"):
        np.testing.assert_allclose(
            diagnostics[coordinate]["reference_hist"],
            diagnostics[coordinate]["model_hist"][0],
        )
        np.testing.assert_allclose(
            diagnostics[coordinate]["wasserstein"],
            0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            diagnostics[coordinate]["ks_histogram"],
            0.0,
            atol=1.0e-12,
        )


def test_cylindrical_rz_density_uses_exact_cell_volume():
    positions = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]
    )
    diagnostics = cylindrical_rz_density_by_phi(
        positions,
        positions[None, ...],
        reference_weights=np.ones(2),
        phi_edges=np.array([-np.pi, np.pi]),
        cylindrical_radius_edges=np.array([0.0, 1.0, 2.0]),
        z_edges=np.array([-1.0, 1.0]),
        min_cell_count=1,
    )

    volume = diagnostics["cell_volume"]
    np.testing.assert_allclose(volume[0, :, 0], [2.0 * np.pi, 6.0 * np.pi])
    assert np.sum(diagnostics["reference_density"] * volume) == pytest.approx(
        1.0
    )
    np.testing.assert_allclose(
        diagnostics["model_density"][0],
        diagnostics["reference_density"],
    )
