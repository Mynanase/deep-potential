from __future__ import annotations

import json

import numpy as np

from experiments.datasets.auriga import AurigaSnapshot
from experiments.diagnostics import (
    load_validation_diagnostics,
    load_validation_metrics,
)
from experiments.validation.auriga_truth import evaluate_auriga_truth
from experiments.validation.plummer import (
    evaluate_plummer_truth,
    plummer_phi,
    plummer_rho,
)


class _IdentityNormalizer:
    mean = np.zeros(6, dtype=np.float32)
    std = np.ones(6, dtype=np.float32)


def _patch_model_loading(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "load_run_preprocessing",
        lambda _: (_IdentityNormalizer(), None),
    )
    monkeypatch.setattr(
        module,
        "require_physics_compatible_transform",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(module, "load_phi", lambda _: (object(), {}, {}))


def test_plummer_truth_validation_writes_data_only_and_masks_cut(tmp_path, monkeypatch):
    import experiments.validation.plummer as module

    _patch_model_loading(monkeypatch, module)

    def exact_fields(_model, _params, positions, _mean, _std, *, batch_size):
        del batch_size
        radius = np.linalg.norm(positions, axis=1)
        acceleration = -positions * (1.0 + radius[:, None] ** 2) ** -1.5
        return {
            "potential": plummer_phi(radius) - 3.0,
            "acceleration": acceleration,
            "density": plummer_rho(radius),
        }

    monkeypatch.setattr(module, "_predict_plummer_fields", exact_fields)
    output_dir = tmp_path / "validation" / "plummer"

    result = evaluate_plummer_truth(
        tmp_path / "df",
        tmp_path / "phi",
        output_dir,
        n_eval=128,
        batch_size=32,
        r_min=1.0,
        r_max=3.0,
        n_r=16,
        slice_grid=17,
        slice_rmax=3.0,
    )

    assert result["metrics"]["potential"]["rmse"] < 1.0e-6
    assert result["metrics"]["density"]["rmse"] < 1.0e-6
    diagnostics = load_validation_diagnostics(output_dir)
    assert not diagnostics["slice_support_mask"][8, 8]
    assert np.all(diagnostics["sample_radius"] >= 1.0)
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "diagnostics.npz").is_file()
    assert not list(output_dir.glob("*.png"))


def test_auriga_potential_only_validation_fits_one_offset(tmp_path, monkeypatch):
    import experiments.validation.auriga_truth as module

    positions = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [1.0, 1.0, -0.5], [2.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    snapshot = AurigaSnapshot(
        eta=np.column_stack([positions, np.zeros_like(positions)]),
        potential=np.sum(positions, axis=1),
        source_index=np.arange(positions.shape[0]),
    ).validate()
    monkeypatch.setattr(module, "load_auriga_snapshot", lambda _: snapshot)
    _patch_model_loading(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_predict_potential",
        lambda _model, _params, positions_std, *, batch_size: (
            np.sum(positions_std, axis=1) - 5.0
        ),
    )
    output_dir = tmp_path / "validation" / "auriga"

    result = evaluate_auriga_truth(
        tmp_path / "data.h5",
        tmp_path / "df",
        tmp_path / "phi",
        output_dir,
        n_eval=None,
        slice_phi_bins=2,
        slice_r_bins=2,
        slice_z_bins=2,
        slice_min_count=1,
    )

    assert result["metrics"]["available_truth"] == {
        "potential": True,
        "acceleration": False,
        "total_density": False,
    }
    assert result["metrics"]["potential"]["fitted_additive_offset"] == 5.0
    diagnostics = load_validation_diagnostics(output_dir)
    assert "sample_truth_potential" in diagnostics
    assert "sample_truth_acceleration" not in diagnostics
    assert not any("density" in key for key in diagnostics)


def test_auriga_acceleration_only_validation_does_not_require_potential(
    tmp_path,
    monkeypatch,
):
    import experiments.validation.auriga_truth as module

    positions = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    snapshot = AurigaSnapshot(
        eta=np.column_stack([positions, np.zeros_like(positions)]),
        acceleration=-positions,
        source_index=np.arange(positions.shape[0]),
    ).validate()
    monkeypatch.setattr(module, "load_auriga_snapshot", lambda _: snapshot)
    _patch_model_loading(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_predict_acceleration",
        lambda _model, _params, positions_std, _std, *, batch_size: -positions_std,
    )
    output_dir = tmp_path / "validation" / "auriga"

    result = evaluate_auriga_truth(
        tmp_path / "data.h5",
        tmp_path / "df",
        tmp_path / "phi",
        output_dir,
        n_eval=None,
        radial_bins=2,
    )

    assert result["metrics"]["available_truth"]["potential"] is False
    assert result["metrics"]["acceleration"]["rmse_vector"] == 0.0
    diagnostics = load_validation_diagnostics(output_dir)
    assert "sample_truth_acceleration" in diagnostics
    assert "sample_truth_potential" not in diagnostics
    assert load_validation_metrics(output_dir)["available_truth"]["total_density"] is False


def test_validation_loaders_fail_explicitly(tmp_path):
    import pytest

    for loader in (load_validation_metrics, load_validation_diagnostics):
        with pytest.raises(FileNotFoundError, match="truth-validation artifact"):
            loader(tmp_path)


def test_validation_metrics_are_plain_json(tmp_path):
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    (validation_dir / "metrics.json").write_text(
        json.dumps({"kind": "one-off"}),
        encoding="utf-8",
    )
    assert load_validation_metrics(validation_dir) == {"kind": "one-off"}
