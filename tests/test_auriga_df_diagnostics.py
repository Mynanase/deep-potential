from __future__ import annotations

import h5py
import jax.numpy as jnp
import numpy as np

from dpjax.normalization import Normalizer
from experiments.diagnostics import load_df_diagnostics
from experiments.diagnostics import load_df_samples
from experiments.plotting import (
    plot_density_profile,
    plot_radial_speed_comparison,
    plot_score_distribution,
    plot_velocity_marginals,
)
from experiments.workflows.evaluation import df as eval_df


def test_single_model_df_evaluation_and_plotting(tmp_path, monkeypatch):
    rng = np.random.default_rng(31)
    eta = rng.normal(size=(256, 6)).astype(np.float32)
    eta[:, :3] *= 1.5
    tracer_weight = np.linspace(0.8, 1.2, eta.shape[0]).astype(np.float32)
    data_path = tmp_path / "halo.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("eta", data=eta)
        handle.create_dataset("tracer_weight", data=tracer_weight)

    model_samples = eta[:128].copy()
    normalizer = Normalizer(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
    )
    monkeypatch.setattr(
        eval_df,
        "load_df",
        lambda run_dir: (
            object(),
            {},
            normalizer,
            {"flow": {}},
            None,
        ),
    )
    monkeypatch.setattr(
        eval_df,
        "sample_apply",
        lambda model, params, key, n, flow_cfg: jnp.asarray(
            model_samples[:n]
        ),
    )
    monkeypatch.setattr(
        eval_df,
        "_score_in_physical_coordinates",
        lambda model, params, normalizer, flow_cfg, rows, batch_size: -rows,
    )

    output_dir = tmp_path / "evaluation"
    result = eval_df.evaluate_df_diagnostics(
        data_path,
        [tmp_path / "seed_42"],
        output_dir,
        n_samples_per_model=128,
        n_score_points=64,
        radial_edges=np.array([0.0, 1.0, 2.0, 5.0]),
        theta_edges=np.array([0.0, np.pi / 2.0, np.pi]),
        phi_edges=np.array([-np.pi, 0.0, np.pi]),
        n_velocity_bins=12,
    )

    assert result["n_models"] == 1
    assert result["score_ensemble"] is None
    assert result["density_semantics"].startswith(
        "Normalized stellar tracer mass density"
    )
    diagnostics_path = output_dir / "df_diagnostics.npz"
    with np.load(diagnostics_path) as diagnostics:
        assert diagnostics["model_density_by_model"].shape[0] == 1
        assert diagnostics["conditional_r_model_hist"].shape[0] == 1

    assert (output_dir / "df_metrics.json").is_file()
    assert (output_dir / "df_samples.npz").is_file()

    diagnostics = load_df_diagnostics(output_dir)
    samples = load_df_samples(output_dir)
    figures = [
        plot_density_profile(diagnostics, dpi=40),
        plot_velocity_marginals(diagnostics, "r", dpi=40),
        plot_velocity_marginals(diagnostics, "theta", dpi=40),
        plot_velocity_marginals(diagnostics, "phi", dpi=40),
        plot_score_distribution(diagnostics, dpi=40),
        plot_radial_speed_comparison(
            samples["reference_eta"],
            samples["model_eta_by_model"][0],
            radius_range=(0.0, 5.0),
            speed_range=(0.0, 5.0),
            bins=8,
            dpi=40,
        ),
        plot_radial_speed_comparison(
            None,
            samples["model_eta_by_model"][0],
            reference_probability_mass=np.full((8, 8), 1.0 / 64.0),
            radius_range=(0.0, 5.0),
            speed_range=(0.0, 5.0),
            bins=(8, 8),
            dpi=40,
        ),
    ]
    assert all(figure.axes for figure in figures)
    assert not (output_dir / "plots").exists()
