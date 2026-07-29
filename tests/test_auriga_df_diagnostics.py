from __future__ import annotations

import h5py
import jax.numpy as jnp
import numpy as np

from dpjax.data import Normalizer
from dpjax.plotting import plot_auriga_df_ensemble
from experiments import eval_auriga_df


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
        eval_auriga_df,
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
        eval_auriga_df,
        "sample_apply",
        lambda model, params, key, n, flow_cfg: jnp.asarray(
            model_samples[:n]
        ),
    )
    monkeypatch.setattr(
        eval_auriga_df,
        "_score_in_physical_coordinates",
        lambda model, params, normalizer, flow_cfg, rows, batch_size: -rows,
    )

    output_dir = tmp_path / "evaluation"
    result = eval_auriga_df.evaluate_auriga_df(
        data_path,
        [tmp_path / "seed_42"],
        output_dir,
        n_samples_per_model=128,
        n_score_points=64,
        radial_edges=np.array([0.0, 1.0, 2.0, 5.0]),
        theta_edges=np.array([0.0, np.pi / 2.0, np.pi]),
        phi_edges=np.array([-np.pi, 0.0, np.pi]),
        n_velocity_bins=12,
        spatial_r_edges=np.array([0.0, 1.5, 3.0, 5.0]),
        spatial_z_edges=np.array([-5.0, 0.0, 5.0]),
        spatial_min_cell_count=1,
    )

    assert result["n_models"] == 1
    assert result["score_ensemble"] is None
    assert result["density_semantics"].startswith(
        "Normalized stellar tracer mass density"
    )
    diagnostics_path = output_dir / "auriga_df_diagnostics.npz"
    with np.load(diagnostics_path) as diagnostics:
        assert diagnostics["model_density_by_model"].shape[0] == 1
        assert diagnostics["conditional_r_model_hist"].shape[0] == 1
        assert diagnostics["spatial_model_density"].shape[0] == 1

    figures = plot_auriga_df_ensemble(
        output_dir / "auriga_df_metrics.json",
        diagnostics_path,
        fig_dir=output_dir / "plots",
        dpi=40,
    )

    assert "score_consistency" not in figures
    for filename in (
        "density_profile.png",
        "df_spatial_rz_by_phi.png",
        "velocity_marginals_by_r.png",
        "velocity_marginals_by_theta.png",
        "velocity_marginals_by_phi.png",
        "score_per_dim_hist.png",
    ):
        assert (output_dir / "plots" / filename).exists()
