from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from dpjax.diagnostics.df import load_df_evaluation, plot_density_profile
from dpjax.diagnostics.phi import load_phi_evaluation, plot_radial_curves
from dpjax.diagnostics.training import load_metrics, plot_training_metrics


def test_training_diagnostics_load_and_plot_without_writing(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "step,loss,score_p99\n0,2.0,10.0\nstep,loss,score_p99\n1,1.0,8.0\n",
        encoding="utf-8",
    )

    metrics = load_metrics(metrics_path)
    figures = plot_training_metrics(metrics)

    np.testing.assert_allclose(metrics["step"], [0.0, 1.0])
    assert set(figures) == {"loss", "df_score_stats"}


def test_persisted_df_and_phi_diagnostics_are_not_recomputed(tmp_path):
    df_dir = tmp_path / "df"
    phi_dir = tmp_path / "phi"
    df_dir.mkdir()
    phi_dir.mkdir()
    (df_dir / "auriga_df_metrics.json").write_text(
        json.dumps({"n_models": 1}), encoding="utf-8"
    )
    np.savez(
        df_dir / "auriga_df_diagnostics.npz",
        radial_edges=np.array([0.0, 1.0, 2.0]),
        reference_density=np.array([1.0, 0.5]),
        model_density=np.array([0.9, 0.4]),
        model_density_by_model=np.array([[0.9, 0.4]]),
    )
    (phi_dir / "eval_stats.json").write_text(
        json.dumps({"residual_std": 0.2}), encoding="utf-8"
    )
    np.savez(
        phi_dir / "radial_curves.npz",
        r=np.array([1.0, 2.0]),
        phi_learned=np.array([-1.0, -0.5]),
        ar_learned=np.array([-0.2, -0.1]),
        rho_learned=np.array([0.3, 0.1]),
    )

    df_eval = load_df_evaluation(df_dir)
    phi_eval = load_phi_evaluation(phi_dir)
    figures = plot_radial_curves(phi_eval["radial"])
    density_figure = plot_density_profile(df_eval["diagnostics"])

    assert df_eval["metrics"]["n_models"] == 1
    assert phi_eval["stats"]["residual_std"] == 0.2
    assert set(figures) == {"potential", "acceleration", "density"}
    assert density_figure.axes[0].get_ylabel() == "tracer density"
