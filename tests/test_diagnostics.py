from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.diagnostics import (
    load_df_diagnostics,
    load_df_metrics,
    load_df_samples,
    load_metrics,
    load_phi_diagnostics,
    load_phi_metrics,
    plot_training_metrics,
)
from experiments.plotting import (
    plot_density_profile,
    plot_mass_density_profile,
    plot_potential_profile,
    plot_radial_acceleration_profile,
)


def test_training_diagnostics_load_and_plot_without_writing(tmp_path):
    import matplotlib.pyplot as plt

    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "step,loss,score_p99\n0,2.0,10.0\nstep,loss,score_p99\n1,1.0,8.0\n",
        encoding="utf-8",
    )

    metrics = load_metrics(metrics_path)
    figures = plot_training_metrics(metrics)

    np.testing.assert_allclose(metrics["step"], [0.0, 1.0])
    assert set(figures) == {"loss", "df_score_stats"}
    for figure in figures.values():
        plt.close(figure)


def test_persisted_df_and_phi_diagnostics_are_not_recomputed(tmp_path):
    import matplotlib.pyplot as plt

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "df_metrics.json").write_text(
        json.dumps({"n_models": 1}), encoding="utf-8"
    )
    np.savez(
        eval_dir / "df_diagnostics.npz",
        radial_edges=np.array([0.0, 1.0, 2.0]),
        reference_density=np.array([1.0, 0.5]),
        model_density=np.array([0.9, 0.4]),
        model_density_by_model=np.array([[0.9, 0.4]]),
    )
    np.savez(
        eval_dir / "df_samples.npz",
        reference_eta=np.zeros((2, 6)),
        model_eta_by_model=np.zeros((1, 2, 6)),
    )
    (eval_dir / "phi_metrics.json").write_text(
        json.dumps({"residual_std": 0.2}), encoding="utf-8"
    )
    np.savez(
        eval_dir / "phi_diagnostics.npz",
        radial_r=np.array([1.0, 2.0]),
        radial_phi_shifted=np.array([-1.0, -0.5]),
        radial_acceleration=np.array([-0.2, -0.1]),
        radial_density=np.array([0.3, 0.1]),
    )

    df_metrics = load_df_metrics(eval_dir)
    df_diagnostics = load_df_diagnostics(eval_dir)
    df_samples = load_df_samples(eval_dir)
    phi_metrics = load_phi_metrics(eval_dir)
    phi_diagnostics = load_phi_diagnostics(eval_dir)
    figures = {
        "potential": plot_potential_profile(
            phi_diagnostics["radial_r"],
            phi_diagnostics["radial_phi_shifted"],
        ),
        "acceleration": plot_radial_acceleration_profile(
            phi_diagnostics["radial_r"],
            phi_diagnostics["radial_acceleration"],
        ),
        "density": plot_mass_density_profile(
            phi_diagnostics["radial_r"],
            phi_diagnostics["radial_density"],
        ),
    }
    density_figure = plot_density_profile(df_diagnostics)

    assert df_metrics["n_models"] == 1
    assert df_samples["reference_eta"].shape == (2, 6)
    assert phi_metrics["residual_std"] == 0.2
    assert set(figures) == {"potential", "acceleration", "density"}
    assert density_figure.axes[0].get_ylabel() == "tracer density"
    for figure in figures.values():
        plt.close(figure)
    plt.close(density_figure)


def test_missing_evaluation_artifacts_fail_explicitly(tmp_path):
    import pytest

    for loader in (
        load_df_metrics,
        load_df_diagnostics,
        load_df_samples,
        load_phi_metrics,
        load_phi_diagnostics,
    ):
        with pytest.raises(FileNotFoundError, match="Missing .* artifact"):
            loader(tmp_path)
