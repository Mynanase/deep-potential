from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.workflows.score_sources as score_sources
from dpjax.normalization import Normalizer
from dpjax.physics.cbe import residual_A
from experiments.validation.plummer import plummer_score_phys_batch
from experiments.workflows.score_sources import (
    FLOW_SCORE_SOURCE,
    PLUMMER_ANALYTIC_SCORE_SOURCE,
    resolve_score_source,
    score_std_batch,
)


def test_score_source_defaults_to_flow_and_validates_explicit_source():
    assert resolve_score_source({}) == FLOW_SCORE_SOURCE
    assert (
        resolve_score_source({"score": {"source": "plummer_analytic"}})
        == PLUMMER_ANALYTIC_SCORE_SOURCE
    )
    with pytest.raises(ValueError, match="score.source must be one of"):
        resolve_score_source({"score": {"source": "unknown"}})
    with pytest.raises(TypeError, match="score must be a mapping"):
        resolve_score_source({"score": "plummer_analytic"})


def test_plummer_analytic_score_respects_standardization_chain_rule():
    eta_phys = np.asarray(
        [
            [0.2, -0.1, 0.3, 0.1, 0.2, -0.1],
            [1.0, 0.2, -0.4, -0.2, 0.1, 0.05],
        ],
        dtype=np.float32,
    )
    normalizer = Normalizer(
        mean=np.asarray([0.1, -0.2, 0.05, 0.02, -0.03, 0.04], dtype=np.float32),
        std=np.asarray([2.0, 1.5, 0.8, 0.7, 1.2, 0.9], dtype=np.float32),
    )
    eta_std = jnp.asarray(normalizer.transform(eta_phys))

    analytic_score = jax.jit(
        lambda batch: score_std_batch(
            PLUMMER_ANALYTIC_SCORE_SOURCE,
            batch,
            normalizer,
        )
    )
    score_std = np.asarray(analytic_score(eta_std))
    score_phys = np.asarray(plummer_score_phys_batch(jnp.asarray(eta_phys)))

    np.testing.assert_allclose(
        score_std,
        score_phys * normalizer.std[None, :],
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    position = eta_phys[:, :3]
    radius_squared = np.sum(position**2, axis=1)
    grad_phi_phys = position * (1.0 + radius_squared[:, None]) ** -1.5
    grad_phi_std = grad_phi_phys * normalizer.std[None, :3]
    cbe_residual = np.asarray(
        residual_A(
            eta_std,
            jnp.asarray(score_std),
            jnp.asarray(grad_phi_std),
            normalizer,
        )
    )
    np.testing.assert_allclose(cbe_residual, 0.0, atol=2.0e-6)


def test_flow_score_source_delegates_to_flow_backend(monkeypatch):
    normalizer = Normalizer(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
    )
    eta_std = jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    captured = {}

    def fake_score_apply(model, params, batch, flow_cfg):
        captured.update(
            model=model,
            params=params,
            batch=np.asarray(batch),
            flow_cfg=flow_cfg,
        )
        return batch + 1.0

    monkeypatch.setattr(score_sources, "score_apply", fake_score_apply)
    result = score_std_batch(
        FLOW_SCORE_SOURCE,
        eta_std,
        normalizer,
        df_model="flow-model",
        df_params={"weight": 3.0},
        flow_cfg={"type": "dummy"},
    )

    np.testing.assert_allclose(result, np.asarray(eta_std) + 1.0)
    assert captured["model"] == "flow-model"
    assert captured["params"] == {"weight": 3.0}
    assert captured["flow_cfg"] == {"type": "dummy"}
