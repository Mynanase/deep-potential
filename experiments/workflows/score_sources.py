"""Resolve phase-space score providers for potential training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from dpjax.flows.api import score_apply
from dpjax.normalization import Normalizer


FLOW_SCORE_SOURCE = "flow"
PLUMMER_ANALYTIC_SCORE_SOURCE = "plummer_analytic"
SUPPORTED_SCORE_SOURCES = frozenset(
    {FLOW_SCORE_SOURCE, PLUMMER_ANALYTIC_SCORE_SOURCE}
)


def resolve_score_source(config: Mapping[str, Any]) -> str:
    """Return the configured score source, defaulting to the trained flow."""
    raw = config.get("score", {})
    if not isinstance(raw, Mapping):
        raise TypeError("score must be a mapping.")
    source = str(raw.get("source", FLOW_SCORE_SOURCE)).strip().lower()
    if source not in SUPPORTED_SCORE_SOURCES:
        choices = ", ".join(sorted(SUPPORTED_SCORE_SOURCES))
        raise ValueError(
            f"score.source must be one of: {choices}; got {source!r}."
        )
    return source


def score_std_batch(
    source: str,
    eta_std_batch: jnp.ndarray,
    normalizer: Normalizer,
    *,
    df_model: Any | None = None,
    df_params: Any | None = None,
    flow_cfg: Mapping[str, Any] | None = None,
) -> jnp.ndarray:
    """Evaluate one configured score field in standardized coordinates."""
    if source == FLOW_SCORE_SOURCE:
        if df_model is None:
            raise ValueError("df_model is required for score.source='flow'.")
        return score_apply(
            df_model,
            df_params,
            eta_std_batch,
            dict(flow_cfg or {}),
        )
    if source == PLUMMER_ANALYTIC_SCORE_SOURCE:
        # Analytic truth remains in the validation layer and is selected only
        # by an explicit oracle configuration.
        from experiments.validation.plummer import plummer_score_std_batch

        return plummer_score_std_batch(
            eta_std_batch,
            jnp.asarray(normalizer.mean),
            jnp.asarray(normalizer.std),
        )
    raise ValueError(f"Unsupported score source: {source!r}.")
