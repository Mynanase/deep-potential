"""On-server numerical self-check for the Plummer score oracle.

Checks three invariants, all CPU-only and independent of any trained Phi:

1. Standardisation chain rule:
   score_std_batch("plummer_analytic") == plummer_score_phys_batch * std
2. Oracle CBE identity: with grad_phi = -nabla_x log f, the collisionless
   residual (form A) is ~0 on the DF support.
3. Contrast with the flow-source score: the same residual is (expected)
   non-zero, since a trained flow fits log f directly and only approximately
   satisfies the potential-determined CBE.

Usage: python analysis/check_oracle_score.py
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from dpjax.normalization import Normalizer
from dpjax.physics.cbe import residual_A
from experiments.datasets.phase_space import load_eta_h5, load_run_preprocessing, resolve_run_support_indices
from experiments.validation.plummer import plummer_score_phys_batch
from experiments.workflows.score_sources import (
    FLOW_SCORE_SOURCE,
    PLUMMER_ANALYTIC_SCORE_SOURCE,
    score_std_batch,
)

DATA_PATH = "data/plummer_n524288_train.h5"
DF_RUN = "runs/plummer_rcut/full-baseline/df"
N_CHECK = 8192
SEED = 2024


def main() -> None:
    normalizer, coord_transform = load_run_preprocessing(DF_RUN)
    assert coord_transform is None or coord_transform.type == "none"
    eta = load_eta_h5(DATA_PATH, dataset="eta")
    indices, source = resolve_run_support_indices(DF_RUN, eta, data_config={})
    support = eta[indices]
    print(f"[check] support source={source} kept={support.shape[0]}/{eta.shape[0]}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(support.shape[0], size=N_CHECK, replace=False)
    eta_phys = support[idx].astype(np.float32)
    eta_std = jnp.asarray(normalizer.transform(eta_phys))

    # --- invariant 1: chain rule through the real score_sources entry ----
    oracle_std = np.asarray(
        score_std_batch(PLUMMER_ANALYTIC_SCORE_SOURCE, eta_std, normalizer)
    )
    analytic_phys = np.asarray(plummer_score_phys_batch(jnp.asarray(eta_phys)))
    chain_err = np.abs(oracle_std - analytic_phys * normalizer.std[None, :])
    print(f"[check 1] chain-rule max abs err = {chain_err.max():.3e}")

    # --- physical analytic gradient of the Plummer potential --------------
    r2 = np.sum(eta_phys[:, :3] ** 2, axis=1)
    grad_phi_phys = eta_phys[:, :3] * (1.0 + r2[:, None]) ** -1.5
    grad_phi_std = jnp.asarray(grad_phi_phys * normalizer.std[None, :3])

    def oracle_residual() -> np.ndarray:
        return np.asarray(
            residual_A(eta_std, jnp.asarray(oracle_std), grad_phi_std, normalizer)
        )

    r_oracle = oracle_residual()
    print("[check 2] oracle CBE residual "
          f"mean={np.mean(r_oracle):+.3e} std={np.std(r_oracle):.3e} "
          f"max|.|={np.max(np.abs(r_oracle)):.3e}")

    # --- contrast: flow score on the same points --------------------------
    from experiments.workflows.artifacts import load_df

    flow_model, flow_params, _, df_cfg, _ = load_df(DF_RUN)
    flow_cfg = df_cfg.get("flow", {})
    flow_std = np.asarray(
        score_std_batch(
            FLOW_SCORE_SOURCE,
            eta_std,
            normalizer,
            df_model=flow_model,
            df_params=flow_params,
            flow_cfg=flow_cfg,
        )
    )
    r_flow = np.asarray(
        residual_A(eta_std, jnp.asarray(flow_std), grad_phi_std, normalizer)
    )
    print("[check 3] flow CBE residual "
          f"mean={np.mean(r_flow):+.3e} std={np.std(r_flow):.3e} "
          f"max|.|={np.max(np.abs(r_flow)):.3e}")
    print("[check ] oracle residual sigma / flow residual sigma = "
          f"{np.std(r_oracle) / np.std(r_flow):.2e}")


if __name__ == "__main__":
    main()