from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.flows.api import build_flow, init_flow, log_prob_apply, score_apply
from dpjax.models.potential import PotentialConfig, PotentialMLP, grad_phi_apply
from dpjax.normalization import Normalizer
from dpjax.physics.cbe import residual_A


def main() -> int:
    key = jax.random.key(0)

    # Fake standardized batch
    x = jax.random.normal(key, shape=(128, 6), dtype=jnp.float32)

    # --- FFJORD (default, recommended) ---
    flow_cfg = {"type": "ffjord", "dim": 6}
    flow = build_flow(flow_cfg)
    params_flow = init_flow(flow, key, flow_cfg)

    lp = log_prob_apply(flow, params_flow, x, flow_cfg)
    score = score_apply(flow, params_flow, x[:16], flow_cfg)

    phi = PotentialMLP(PotentialConfig())
    params_phi = phi.init(key, x[:, :3])["params"]
    grad_phi = grad_phi_apply(phi, params_phi, x[:16, :3])

    norm = Normalizer(mean=np.zeros(6, dtype=np.float32), std=np.ones(6, dtype=np.float32))
    r = residual_A(x[:16], score, grad_phi, norm)

    print("FFJORD log_prob shape:", lp.shape)
    print("FFJORD score shape:", score.shape)
    print("grad_phi shape:", grad_phi.shape)
    print("residual shape:", r.shape)
    print("finite?", bool(jnp.all(jnp.isfinite(lp))) and bool(jnp.all(jnp.isfinite(r))))

    # --- RealNVP (legacy, emits FutureWarning) ---
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flow_cfg_legacy = {"type": "realnvp", "dim": 6}
        flow_legacy = build_flow(flow_cfg_legacy)
        params_legacy = init_flow(flow_legacy, key, flow_cfg_legacy)
        lp_legacy = log_prob_apply(flow_legacy, params_legacy, x, flow_cfg_legacy)
        assert any(issubclass(wi.category, FutureWarning) for wi in w), \
            "Expected FutureWarning for realnvp"
    print("RealNVP log_prob shape:", lp_legacy.shape)

    # --- Missing type should raise ValueError ---
    try:
        build_flow({"dim": 6})
        raise AssertionError("Expected ValueError for missing flow.type")
    except ValueError as exc:
        assert "flow.type is required" in str(exc)
    print("Missing type correctly raises ValueError")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
