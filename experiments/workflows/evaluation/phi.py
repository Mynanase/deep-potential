"""Operational potential evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.models.potential import (
    grad_phi_apply,
    laplacian_phi_apply,
    phi_apply,
)
from dpjax.physics.cbe import residual_A
from dpjax.physics.units import (
    density_from_laplacian,
    summarize_density_sign,
)
from experiments.datasets.phase_space import (
    iter_batches,
    load_df_support_eta,
    require_physics_compatible_transform,
)
from experiments.paths import ensure_dir, resolve_path
from experiments.workflows.artifacts import load_df, load_phi
from experiments.workflows.evaluation.units import gravitational_constant_for_system
from experiments.workflows.score_sources import (
    resolve_score_source,
    score_std_batch,
)

# ---------------------------------------------------------------------------
# Operational evaluation workflow – called by run_eval
# ---------------------------------------------------------------------------

def run_eval_phi(
    data_path: str | Path,
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    n_eval: int = 32768,
    batch_size: int = 4096,
    seed: int = 0,
    r_min: float = 1.0e-3,
    r_max: float = 10.0,
    n_r: int = 256,
    r_ref: float = 1.0,
    system: str = "generic",
    compute_slice: bool = True,
    slice_grid: int = 128,
    slice_rmax: float | None = None,
    gravitational_constant: float | None = None,
) -> dict[str, Any]:
    """Evaluate trained Phi/DF on residual stats and radial/slice diagnostics.

    Returns
    -------
    dict
        ``{"metrics": dict, "diagnostics": dict, "out_dir": Path}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)

    df_model, df_params, normalizer, df_cfg, coord_transform = load_df(df_run_dir)
    require_physics_compatible_transform(
        coord_transform,
        operation="Phi/CBE evaluation",
    )
    flow_cfg = df_cfg.get("flow", {})
    phi_model, phi_params, phi_cfg = load_phi(phi_run_dir)
    score_source = resolve_score_source(phi_cfg)

    out_dir = ensure_dir(out_dir or phi_run_dir)
    system = str(system).lower()
    density_g = gravitational_constant_for_system(
        system,
        gravitational_constant,
    )

    df_data_cfg = df_cfg.get("data", {})
    support_eta, support_source, source_n = load_df_support_eta(
        data_path,
        df_run_dir,
        df_data_cfg,
        coordinate_transform=coord_transform,
    )
    print(
        "[eval_phi] DF support: "
        f"source={support_source}, kept={support_eta.shape[0]}/"
        f"{source_n}"
    )
    print(f"[eval_phi] score_source={score_source}")
    eta_std = normalizer.transform(support_eta)

    n_total = eta_std.shape[0]
    n_eval = int(min(n_eval, n_total))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_eval, replace=False)
    eta_eval = eta_std[idx]

    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)

    @jax.jit
    def residual_batch(eta_std_batch: jnp.ndarray) -> jnp.ndarray:
        score_std = score_std_batch(
            score_source,
            eta_std_batch,
            normalizer,
            df_model=df_model,
            df_params=df_params,
            flow_cfg=flow_cfg,
        )
        grad_phi_std = grad_phi_apply(phi_model, phi_params, eta_std_batch[:, :3])
        return residual_A(eta_std_batch, score_std, grad_phi_std, normalizer)

    rs: list[np.ndarray] = []
    for batch_np in iter_batches(eta_eval, batch_size=int(batch_size), rng=rng, shuffle=False, drop_remainder=False):
        r = residual_batch(jnp.asarray(batch_np)).astype(jnp.float32)
        rs.append(np.asarray(r))

    r_all = np.concatenate(rs, axis=0)

    # Keep per-point residuals with physical coordinates for offline plotting.
    eta_eval_phys = normalizer.inverse(eta_eval)

    stats = {
        "n_eval": int(r_all.shape[0]),
        "score_source": score_source,
        "residual_mean": float(np.mean(r_all)),
        "residual_std": float(np.std(r_all)),
        "residual_p99_abs": float(np.percentile(np.abs(r_all), 99.0)),
        "residual_p999_abs": float(np.percentile(np.abs(r_all), 99.9)),
        "residual_max_abs": float(np.max(np.abs(r_all))),
        "density_semantics": "total_gravitating_density",
        "density_gravitational_constant": density_g,
    }
    # Radial curves along x-axis
    r = np.geomspace(r_min, r_max, num=int(n_r)).astype(np.float32)
    x_phys = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1)
    x_std = (x_phys - mean_x[None, :]) / std_x[None, :]

    x_std_j = jnp.asarray(x_std)
    phi_learned = np.asarray(phi_apply(phi_model, phi_params, x_std_j)).astype(np.float32)

    grad_phi_std = np.asarray(grad_phi_apply(phi_model, phi_params, x_std_j)).astype(np.float32)
    grad_phi_phys = grad_phi_std / std_x[None, :]

    # Along x-axis, radial acceleration equals -dPhi/dx
    ar_learned = -grad_phi_phys[:, 0]

    r_ref = float(r_ref)
    i_ref = int(np.argmin(np.abs(r - r_ref)))
    phi_learned_shift = phi_learned - phi_learned[i_ref]

    # Total gravitating density from the Poisson equation.
    std_x_j = jnp.asarray(std_x)
    lap_phys = np.asarray(
        laplacian_phi_apply(phi_model, phi_params, x_std_j, std_x=std_x_j),
        dtype=np.float32,
    )
    rho_learned = density_from_laplacian(
        lap_phys,
        gravitational_constant=density_g,
    )

    slice_data: dict[str, np.ndarray] | None = None
    if compute_slice:
        r_xy = np.sqrt(eta_eval_phys[:, 0] ** 2 + eta_eval_phys[:, 1] ** 2)
        rmax_slice = float(slice_rmax) if slice_rmax is not None else float(max(np.percentile(r_xy, 99.0), 1.0e-6))
        grid = int(slice_grid)
        xs = np.linspace(-rmax_slice, rmax_slice, grid, dtype=np.float32)
        ys = np.linspace(-rmax_slice, rmax_slice, grid, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        xyz = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size, dtype=np.float32)], axis=-1)
        xyz_std = (xyz - mean_x[None, :]) / std_x[None, :]
        phi_slices: list[np.ndarray] = []
        rho_slices: list[np.ndarray] = []
        acc_slices: list[np.ndarray] = []
        for i in range(0, xyz_std.shape[0], int(batch_size)):
            sl = slice(i, min(i + int(batch_size), xyz_std.shape[0]))
            x_batch = jnp.asarray(xyz_std[sl])
            phi_b = np.asarray(phi_apply(phi_model, phi_params, x_batch), dtype=np.float32)
            grad_b = np.asarray(grad_phi_apply(phi_model, phi_params, x_batch), dtype=np.float32)
            grad_phys_b = grad_b / std_x[None, :]
            lap_b = np.asarray(laplacian_phi_apply(phi_model, phi_params, x_batch, std_x=std_x_j), dtype=np.float32)
            phi_slices.append(phi_b)
            rho_slices.append(
                density_from_laplacian(
                    lap_b,
                    gravitational_constant=density_g,
                )
            )
            acc_slices.append(np.linalg.norm(-grad_phys_b, axis=-1))
        phi_img = np.concatenate(phi_slices).reshape(X.shape)
        rho_img = np.concatenate(rho_slices).reshape(X.shape)
        acc_img = np.concatenate(acc_slices).reshape(X.shape)
        slice_data = {"x": xs, "y": ys, "phi": phi_img, "rho": rho_img, "acc_mag": acc_img}
        stats["slice_density"] = summarize_density_sign(rho_img)

    diagnostics = {
        "residual_x": eta_eval_phys[:, 0],
        "residual_y": eta_eval_phys[:, 1],
        "residual_z": eta_eval_phys[:, 2],
        "residual": r_all,
        "radial_r": r,
        "radial_phi": phi_learned,
        "radial_phi_shifted": phi_learned_shift,
        "radial_acceleration": ar_learned,
        "radial_density": rho_learned,
    }
    if slice_data is not None:
        diagnostics.update(
            {
                "slice_x": slice_data["x"],
                "slice_y": slice_data["y"],
                "slice_phi": slice_data["phi"],
                "slice_density": slice_data["rho"],
                "slice_acceleration_magnitude": slice_data["acc_mag"],
            }
        )

    (out_dir / "phi_metrics.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(out_dir / "phi_diagnostics.npz", **diagnostics)
    print(json.dumps(stats, indent=2))
    print(f"Wrote Phi evaluation artifacts to {out_dir}")

    return {
        "metrics": stats,
        "diagnostics": diagnostics,
        "out_dir": out_dir,
    }
