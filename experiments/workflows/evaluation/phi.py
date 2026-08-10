"""Operational potential evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.flows.api import score_apply
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
from experiments.plotting.diagnostics import (
    plot_laplacian_density_diagnostics,
    plot_potential_density_overview,
)
from experiments.validation.plummer import plummer_ar, plummer_phi
from experiments.validation.units import gravitational_constant_for_system
from experiments.workflows.artifacts import load_df, load_phi

# ---------------------------------------------------------------------------
# Operational evaluation workflow – called by run_eval
# ---------------------------------------------------------------------------

def run_eval_phi(
    data_path: str | Path,
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    n_eval: int = 32768,
    batch_size: int = 4096,
    seed: int = 0,
    r_min: float = 1.0e-3,
    r_max: float = 10.0,
    n_r: int = 256,
    r_ref: float = 1.0,
    system: str = "generic",
    plot_overview: bool = True,
    slice_grid: int = 128,
    slice_rmax: float | None = None,
    fig_fmt: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 180,
    gravitational_constant: float | None = None,
) -> dict[str, Any]:
    """Evaluate trained Phi/DF on residual stats and radial/slice diagnostics.

    Returns
    -------
    dict
        ``{"stats": dict, "radial": dict, "out_dir": Path, "plots_dir": Path}``
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
    phi_model, phi_params, _ = load_phi(phi_run_dir)

    out_dir = ensure_dir(out_dir or phi_run_dir)
    plots_dir = ensure_dir(plots_dir or (out_dir / "plots"))
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
        score_std = score_apply(df_model, df_params, eta_std_batch, flow_cfg)
        grad_phi_std = grad_phi_apply(phi_model, phi_params, eta_std_batch[:, :3])
        return residual_A(eta_std_batch, score_std, grad_phi_std, normalizer)

    rs: list[np.ndarray] = []
    for batch_np in iter_batches(eta_eval, batch_size=int(batch_size), rng=rng, shuffle=False, drop_remainder=False):
        r = residual_batch(jnp.asarray(batch_np)).astype(jnp.float32)
        rs.append(np.asarray(r))

    r_all = np.concatenate(rs, axis=0)

    # Save per-point residuals with physical coordinates for spatial map
    eta_eval_phys = normalizer.inverse(eta_eval)
    np.savez(
        out_dir / "residual_spatial.npz",
        x=eta_eval_phys[:, 0],
        y=eta_eval_phys[:, 1],
        z=eta_eval_phys[:, 2],
        residual=r_all,
    )
    print(f"Saved residual_spatial.npz ({r_all.shape[0]} points)")

    stats = {
        "n_eval": int(r_all.shape[0]),
        "residual_mean": float(np.mean(r_all)),
        "residual_std": float(np.std(r_all)),
        "residual_p99_abs": float(np.percentile(np.abs(r_all), 99.0)),
        "residual_p999_abs": float(np.percentile(np.abs(r_all), 99.9)),
        "residual_max_abs": float(np.max(np.abs(r_all))),
        "density_semantics": "total_gravitating_density",
        "density_gravitational_constant": density_g,
    }
    (out_dir / "eval_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

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

    np.savez(
        out_dir / "radial_curves.npz",
        r=r,
        phi_learned=phi_learned,
        phi_learned_shift=phi_learned_shift,
        ar_learned=ar_learned,
        rho_learned=rho_learned,
    )

    phi_true = None
    ar_true = None
    rho_analytic = None
    if system == "plummer":
        phi_true = plummer_phi(r)
        ar_true = plummer_ar(r)
        phi_true_ref = float(plummer_phi(np.array([r_ref], dtype=np.float32))[0])
        phi_learned_shift = phi_learned - phi_learned[i_ref] + phi_true_ref
        rho_analytic = (3.0 / (4.0 * np.pi)) * (1.0 + r ** 2) ** (-2.5)
        np.savez(
            out_dir / "radial_curves_plummer.npz",
            r=r,
            phi_learned=phi_learned,
            phi_learned_shift=phi_learned_shift,
            phi_true=phi_true,
            ar_learned=ar_learned,
            ar_true=ar_true,
            rho_learned=rho_learned,
            rho_analytic=rho_analytic,
        )

    slice_data: dict[str, np.ndarray] | None = None
    if plot_overview:
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
        np.savez(plots_dir / "phi_slice_xy.npz", x=xs, y=ys, phi=phi_img, rho=rho_img, acc_mag=acc_img)
        slice_data = {"x": xs, "y": ys, "phi": phi_img, "rho": rho_img, "acc_mag": acc_img}
        density_summary = summarize_density_sign(rho_img)
        (plots_dir / "rho_slice_xy_summary.json").write_text(
            json.dumps(density_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        plot_laplacian_density_diagnostics(
            xs,
            ys,
            rho_img,
            density_label=(
                r"$\rho_{\rm total}$"
                if system == "halo"
                else r"$\rho$"
            ),
            fig_dir=plots_dir,
            fig_fmt=fig_fmt,
            dpi=int(dpi),
            filename="rho_laplacian_diagnostics",
        )
        plot_potential_density_overview(
            r,
            phi_learned_shift,
            rho_learned,
            xs,
            ys,
            phi_img,
            rho_img,
            ar_learned=ar_learned,
            phi_true=phi_true,
            rho_true=rho_analytic,
            ar_true=ar_true,
            data_xy=eta_eval_phys[:, :2] if system != "plummer" else None,
            title="Plummer Potential / Density Overview" if system == "plummer" else "Halo Potential / Density Overview",
            density_label=(
                r"$\rho_{\rm total}=\nabla^2\Phi/(4\pi G)$"
                if system == "halo"
                else r"$\rho=\nabla^2\Phi/(4\pi G)$"
            ),
            fig_dir=plots_dir,
            fig_fmt=fig_fmt,
            dpi=int(dpi),
            filename="potential_density_overview",
        )
        print(f"Wrote potential density overview to {plots_dir}")

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        if system == "plummer" and phi_true is not None and ar_true is not None:
            plt.figure()
            plt.plot(r, phi_true, label="Plummer analytic")
            plt.plot(r, phi_learned_shift, label="Learned (shifted)")
            plt.xscale("log")
            plt.xlabel("r")
            plt.ylabel("Phi(r)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "phi_r_plummer.png", dpi=150)
            plt.close()

            plt.figure()
            plt.plot(r, ar_true, label="Plummer analytic")
            plt.plot(r, ar_learned, label="Learned")
            plt.xscale("log")
            plt.xlabel("r")
            plt.ylabel("a_r(r)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "ar_r_plummer.png", dpi=150)
            plt.close()

        print(f"Wrote plots to {plots_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"Plot skipped: {e}")

    return {
        "stats": stats,
        "radial": {
            "r": r, "phi_learned": phi_learned, "phi_learned_shift": phi_learned_shift,
            "phi_true": phi_true, "ar_learned": ar_learned, "ar_true": ar_true,
        },
        "slice": slice_data,
        "out_dir": out_dir,
        "plots_dir": plots_dir,
    }
