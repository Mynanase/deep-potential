"""Compare trained DF + Phi (runs/plummer_rcut/full) against analytic Plummer.

Part A: distribution function diagnostics  — real (analytic-sampled) data vs FFJORD samples
Part B: potential diagnostics            — trained MLP Phi vs analytic Plummer solution
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/localdisk/kosmos/my-deep-potential")

from dpjax.flows.api import sample_apply
from dpjax.models.potential import grad_phi_apply, laplacian_phi_apply, phi_apply
from dpjax.physics.units import density_from_laplacian
from experiments.datasets.phase_space import load_eta_h5
from experiments.plotting.flow_projections import plot_2d_marginals_grid
from experiments.workflows.artifacts import load_df, load_phi
from experiments.workflows.evaluation.units import gravitational_constant_for_system

ROOT = Path("/localdisk/kosmos/my-deep-potential")
DATA = ROOT / "data/plummer_n524288_train.h5"
DF_RUN = ROOT / "runs/plummer_rcut/full/trial_00/df"
PHI_RUN = ROOT / "runs/plummer_rcut/full/trial_00/phi"
OUT = ROOT / "runs/plummer_rcut/full/eval/plots/compare"
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
N_SAMPLE = 131_072
N_R = 256
R_MIN, R_MAX, R_REF = 1.0e-3, 10.0, 1.0
GRID = 128
BATCH = 4096
G = gravitational_constant_for_system("plummer", None)

# ---------------------------------------------------------------------------
# Analytic Plummer (a = M = G = 1)
# ---------------------------------------------------------------------------
def phi_analytic(r: np.ndarray) -> np.ndarray:
    return -1.0 / np.sqrt(1.0 + r**2)

def rho_analytic(r: np.ndarray) -> np.ndarray:
    return 3.0 / (4.0 * np.pi) * (1.0 + r**2) ** -2.5

def ar_analytic(r: np.ndarray) -> np.ndarray:
    return -r / (1.0 + r**2) ** 1.5

# ---------------------------------------------------------------------------
# Part A: DF diagnostics (real data vs model samples)
# ---------------------------------------------------------------------------
print("[A] loading real data ...")
eta_real = load_eta_h5(DATA)
print(f"[A] real eta: {eta_real.shape}")

print("[A] loading DF model ...")
df_model, df_params, normalizer, df_cfg, _ = load_df(DF_RUN)
rng = jax.random.PRNGKey(RNG_SEED)
eta_std_sample = np.asarray(sample_apply(df_model, df_params, rng, N_SAMPLE, flow_cfg=df_cfg.get("flow")))
eta_sample = np.asarray(normalizer.inverse(eta_std_sample))
print(f"[A] DF samples: {eta_sample.shape}")

x_r, y_r, z_r, vx_r, vy_r, vz_r = eta_real.T
x_s, y_s, z_s, vx_s, vy_s, vz_s = eta_sample.T

# --- radial density profile (real vs model vs analytic rho) ---
r_real = np.linalg.norm(eta_real[:, :3], axis=1)
r_samp = np.linalg.norm(eta_sample[:, :3], axis=1)
r_edges = np.geomspace(max(r_real.min(), 1e-3), r_real.max() * 1.01, 40)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
ax.hist(r_real, bins=r_edges, density=True, histtype="step", color="black", lw=1.6, label="real (analytic sample)")
ax.hist(r_samp, bins=r_edges, density=True, histtype="step", color="C1", lw=1.4, label="DF model sample")
r_c = np.sqrt(r_edges[:-1] * r_edges[1:])
rho_norm = rho_analytic(r_c)
# scale analytic rho by 4\pi r^2 volume factor -> same display as 3D radial density from hist
ax.plot(r_c, 4.0 * np.pi * r_c**2 * rho_norm, color="C2", ls="--", lw=1.2,
        label=r"analytic $4\pi r^2\rho$")
ax.set_xscale("log")
ax.set_xlabel("r")
ax.set_ylabel("density")
ax.set_title("DF: radial density profile (full data)")
ax.legend()
ax.grid(True, alpha=0.2, which="both")
fig.tight_layout()
fig.savefig(OUT / "df_radial_density.png", bbox_inches="tight")
plt.close(fig)

# --- velocity marginals ---
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150)
for ax, (vr, vs, tag) in zip(axes, [(vx_r, vx_s, "vx"), (vy_r, vy_s, "vy"), (vz_r, vz_s, "vz")]):
    lo, hi = np.percentile(vr, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    ax.hist(vr, bins=bins, density=True, histtype="step", color="black", lw=1.6, label="real")
    ax.hist(vs, bins=bins, density=True, histtype="step", color="C1", lw=1.4, label="model")
    ax.set_xlabel(tag)
    ax.set_title(f"velocity marginal {tag}")
    ax.legend()
    ax.grid(True, alpha=0.2)
fig.suptitle("DF: velocity marginals (full)")
fig.tight_layout()
fig.savefig(OUT / "df_velocity_marginals.png", bbox_inches="tight")
plt.close(fig)

# --- 2D marginals grid (reuses the project's plot_2d_marginals_grid) ---
coords = {
    "x": x_r, "y": y_r, "z": z_r,
    "vx": vx_r, "vy": vy_r, "vz": vz_r,
}
coords_s = {
    "x": x_s, "y": y_s, "z": z_s,
    "vx": vx_s, "vy": vy_s, "vz": vz_s,
}
plot_2d_marginals_grid(
    coords, coords_s,
    dims=[("x", "y"), ("x", "z"), ("vx", "vy")],
    fig_dir=str(OUT),
    fig_fmt=("png",),
)
print(f"[A] wrote DF figures to {OUT}")

# ---------------------------------------------------------------------------
# Part B: Phi diagnostics (trained MLP vs analytic)
# ---------------------------------------------------------------------------
print("[B] loading Phi model ...")
phi_model, phi_params, _ = load_phi(PHI_RUN)
mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
std_x = np.asarray(normalizer.std[:3], dtype=np.float32)

r = np.geomspace(R_MIN, R_MAX, N_R).astype(np.float32)
x_phys = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=-1)
x_std = (x_phys - mean_x[None, :]) / std_x[None, :]
x_std_j = jnp.asarray(x_std)

phi_learned = np.asarray(phi_apply(phi_model, phi_params, x_std_j), dtype=np.float32)
grad_std = np.asarray(grad_phi_apply(phi_model, phi_params, x_std_j), dtype=np.float32)
grad_phys = grad_std / std_x[None, :]
ar_learned = -grad_phys[:, 0]
lap_phys = np.asarray(
    laplacian_phi_apply(phi_model, phi_params, x_std_j, std_x=jnp.asarray(std_x)),
    dtype=np.float32,
)
rho_learned = density_from_laplacian(lap_phys, gravitational_constant=G)

i_ref = int(np.argmin(np.abs(r - R_REF)))
phi_shift_learned = phi_learned - phi_learned[i_ref]
phi_shift_truth = phi_analytic(r) - phi_analytic(r)[i_ref]
rho_truth = rho_analytic(r)
ar_truth = ar_analytic(r)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
axes[0].plot(r, phi_shift_learned, color="C0", lw=1.4, label="trained MLP")
axes[0].plot(r, phi_shift_truth, color="black", ls="--", lw=1.2, label=r"analytic $-\frac{1}{\sqrt{1+r^2}}$")
axes[0].set_xscale("log")
axes[0].set_ylabel(r"$\Phi-\Phi(r_\mathrm{ref})$")
axes[0].set_title("Potential")
axes[1].plot(r, rho_learned, color="C0", lw=1.4, label="trained MLP (Poisson)")
axes[1].plot(r, rho_truth, color="black", ls="--", lw=1.2, label=r"analytic $\rho$")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_ylabel(r"$\rho$")
axes[1].set_title("Total density")
axes[2].plot(r, ar_learned, color="C0", lw=1.4, label="trained MLP")
axes[2].plot(r, ar_truth, color="black", ls="--", lw=1.2, label="analytic")
axes[2].set_xscale("log")
axes[2].set_ylabel(r"$a_r$")
axes[2].set_title("Radial acceleration")
for ax in axes:
    ax.set_xlabel("r")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=8)
fig.suptitle("Phi: trained MLP vs analytic Plummer (radial, full)")
fig.tight_layout()
fig.savefig(OUT / "phi_radial_profiles.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
axes[0].plot(r, phi_shift_learned - phi_shift_truth, color="C2", lw=1.2)
axes[0].axhline(0, color="0.3", lw=0.8)
axes[0].set_yscale("symlog", linthresh=1e-6)
axes[0].set_title(r"$\Delta\Phi_\mathrm{shift}$")
axes[1].plot(r, rho_learned - rho_truth, color="C2", lw=1.2)
axes[1].axhline(0, color="0.3", lw=0.8)
axes[1].set_yscale("symlog", linthresh=1e-6)
axes[1].set_title(r"$\Delta\rho$")
axes[2].plot(r, ar_learned - ar_truth, color="C2", lw=1.2)
axes[2].axhline(0, color="0.3", lw=0.8)
axes[2].set_yscale("symlog", linthresh=1e-6)
axes[2].set_title(r"$\Delta a_r$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("r")
    ax.grid(True, alpha=0.2, which="both")
fig.suptitle("Phi: residual vs analytic (full)")
fig.tight_layout()
fig.savefig(OUT / "phi_radial_residuals.png", bbox_inches="tight")
plt.close(fig)

# --- xy slices ---
rmax_slice = 6.0
xs = np.linspace(-rmax_slice, rmax_slice, GRID, dtype=np.float32)
ys = np.linspace(-rmax_slice, rmax_slice, GRID, dtype=np.float32)
X, Y = np.meshgrid(xs, ys, indexing="xy")
xyz = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size, dtype=np.float32)], axis=-1)
xyz_std = (xyz - mean_x[None, :]) / std_x[None, :]

phi_img = np.empty(X.size, dtype=np.float32)
rho_img = np.empty(X.size, dtype=np.float32)
acc_img = np.empty(X.size, dtype=np.float32)
for i in range(0, xyz_std.shape[0], BATCH):
    sl = slice(i, min(i + BATCH, xyz_std.shape[0]))
    xb = jnp.asarray(xyz_std[sl])
    phi_img[sl] = np.asarray(phi_apply(phi_model, phi_params, xb), dtype=np.float32)
    gb = np.asarray(grad_phi_apply(phi_model, phi_params, xb), dtype=np.float32)
    gb_phys = gb / std_x[None, :]
    acc_img[sl] = np.linalg.norm(-gb_phys, axis=-1)
    lb = np.asarray(laplacian_phi_apply(phi_model, phi_params, xb, std_x=jnp.asarray(std_x)), dtype=np.float32)
    rho_img[sl] = density_from_laplacian(lb, gravitational_constant=G)
phi_img = phi_img.reshape(X.shape)
rho_img = rho_img.reshape(X.shape)
acc_img = acc_img.reshape(X.shape)

R2 = X**2 + Y**2
phi_truth_img = -1.0 / np.sqrt(1.0 + R2)
rho_truth_img = 3.0 / (4.0 * np.pi) * (1.0 + R2) ** -2.5
acc_truth_img = np.sqrt(R2) / (1.0 + R2) ** 1.5
phi_truth_img = phi_truth_img - (phi_truth_img[GRID // 2, GRID // 2])
phi_img_shift = phi_img - phi_img[GRID // 2, GRID // 2]

panels = [
    ("potential", phi_img_shift, phi_truth_img, r"$\Phi_\mathrm{shift}$"),
    ("density", rho_img, rho_truth_img, r"$\rho$"),
    ("acceleration", acc_img, acc_truth_img, r"$|a|$"),
]
for tag, m, t, label in panels:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    norm_log = tag != "potential"
    for ax, arr, title in ((axes[0], m, "model"), (axes[1], t, "analytic")):
        if norm_log:
            vmin = np.percentile(arr[arr > 0], 1) if np.any(arr > 0) else 1e-6
            im = ax.pcolormesh(xs, ys, np.maximum(arr, vmin), cmap="magma",
                               shading="auto", norm=matplotlib.colors.LogNorm(vmin=vmin))
        else:
            im = ax.pcolormesh(xs, ys, arr, cmap="viridis", shading="auto")
            fig.colorbar(im, ax=ax, label=label)
        ax.set_title(title)
        ax.set_aspect("equal")
    d = m - t
    d_show = d
    if norm_log:
        vmax = np.nanpercentile(np.abs(d), 99.0)
        im = axes[2].pcolormesh(xs, ys, d_show, cmap="coolwarm", shading="auto",
                                vmin=-vmax, vmax=vmax)
    else:
        vmax = max(np.nanpercentile(np.abs(d), 99.0), 1e-9)
        im = axes[2].pcolormesh(xs, ys, d_show, cmap="coolwarm", shading="auto",
                                vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=axes[2], label=f"model - analytic ({label})")
    axes[2].set_title("difference")
    axes[2].set_aspect("equal")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"Phi slice: {tag} (full) — model vs analytic Plummer")
    fig.tight_layout()
    fig.savefig(OUT / f"phi_slice_{tag}.png", bbox_inches="tight")
    plt.close(fig)

print("=" * 60)
print(f"All comparison figures written to {OUT}")
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)