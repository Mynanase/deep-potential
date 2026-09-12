#!/usr/bin/env python
"""Plot acceleration and signed density along the three positive halo axes."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
from astropy.constants import G

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import potential


def plot_potential(input_fname, potential_dir, output_dir, n_points=128):
    # Always recover physical units from the same input that trained the model.
    with h5py.File(input_fname, "r") as f:
        length_scale = float(f.attrs["length_scale_kpc"])
        velocity_scale = float(f.attrs["velocity_scale_kms"])
        r_min = float(f.attrs["phi_r_min_kpc"])
        r_max = float(f.attrs["phi_r_max_kpc"])
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    model, _ = potential.PotentialModel.load(Path(potential_dir), load_history=False)
    r = np.geomspace(max(r_min, 1e-3), r_max, n_points)
    points = np.zeros((3, n_points, 3), dtype="f4")
    for i in range(3):
        points[i, :, i] = r / length_scale
    derivative_fn = eqx.filter_jit(jax.vmap(potential.calc_phi_derivatives, in_axes=(None, 0)))
    grad, laplacian = derivative_fn(model.phi_model, jnp.asarray(points.reshape(-1, 3)))
    acceleration = -np.asarray(grad).reshape(3, n_points, 3) * velocity_scale**2 / length_scale
    grav_const = G.to_value("kpc km2 / (s2 solMass)")
    density = np.asarray(laplacian).reshape(3, n_points) * velocity_scale**2 / (4 * np.pi * grav_const * length_scale**2)
    if not (np.isfinite(acceleration).all() and np.isfinite(density).all()):
        raise ValueError("The model produced non-finite accelerations or densities.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "potential_axes.npz", r_kpc=r,
             acceleration_kms2_per_kpc=acceleration, density_msun_per_kpc3=density)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, label in enumerate(("x axis", "y axis", "z axis")):
        axes[0].plot(r, acceleration[i, :, i], label=label)
        axes[1].plot(r, density[i], label=label)
    axes[0].set_ylabel(r"$a_r$ [$({\rm km/s})^2/{\rm kpc}$]")
    axes[1].set_ylabel(r"$\rho$ [$M_\odot/{\rm kpc}^3$]")
    # Preserve the sign: a negative Laplacian must remain visible.
    axes[1].set_yscale("symlog", linthresh=max(1.0, np.max(np.abs(density)) * 1e-3))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("r [kpc]")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.legend()
    fig.suptitle("Halo12: axis profiles (not spherical averages)")
    fig.savefig(output_dir / "potential_axes.png", dpi=150)
    plt.close(fig)
    print(f"Saved axis profiles to {output_dir}")


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--potential-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=128)
    args = parser.parse_args()
    plot_potential(args.input, args.potential_dir, args.output_dir, args.n_points)


if __name__ == "__main__":
    main()
