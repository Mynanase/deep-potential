#!/usr/bin/env python
"""Convert an exported Auriga stellar sample to the input used by fit_all.py.

Input coordinates must already be centred, in kpc and km/s. This is not a
Gadget snapshot reader: cosmological conversion and alignment belong upstream
of this script. Supports the Halo12 PartType4 export and the old eta files.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import h5py
import numpy as np


def load_snapshot(fname, group=None):
    with h5py.File(fname, "r") as f:
        g = f[group] if group else (f if "eta" in f else f["PartType4"])
        if "length_scale_kpc" in f.attrs:
            raise ValueError("Input is already scaled for training; use the physical source file.")
        attrs = dict(f.attrs)
        if "eta" in g:
            eta = g["eta"][:].astype("f4")
            units = (attrs.get("length_unit"), attrs.get("velocity_unit"))
            if units != ("kpc", "km/s"):
                raise ValueError(f"Expected eta units kpc, km/s; got {units}.")
        else:
            columns = ("x", "y", "z", "vx", "vy", "vz")
            for name, unit in zip(columns, ("kpc",) * 3 + ("km/s",) * 3):
                if name not in g or g[name].attrs.get("units") != unit:
                    raise ValueError(f"Expected exported {name} with units={unit}.")
            eta = np.column_stack([g[name][:] for name in columns]).astype("f4")
        if eta.ndim != 2 or eta.shape[1] != 6 or not np.isfinite(eta).all():
            raise ValueError("Expected finite eta with shape (N, 6).")
        fields = {}
        for name, aliases in {
            "mass": ("mass", "Masses"),
            "particle_id": ("particle_id", "ParticleIDs"),
            "source_index": ("source_index",),
        }.items():
            for alias in aliases:
                if alias in g:
                    fields[name] = g[alias][:]
                    if fields[name].shape != (len(eta),):
                        raise ValueError(f"{alias} must have one entry per particle.")
                    break
        fields.setdefault("source_index", np.arange(len(eta), dtype=np.int64))
        if "Header" in f:
            attrs.update({f"header_{k}": v for k, v in f["Header"].attrs.items()})
    return eta, fields, attrs


def prepare_data(input_fname, output_fname, group=None, weighting="mass",
                 length_scale=10.0, velocity_scale=100.0, train_r_max=75.0,
                 phi_r_min=1.0, phi_r_max=70.0, max_particles=None, seed=0):
    """Shuffle once before the upstream contiguous train/validation split.

    The DF uses all selected particles, including the boundary padding.
    eta.attrs defines only the interior in which flow samples train Phi.
    """
    if not (length_scale > 0 and velocity_scale > 0
            and 0 <= phi_r_min < phi_r_max < train_r_max
            and np.isfinite([length_scale, velocity_scale, train_r_max]).all()):
        raise ValueError("Use positive scales and 0 <= phi_r_min < phi_r_max < train_r_max.")
    if max_particles is not None and max_particles < 8:
        raise ValueError("max_particles must be at least 8.")
    if weighting not in ("mass", "number"):
        raise ValueError("weighting must be mass or number.")
    eta, fields, source_attrs = load_snapshot(input_fname, group)
    r = np.linalg.norm(eta[:, :3], axis=1)
    if r.max() < phi_r_max:
        raise ValueError("The source does not reach phi_r_max; reduce the inference radius.")
    indices = np.flatnonzero(r < train_r_max)
    indices = np.random.default_rng(seed).permutation(indices)
    if max_particles is not None:
        indices = indices[:max_particles]
    if len(indices) < 8:
        raise ValueError("Too few particles remain after selection.")
    eta = eta[indices]
    fields = {k: v[indices] for k, v in fields.items()}
    if weighting == "mass":
        if "mass" not in fields:
            raise ValueError("Mass weighting requires mass or Masses in the source.")
        mass = fields["mass"].astype("f8")
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("Particle masses must be finite and positive.")
        weights = mass / mass.mean()
    else:
        weights = np.ones(len(eta))
    eta[:, :3] /= length_scale
    eta[:, 3:] /= velocity_scale

    output_fname = Path(output_fname)
    output_fname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_fname, "x") as f:
        # Keep the old sample/frame description, not its training machinery.
        f.attrs.update(source_attrs)
        f.attrs.update({
            "source_file": str(Path(input_fname).resolve()),
            "length_scale_kpc": length_scale,
            "velocity_scale_kms": velocity_scale,
            "length_unit": "dimensionless", "velocity_unit": "dimensionless",
            "weighting": weighting, "shuffle_seed": seed,
            "train_r_max_kpc": train_r_max,
            "phi_r_min_kpc": phi_r_min, "phi_r_max_kpc": phi_r_max,
            "n_source": len(r), "n_selected": len(eta),
        })
        # No simulation potential or acceleration is fed into training.
        # Avoid carrying misleading physical-unit labels from old root attrs.
        for key in ("potential_unit", "acceleration_unit", "schema", "tracer_weight_definition"):
            if key in f.attrs:
                del f.attrs[key]
        ds = f.create_dataset("eta", data=eta, compression="lzf")
        ds.attrs.update({"volume_type": "sphere", "r_in": phi_r_min / length_scale,
                         "r_out": phi_r_max / length_scale})
        f.create_dataset("weights", data=weights.astype("f4"), compression="lzf")
        for name, values in fields.items():
            ds = f.create_dataset(name, data=values, compression="lzf")
            if name == "mass":
                ds.attrs["units"] = "Msun"
    print(f"Saved {len(eta):,} / {len(r):,} particles to {output_fname}")
    print(f"DF: r < {train_r_max:g} kpc; Phi: {phi_r_min:g} <= r <= {phi_r_max:g} kpc")
    print(f"Units: L={length_scale:g} kpc, V={velocity_scale:g} km/s; weighting={weighting}")


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group", default=None)
    parser.add_argument("--weighting", choices=("mass", "number"), default="mass")
    parser.add_argument("--length-scale", type=float, default=10.0, help="Length unit in kpc.")
    parser.add_argument("--velocity-scale", type=float, default=100.0, help="Velocity unit in km/s.")
    parser.add_argument("--train-r-max", type=float, default=75.0, help="DF outer radius, kpc.")
    parser.add_argument("--phi-r-min", type=float, default=1.0, help="Phi inner radius, kpc.")
    parser.add_argument("--phi-r-max", type=float, default=70.0, help="Phi outer radius, kpc.")
    parser.add_argument("--max-particles", type=int, default=None, help="Uniform subsample for a smoke run.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.group, args.weighting,
                 args.length_scale, args.velocity_scale, args.train_r_max,
                 args.phi_r_min, args.phi_r_max, args.max_particles, args.seed)


if __name__ == "__main__":
    main()
