#!/usr/bin/env python
"""Verify an s11 evaluation file against the seed-0 reference file.

Usage: check_s11_file.py NEW REF
Checks: same particle count, shuffle_seed == 11, identical particle_id set,
converted (dimensionless) units.  Exits non-zero on any mismatch.
"""
import sys

import h5py
import numpy as np


def main():
    new_path, ref_path = sys.argv[1], sys.argv[2]
    with h5py.File(new_path, "r") as f_new, h5py.File(ref_path, "r") as f_ref:
        n_new = f_new["eta"].shape[0]
        n_ref = f_ref["eta"].shape[0]
        assert n_new == n_ref, f"n mismatch: {n_new} vs reference {n_ref}"
        seed = int(f_new.attrs.get("shuffle_seed", -1))
        assert seed == 11, f"shuffle_seed={seed}, expected 11"
        pid_new = np.sort(np.asarray(f_new["particle_id"][:], dtype=np.int64))
        pid_ref = np.sort(np.asarray(f_ref["particle_id"][:], dtype=np.int64))
        assert np.array_equal(pid_new, pid_ref), "particle_id set differs from reference"
        attrs = dict(f_new.attrs)
        attrs.update(dict(f_new["eta"].attrs))
        units = (attrs.get("length_unit"), attrs.get("velocity_unit"))
        # prepare_data.py output is already nondimensionalized (q=x/L, p=v/V);
        # the kpc / km-s requirement applies to the *source* file, not here.
        assert units == ("dimensionless", "dimensionless"), f"unexpected units: {units}"
        print(f"s11 file verified: n={n_new}, shuffle_seed=11, pid set matches "
              f"reference, dimensionless units, weighting={attrs.get('weighting')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
