#!/usr/bin/env python3
"""Check whether an eta-only HDF5 (training data) is a contiguous slice
of a stars snapshot (with Masses / Potential) by matching 6D coordinates.

Usage:
    python scripts/check_particle_alignment.py \
        --eta-h5 data/halo_12_train.h5 \
        --stars-hdf5 data/halo_12_stars.hdf5

If a contiguous slice is detected, the script prints the start/end index
in the stars-hdf5 file, so downstream scripts can pair Masses / Potential
one-to-one with eta rows.

Strategy:
  1. Take the first k=10 rows of eta-h5.
  2. For each row, do exact 6D coord match against stars-hdf5 (brute force
     over N particles, ~1s for N=1.6e6).
  3. If matches are a contiguous [start, start+k), verify by comparing
     eta[0:1000] vs stars[start:start+1000].
"""
import argparse
import sys
import numpy as np
import h5py


def load_eta_and_meta(path):
    """Return a dict with 'eta' and optionally 'mass', 'potential', 'ids'."""
    out = {}
    with h5py.File(path, "r") as f:
        if "eta" in f:
            out["eta"] = np.asarray(f["eta"], dtype=np.float64)
        elif "PartType4" in f:
            g = f["PartType4"]
            out["eta"] = np.stack(
                [np.asarray(g[k], dtype=np.float64)
                 for k in ["x", "y", "z", "vx", "vy", "vz"]],
                axis=1,
            )
            if "Masses" in g:
                out["mass"] = np.asarray(g["Masses"], dtype=np.float64)
            if "Potential" in g:
                out["potential"] = np.asarray(g["Potential"], dtype=np.float64)
            if "ParticleIDs" in g:
                out["ids"] = np.asarray(g["ParticleIDs"])
        else:
            raise KeyError(f"No 'eta' or 'PartType4' in {path}")
    return out


def find_row_indices(eta_small, eta_big, atol=1e-3):
    """For each row in eta_small, return the index of the matching row
    in eta_big (or -1 if not found). Assumes unique 6D coords."""
    indices = []
    n_big = eta_big.shape[0]
    for r in eta_small:
        # brute-force sum abs diff over 6D
        # Vectorized: chunked to save memory
        diff = np.sum(np.abs(eta_big - r), axis=1)
        idx = np.where(diff < atol)[0]
        if len(idx) == 0:
            indices.append(-1)
        elif len(idx) == 1:
            indices.append(int(idx[0]))
        else:
            # Multiple matches — pick the first by default but warn
            print(f"  WARN: {len(idx)} duplicate matches for row, picking first", file=sys.stderr)
            indices.append(int(idx[0]))
    return indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eta-h5", required=True,
                   help="eta-only h5 (e.g. data/halo_12_train.h5)")
    p.add_argument("--stars-hdf5", required=True,
                   help="stars snapshot with Masses (e.g. data/halo_12_stars.hdf5)")
    p.add_argument("--k", type=int, default=10,
                   help="number of rows to match (default 10)")
    p.add_argument("--verify", type=int, default=1000,
                   help="number of rows to verify once slice is found")
    args = p.parse_args()

    a = load_eta_and_meta(args.eta_h5)
    b = load_eta_and_meta(args.stars_hdf5)

    print(f"eta-h5        N = {a['eta'].shape[0]}  ({args.eta_h5})")
    print(f"stars-hdf5    N = {b['eta'].shape[0]}  ({args.stars_hdf5})")
    print(f"extra fields in stars: {[k for k in b if k not in ('eta',)]}")

    print(f"\neta-h5 first 3 rows:")
    print(a["eta"][:3])
    print(f"stars-hdf5 first 3 rows:")
    print(b["eta"][:3])

    # Step 1: find first k rows of a in b
    print(f"\nMatching eta-h5 first {args.k} rows against stars-hdf5 ...")
    indices = find_row_indices(a["eta"][:args.k], b["eta"])

    if any(idx < 0 for idx in indices):
        print(f"  FAIL: not all {args.k} rows matched.")
        print(f"  positions: {indices}")
        print("  Suggest: the two files are not in same ordering, "
              "need ParticleIDs-based alignment.")
        sys.exit(1)

    print(f"  positions in stars-hdf5: {indices}")
    diffs = np.diff(indices)
    print(f"  consecutive diffs:     {diffs.tolist()}")

    if len(diffs) > 0 and np.all(diffs == 1):
        start = indices[0]
        end = start + a["eta"].shape[0]
        print(f"\n  Contiguous slice detected: [{start}, {end}) "
              f"of stars-hdf5 (size = {a['eta'].shape[0]})")

        if end <= b["eta"].shape[0]:
            # verify
            v = args.verify
            v = min(v, a["eta"].shape[0], b["eta"].shape[0] - start)
            ok = np.allclose(a["eta"][:v], b["eta"][start:start + v], atol=1e-4)
            print(f"  Verify eta-h5[0:{v}] vs stars[{start}:{start + v}]: allclose = {ok}")
            if ok:
                print(f"\n  >>> ALIGN: eta-h5 rows 0..{a['eta'].shape[0]-1} "
                      f"correspond to stars-hdf5 rows {start}..{end - 1}.")
                print(f"  >>> To pair Masses / Potential with eta-h5, "
                      f"use stars_slice = slice({start}, {end}).")
            else:
                print("\n  ⚠ contiguous slice but verify failed — possible ordering mismatch deeper in file")
        else:
            print(f"  FAIL: required slice [{start}, {end}) exceeds stars-hdf5 (N={b['eta'].shape[0]})")
    else:
        print("\n  ⚠ Matched rows are DIFFERENT ordering; eta-h5 is a SUBSET")
        print("    but not contiguous. Possible causes: train/val shuffle, random subsample.")
        print("    You will need sort-based alignment before using Masses.")
        # Try sorted matching instead
        print("\n  Trying sort-based alignment ...")
        # Build a key by combining 6D coords into a hashable; use lexsort.
        a_idx = np.lexsort(a["eta"].T[::-1])
        b_idx = np.lexsort(b["eta"].T[::-1])
        a_sorted = a["eta"][a_idx]
        b_sorted = b["eta"][b_idx]
        # two-pointer scan: count rows of a appearing in b
        n_match, i, j = 0, 0, 0
        while i < a_idx.shape[0] and j < b_idx.shape[0]:
            cmp = np.lexsort((a_sorted[i],))[0]  # placeholder
            # just compare rows
            diff = np.sum(np.abs(a_sorted[i] - b_sorted[j]))
            if diff < 1e-3:
                n_match += 1
                i += 1
                j += 1
            elif np.lexsort(a_sorted[[i]].T.T[0]) < np.lexsort(b_sorted[[j]].T.T[0]):
                # not robust; just advance j
                i += 1
            else:
                j += 1
        print(f"  sort-based {n_match}/{a_idx.shape[0]} rows found in stars-hdf5")
        if n_match == a_idx.shape[0]:
            print("  >>> ALIGN via sort: every eta-h5 row has a unique match in stars-hdf5.")
            print(f"      Use lexsort indices a_idx and b_idx to map.")


if __name__ == "__main__":
    main()