# Halo12 outer-clump removal experiment

## Scope and current status

This branch treats outer-clump removal as an experimental intervention on the
host-halo distribution function, not as a known ground-truth particle label.
The source file `data/halo_12_stars.hdf5` remains unchanged. The cleaning step
produces a canonical Deep Potential HDF5, an exact removed-row artifact, and a
static diagnostic:

```text
data/auriga/halo12_all_mass_clean_outer_clump.h5
runs/halo12/outer-clump-removal/outer_clump_mask.npz
runs/halo12/outer-clump-removal/outer_clump_diagnostic.png
```

Generate them with:

```bash
python -m experiments.datasets.clean_outer_clump
```

Existing outputs are not overwritten unless `--overwrite` is supplied.

## Operational selection

The selection searches only at `r >= 40 kpc`, counts three-dimensional
neighbors within `1 kpc`, and labels particles with at least 16 neighbors as
density-core particles. Starting at the densest particle, core particles are
linked transitively and their one-link border is included. This is a
single-cluster DBSCAN density-connectivity rule implemented with SciPy
`cKDTree`; it does not require a spherical, tubular, or rectangular model.

For the current Halo12 source:

| Quantity | Value |
|---|---:|
| Source rows | 1,652,969 |
| Outer search rows (`r >= 40 kpc`) | 58,979 |
| Removed rows | 9,683 (0.586%) |
| Median position `(x,y,z)` | `(2.092, -9.411, -58.095) kpc` |
| Radial range | `54.09--63.45 kpc` |
| Positional PCA scales | `(1.950, 1.652, 0.919) kpc` |
| Mean velocity `(vx,vy,vz)` | `(-152.79, -224.46, -4.43) km/s` |
| Velocity PCA scales | `(35.15, 34.03, 24.39) km/s` |
| Peak neighbors within 1 kpc | 1,399 |
| Maximum after removing the component | 16 |

The component is therefore coherent in both position and velocity. It is not
only a projection artifact. An axis-aligned box covering the selected rows
would contain 9,793 particles, including 110 rows not density-connected to the
component. The density selection avoids deleting those ambient rows.

## Parameter sensitivity

The table compares alternative selections with the default mask. `J` is the
Jaccard overlap in removed source indices.

| `r_min` | link length | min neighbors | removed | `J` |
|---:|---:|---:|---:|---:|
| 30 | 1.00 | 16 | 9,683 | 1.000 |
| 40 | 0.75 | 8 | 9,566 | 0.987 |
| 40 | 0.75 | 16 | 9,285 | 0.959 |
| 40 | 0.75 | 32 | 8,833 | 0.912 |
| 40 | 1.00 | 8 | 9,757 | 0.992 |
| 40 | 1.00 | 16 | 9,683 | 1.000 |
| 40 | 1.00 | 32 | 9,536 | 0.985 |
| 40 | 1.25 | 8 | 9,805 | 0.988 |
| 40 | 1.25 | 16 | 9,791 | 0.989 |
| 40 | 1.25 | 32 | 9,753 | 0.993 |
| 50 | 1.00 | 16 | 9,683 | 1.000 |

The center is unchanged across these settings. Membership at the diffuse edge
is the main uncertainty; the `0.5 kpc` link-length result is more conservative
(7,771 rows, `J=0.803`) and should be retained as a sensitivity experiment if
the default clean run materially changes the inferred potential.

## Reversibility and data identity

The mask NPZ stores the source phase-space SHA-256, detection parameters,
removed input-row indices, original `source_index`, ParticleIDs, and local
neighbor counts. The clean HDF5 stores the same removed identity under its
`cleaning` group. Every retained row has been verified to match the source
position, velocity, ParticleID, mass, and potential exactly after the canonical
float32 conversion. `tracer_weight` is recomputed as
`mass / mean(clean mass)` and has mean one.

## Sigma-clipping preflight

The current static Halo12 run config applies a global, weighted
`clip_sigma=4.5` selection after loading the data. Replaying that exact
selection gives:

| Dataset | Source rows | DF support | Globally clipped |
|---|---:|---:|---:|
| Original | 1,652,969 | 1,602,922 | 50,047 |
| Clean | 1,643,286 | 1,596,306 | 46,980 |

Crucially, the original 4.5-sigma support retains `0 / 9,683` target-clump
members. Removing the clump before recomputing the global normalizer also moves
the clipping boundary: compared by original `source_index`, 6,733 background
rows are retained only by the original support and 117 only by the clean
support. Therefore, training the original and clean files with
`clip_sigma=4.5` would not measure the effect of this clump; it would mostly
compare two nearby global clipping boundaries after the clump was already gone.

## Deep Potential comparison

The causal pair disables global clipping so that the explicit density mask is
the only row-selection difference:

```bash
python -m experiments.run_df configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run_df \
  configs/runs/halo12_clean_outer_clump_v1.yaml
python -m experiments.run_phi configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run_phi \
  configs/runs/halo12_clean_outer_clump_v1.yaml
python -m experiments.run_eval configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run_eval \
  configs/runs/halo12_clean_outer_clump_v1.yaml
python -m experiments.run_plot configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run_plot \
  configs/runs/halo12_clean_outer_clump_v1.yaml
```

`configs/runs/halo12_static_v1.yaml` remains a third reference for the broader
global 4.5-sigma clipping policy, not the causal raw-versus-clump-clean pair.

Validate the paired configs, exact source-index partition, phase-space hash,
row alignment, and tracer weights before launching an expensive run:

```bash
python -m experiments.validation.halo12_clump_pair
```

After both DF checkpoints exist, run the common-target evaluation. It evaluates
both flows against the clean host data and uses identical score points, rather
than comparing two independently sampled evaluation targets:

```bash
python -m experiments.validation.halo12_clump_pair --evaluate-df
```

The scientific comparison should not stop at validation NLL. Compare original
and clean runs using:

1. held-out density and velocity diagnostics globally and at `50--65 kpc`;
2. score norms and score stability near the removed component and on ambient
   host-halo points at the same radius;
3. CBE residuals and their seed stability;
4. recovered potential and acceleration diagnostics against simulator truth;
5. the conservative `0.5 kpc` mask if the default mask changes the conclusion.

An improvement in the clean run would support the hypothesis that a coherent,
non-equilibrium outer component was contaminating the stationary host-halo DF.
It would not by itself prove that every removed particle is unbound or that
the component should be excluded for all scientific targets.
