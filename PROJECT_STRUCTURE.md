# Project structure

The repository separates the array-only numerical library from data and
experiment operations.

```text
deep-potential/
├── dpjax/                         # reusable numerical core
│   ├── normalization.py           # (N, 6) validation and model normalization
│   ├── diagnostics/df.py          # reference vs generated six-dimensional data
│   ├── flows/                     # DF models and array-level APIs
│   ├── models/                    # potential model and derivatives
│   ├── physics/                   # CBE and generic physical calculations
│   └── utils/                     # numerical tree helpers
├── experiments/                   # repository-specific operational layer
│   ├── datasets/                  # HDF5, Auriga, Plummer and row selection
│   ├── workflows/                 # YAML, logging, checkpoints and run layout
│   ├── diagnostics/               # persisted artifact readers and metrics
│   ├── plotting/                  # all matplotlib figure builders
│   ├── validation/                # reusable optional truth-check functions
│   ├── launch.py                  # detached stage launcher, logs and PID files
│   ├── runtime.py                 # pre-JAX project environment defaults
│   ├── run_df.py
│   ├── run_phi.py
│   └── run_eval.py
├── analysis/
│   ├── {halo12,plummer_rcut}.py   # read-only Marimo applications
│   └── validate_*_truth.py        # editable one-off truth scripts
├── configs/
│   ├── models/{df,phi}/           # reusable model/training recipes
│   └── runs/                      # data, execution, evaluation and plots
├── tests/
└── runs/                          # generated artifacts
```

## Core contract

The core accepts a finite floating-point array with shape `(N, 6)` and column
order `[x, y, z, vx, vy, vz]`. It does not know the source dataset, file
format, directory layout, simulator truth, or plotting convention.

`dpjax.normalization` is model state rather than upstream data preparation.
Its mean/std values are needed to train stably and convert scores and potential
gradients back to physical coordinates. Persistence of those values belongs to
`experiments.datasets.phase_space`.

## Dependency direction

```text
configs / files / datasets
            │
            ▼
experiments.workflows ─────► dpjax
            │                  ▲
            ▼                  │
experiments.diagnostics        │
experiments.plotting ──────────┘
            │
            ▼
analysis/halo12.py
```

The dependency is one-way: `experiments -> dpjax`. Core code must not import
`experiments`, HDF5, YAML, Matplotlib, Orbax, W&B, or filesystem path helpers.

## Configuration contract

Configuration files have exactly two roles:

- `configs/models/**/*.yaml` uses schema `dpjax.model.v1` and contains model
  architecture, loss, optimizer, schedule, regularization, batch size and
  epochs;
- `configs/runs/*.yaml` uses schema `dpjax.run.v2`; each file describes one
  concrete experiment and contains data,
  preprocessing, model references, seeds, execution, evaluation and plots.

Model YAML is semantically aligned with `dpjax`, but it is loaded and validated
by `experiments.workflows`; the numerical core never reads YAML. See
`docs/architecture_operation_guide.md` for the extension workflow.

## Output contract

```text
runs/<experiment>/
├── run.yaml
├── logs/
├── df/
├── phi/
├── eval/                         # flat df_*/phi_* diagnostics
└── plots/
```

The run has no required trial subdirectory. `logs/{df,phi,eval}.log` and matching
PID files are written when stages are started through `experiments.launch`.

Potential and acceleration truth are not part of the core phase-space input.
They are absent from the run schema and `run_eval.py`. When needed for a
one-off synthetic/simulation check, standalone analysis scripts write optional
`validation/<kind>/{metrics.json,diagnostics.npz}` artifacts for Marimo.
