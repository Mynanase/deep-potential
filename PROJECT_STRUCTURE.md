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
│   ├── run.py                     # df/phi/all composite subprocess runner
│   ├── run_df.py                  # train DF
│   ├── eval_df.py                 # derive DF evaluation artifacts
│   ├── plot_df.py                 # render DF figures
│   ├── run_phi.py                 # train Phi
│   ├── eval_phi.py                # derive Phi evaluation artifacts
│   ├── plot_phi.py                # render Phi figures
│   ├── run_eval.py                # compatibility aggregate evaluation
│   ├── run_plot.py                # compatibility aggregate plotting
│   └── list_runs.py
├── analysis/
│   └── validate_*_truth.py        # editable one-off truth scripts
├── notebooks/
│   └── figure_debug.ipynb         # saved-artifact, single-figure debugging
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
plot_df / plot_phi / figure_debug.ipynb
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
└── results/
    ├── data/                     # flat df_*/phi_* diagnostics
    ├── figures/                  # official PNG/PDF
    ├── debug/                    # notebook output
    ├── manifest.json
    └── report.md
```

The run has no required trial subdirectory. Detached launch names map directly
to single stages (`df`, `eval-df`, `plot-df`, `phi`, `eval-phi`, `plot-phi`) or
pipelines (`df-pipeline`, `phi-pipeline`, `all`); each gets a matching log and
PID file under `logs/`. The legacy `eval` and `plot` launcher names remain for
the compatibility entry points.

`results/data/df_*` and `results/data/phi_*` evaluation files are derived from
the immutable `run.yaml`, input data, and trained checkpoints. They may be
regenerated independently; plotting consumes them and does not train or
evaluate. Composite `experiments.run df|phi|all` commands execute each worker
in a separate subprocess and stop at the first failure without deleting prior
artifacts. Training restart behavior remains governed by `execution.resume`.

Potential and acceleration truth are not part of the core phase-space input.
They are absent from the run schema and routine evaluation entry points. When
needed for a one-off synthetic/simulation check, standalone analysis scripts
write optional flat `validation_<kind>_*` artifacts under `results/data/`.
