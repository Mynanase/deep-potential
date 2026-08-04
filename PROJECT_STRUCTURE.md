# Project structure

The active code has a deliberately small public surface: three run entry points,
one workflow package and one Marimo analysis app.

```text
deep-potential/
├── analysis/
│   └── halo12.py
├── configs/
│   ├── runs/                  # complete experiment specifications
│   └── *.yaml                 # model/stage hyperparameters
├── dpjax/
│   ├── diagnostics/           # read-only artifact readers and lightweight plots
│   ├── workflows/
│   │   ├── config.py          # run schema, layout and safety checks
│   │   ├── logging.py         # CSV/W&B/TensorBoard logging
│   │   ├── training/          # DF and Phi implementations
│   │   └── evaluation/        # DF, Phi and Auriga truth implementations
│   ├── data.py
│   ├── datasets/
│   ├── flows/
│   ├── models/
│   ├── physics/
│   └── plotting/
├── experiments/
│   ├── run_df.py
│   ├── run_phi.py
│   ├── run_eval.py
│   └── <data and smoke utilities>
├── jobs/                      # environment bootstrap and data preparation only
├── tests/
└── runs/                      # generated artifacts; not source code
```

## Dependency direction

```text
configs/runs/*.yaml
        │
        ▼
experiments.run_{df,phi,eval}       analysis/halo12.py
        │                                  │
        ▼                                  ▼
dpjax.workflows                    dpjax.diagnostics
        │                                  │
        └──────────────► dpjax core ◄──────┘
```

`experiments/` contains orchestration only. It must not contain numerical
training/evaluation implementations. `dpjax.workflows` must not import
`experiments`. `analysis/` reads persisted results and must not invoke a
training workflow.

## Experiment lifecycle

For every trial, DF and Phi live under the same directory. This makes their
pairing explicit and prevents accidentally evaluating a Phi checkpoint against
an unrelated DF:

```text
runs/<name>/<trial>/df  ->  runs/<name>/<trial>/phi
                         ->  runs/<name>/<trial>/eval
                         ->  runs/<name>/<trial>/plots
```

Long-running stages are separate processes so they can be resumed and monitored
independently. The resolved `run.yaml`, per-stage `config.yaml`, checkpoints and
metrics provide the reproducibility boundary.

See `docs/server_agent_run_guide.md` for concrete commands.
