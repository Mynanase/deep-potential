# deep-potential

JAX/Flax implementation of a two-stage method for recovering gravitational
potentials from phase-space snapshots:

1. fit a distribution function (DF) with a normalizing flow;
2. freeze the DF and fit a potential network with the collisionless Boltzmann
   equation.

Experiments use one checked-in run YAML and three stable worker entry points:

```bash
python -m experiments.run_df configs/runs/halo12_static_v1.yaml
python -m experiments.run_phi configs/runs/halo12_static_v1.yaml
python -m experiments.run_eval configs/runs/halo12_static_v1.yaml
```

For a detached server run with automatic console logs, use the project launcher:

```bash
python -m experiments.launch phi configs/runs/halo12_static_v1.yaml
```

The expensive stages are standalone processes. Post-training exploration uses
the git-friendly Marimo app at `analysis/halo12.py` and only reads saved
artifacts.

The `dpjax` package is an array-only numerical core with a single `(N, 6)`
phase-space contract. Dataset adapters, file I/O, plotting, run orchestration,
and optional simulator-truth validation live under `experiments/`.

Configuration is split into reusable model recipes under `configs/models` and
operational runs under `configs/runs`.

See `README_JAX.md` for setup, `PROJECT_STRUCTURE.md` for architecture, and
`docs/architecture_operation_guide.md` for modification and execution steps.

Historical TensorFlow/Sonnet code under `archive/legacy_tensorflow/` is not part
of the supported runtime.
