# deep-potential

JAX/Flax implementation of a two-stage method for recovering gravitational
potentials from phase-space snapshots:

1. fit a distribution function (DF) with a normalizing flow;
2. freeze the DF and fit a potential network with the collisionless Boltzmann
   equation.

Experiments use one checked-in run YAML. For routine work, run each stage
explicitly so that training, evaluation, and plotting can be repeated or
recovered independently:

```bash
python -m experiments.run_df configs/runs/halo12_static_v1.yaml
python -m experiments.eval_df configs/runs/halo12_static_v1.yaml
python -m experiments.plot_df configs/runs/halo12_static_v1.yaml

python -m experiments.run_phi configs/runs/halo12_static_v1.yaml
python -m experiments.eval_phi configs/runs/halo12_static_v1.yaml
python -m experiments.plot_phi configs/runs/halo12_static_v1.yaml
```

Thin composite commands are available when the whole chain should run in the
foreground:

```bash
python -m experiments.run df configs/runs/halo12_static_v1.yaml
python -m experiments.run phi configs/runs/halo12_static_v1.yaml
python -m experiments.run all configs/runs/halo12_static_v1.yaml
```

Every expensive worker is still a separate process. A composite command stops
at the first failed worker and leaves completed artifacts intact; rerun the
failed single-stage command after correcting the cause. Checkpoint continuation
is controlled only by `execution.resume` in the run YAML.

For a detached server run with automatic console logs, use the project
launcher. It accepts the corresponding stage and pipeline names, for example:

```bash
python -m experiments.launch all configs/runs/halo12_static_v1.yaml
python -m experiments.launch eval-df configs/runs/halo12_static_v1.yaml
```

`experiments.run_eval` and `experiments.run_plot --only` remain compatibility
interfaces for older aggregate workflows. New automation should use the
stage-specific entry points above. Evaluation JSON/NPZ files are derived from
the saved run snapshot and checkpoints and can be regenerated without
retraining. Post-training figure debugging uses
`notebooks/figure_debug.ipynb`; it only reads saved artifacts and renders one
explicitly selected figure at a time.

The `dpjax` package is an array-only numerical core with a single `(N, 6)`
phase-space contract. Dataset adapters, file I/O, plotting, run orchestration,
and optional simulator-truth validation live under `experiments/`.

Configuration is split into reusable model recipes under `configs/models` and
operational runs under `configs/runs`.

See `README_JAX.md` for setup, `PROJECT_STRUCTURE.md` for architecture, and
`docs/architecture_operation_guide.md` for modification and execution steps.

Historical TensorFlow/Sonnet code under `archive/legacy_tensorflow/` is not part
of the supported runtime.
