# Experiment entry points

This package exposes three foreground training/evaluation workers:

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.run_eval configs/runs/<run>.yaml
```

For detached server execution, use the single launcher instead of hand-written
`nohup`, environment, and redirection commands:

```bash
python -m experiments.launch phi configs/runs/<run>.yaml
```

`experiments.runtime` applies project-safe process defaults before JAX is
imported. `experiments.launch` owns detachment, run-scoped console logs, and PID
files; the workers remain foreground commands that are easy to test and debug.

Argument parsing is limited to the run YAML path. Repository-specific behavior
lives under `experiments.datasets`, `experiments.workflows`,
`experiments.diagnostics`, `experiments.plotting`, and
`experiments.validation`.

`experiments.workflows.model_config` validates reusable `configs/models` files
and rejects data, logging, checkpoint, GPU, evaluation, and plotting fields.
`experiments.workflows.config` owns `configs/runs` and resolves model recipes
with the selected run's data and execution settings before a stage starts.

`dpjax` is the array-only numerical dependency. New data adapters, artifact
formats, figures, or simulator-truth checks belong here and must not require a
change to `dpjax`.
