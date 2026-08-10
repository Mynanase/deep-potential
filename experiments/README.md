# Experiment entry points

This package exposes exactly three supported training/evaluation commands:

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.run_eval configs/runs/<run>.yaml
```

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
