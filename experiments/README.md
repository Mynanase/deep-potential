# Experiment entry points

This package exposes exactly three supported training/evaluation commands:

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.run_eval configs/runs/<run>.yaml
```

They are thin adapters: argument parsing is limited to the run YAML path, while
all reusable behavior lives in `dpjax.workflows`.

The other modules in this directory are data preparation, inspection and smoke
utilities. Do not add model implementations, plotting pipelines, or alternative
training CLIs here.
