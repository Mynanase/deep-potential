# Experiment entry points

Each run YAML describes one concrete experiment and one output directory:

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.run_eval configs/runs/<run>.yaml
```

For detached server execution, use the launcher:

```bash
python -m experiments.launch phi configs/runs/<run>.yaml
```

The launcher owns process detachment, logs, PID files, and runtime environment
defaults. The worker entry points remain foreground commands that accept only
the run YAML path.

Repository-specific behavior belongs under `experiments.datasets`,
`experiments.workflows`, `experiments.diagnostics`, `experiments.plotting`, and
`experiments.validation`. `dpjax` remains the array-only numerical dependency.

## Output contract

```text
runs/<experiment>/
├── run.yaml
├── logs/
├── df/
├── phi/
├── eval/          # flat df_*/phi_* JSON/NPZ artifacts
└── plots/
```

DF evaluation artifacts are loaded independently with `load_df_metrics`,
`load_df_diagnostics`, and `load_df_samples`. Plotting functions accept loaded
arrays and return figures; they do not perform I/O or save images.

The generated `run.yaml` is an immutable resolved training snapshot. Existing
checkpoints are not overwritten. Set `execution.resume: true` only when the
resolved stage configuration matches the saved configuration.

## W&B tracking

Install the tracking extra and authenticate once:

```bash
pip install -e ".[operations,tracking]"
wandb login
```

Use `logging.backend: wandb` and `mode: online` for live tracking, or
`mode: offline` followed by `wandb sync`. Never store `WANDB_API_KEY` in a
checked-in YAML.

## Analysis

```bash
marimo edit analysis/halo12.py
marimo edit analysis/plummer_rcut.py
```

Marimo reads JSON/NPZ artifacts and composes functions from
`experiments.plotting`; it never starts training or expensive evaluation.

## Optional truth checks

Truth checks are deliberately outside the run YAML and `run_eval.py`. Edit the
constants in a one-off script, generate arrays once, and let Marimo read them:

```bash
python analysis/validate_plummer_truth.py
python analysis/validate_auriga_truth.py
```

They create `validation/<kind>/{metrics.json,diagnostics.npz}` below the chosen
experiment directory. The scripts do not render or save figures.
