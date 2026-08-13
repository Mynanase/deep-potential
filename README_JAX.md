# Deep Potential JAX quick start

## Environment

```bash
conda activate dp-jax
pip install -e ".[operations,notebook,tracking]"
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The supported training and evaluation interface is a run-level YAML. Copy the
example before changing scientific parameters:

```bash
cp configs/runs/halo12_static_v1.yaml configs/runs/my_run.yaml
```

Set `name`, `output_dir`, `data`, trial seeds, execution, evaluation and plots
in that file. Each trial references reusable model recipes under
`configs/models/{df,phi}`. Model files contain architecture, loss, optimizer and
training hyperparameters only. Do not pass model parameters as shell arguments.

The complete field ownership and extension workflow is documented in
`docs/architecture_operation_guide.md`.

## Run the stages

```bash
python -m experiments.run_df configs/runs/my_run.yaml
python -m experiments.run_phi configs/runs/my_run.yaml
python -m experiments.run_eval configs/runs/my_run.yaml
```

Each entry point accepts only the run YAML path. `run_phi` automatically uses
the DF from the same trial. `run_eval` persists metrics and arrays before they
are displayed by Marimo.

The reusable `dpjax` package accepts six-dimensional arrays only. HDF5 and
dataset-specific preparation live in `experiments.datasets`; adding another
data source or plot does not change the numerical core.

For a disconnected SSH session, launch one stage per background process:

```bash
# Start DF first.
python -m experiments.launch df configs/runs/my_run.yaml

# Start each later stage only after the preceding stage succeeds.
python -m experiments.launch phi configs/runs/my_run.yaml
python -m experiments.launch eval configs/runs/my_run.yaml
```

The launcher detaches from SSH, writes the PID, and captures stdout, stderr,
warnings, progress, and tracebacks. Monitor with:

```bash
tail -f runs/my_run/logs/phi.log
```

The project defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX is
imported. When all allocated GPUs should be used, do not set
`CUDA_VISIBLE_DEVICES`; JAX sees all devices exposed by the server/container.
A scheduler or parent environment may still restrict visible devices.

## W&B tracking

Install the tracking extra and authenticate once on each server user account:

```bash
pip install -e ".[operations,tracking]"
wandb login
```

Enable live W&B tracking in the run YAML:

```yaml
logging:
  backend: wandb       # wandb+tb also keeps TensorBoard output
  project: deep-potential
  mode: online         # online, offline, or disabled
  # entity: your-user-or-team
```

Use `backend: csv` to disable W&B while retaining local metrics. Use
`mode: offline` on a machine without network access and run `wandb sync` on the
generated stage `wandb/offline-run-*` directory later. Never store
`WANDB_API_KEY` in a checked-in YAML; use `wandb login` or a protected server
environment variable.

## Output contract

```text
runs/<run-name>/
├── run.yaml
├── logs/          # launcher console logs and PID files
├── trial_00/
│   ├── df/       # checkpoint, preprocessing, config, metrics
│   ├── phi/      # checkpoint, config, metrics
│   ├── eval/     # persisted JSON/NPZ diagnostics
│   ├── plots/    # rendered figures
│   └── validation/ # optional simulator-truth checks
└── summary/      # multi-trial DF summaries
```

The generated `run.yaml` is an immutable resolved snapshot. Existing
checkpoints are not overwritten. Set `execution.resume: true` only when the
current resolved configuration matches the saved stage configuration.

## Analysis

```bash
marimo edit analysis/halo12.py
```

Choose the run directory in the UI. The app reads existing `metrics.csv`, JSON,
NPZ and image artifacts; it never starts training.

## Data utilities

The remaining utility CLIs are intentionally separate from the run workflow:

```bash
python -m experiments.gendata_plummer --help
python -m experiments.prepare_auriga --help
python -m experiments.inspect_data --help
python -m experiments.smoke_dpjax --help
```

## Validation

```bash
pytest -q
marimo check --strict analysis/halo12.py
```

GPU training should be run from the user terminal or server environment. For
CPU-only smoke tests, set `JAX_PLATFORM_NAME=cpu`.
