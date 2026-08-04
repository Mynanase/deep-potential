# Deep Potential JAX quick start

## Environment

```bash
conda activate dp-jax
pip install -e ".[notebook,tracking]"
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The supported training and evaluation interface is a run-level YAML. Copy the
example before changing scientific parameters:

```bash
cp configs/runs/halo12_static_v1.yaml configs/runs/my_run.yaml
```

Set `name`, `output_dir`, `data.path`, trial seeds, stage config paths and
overrides in that file. Model-specific hyperparameters remain in the referenced
DF and Phi YAML files. Do not pass model parameters as shell arguments.

## Run the stages

```bash
python -m experiments.run_df configs/runs/my_run.yaml
python -m experiments.run_phi configs/runs/my_run.yaml
python -m experiments.run_eval configs/runs/my_run.yaml
```

Each entry point accepts only the run YAML path. `run_phi` automatically uses
the DF from the same trial. `run_eval` persists metrics and arrays before they
are displayed by Marimo.

For a disconnected SSH session, launch one stage per background process:

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_df configs/runs/my_run.yaml \
  > logs/my-run-df.log 2>&1 < /dev/null &
```

Repeat with `run_phi` and `run_eval` only after the preceding stage succeeds.
Monitor with `tail -f logs/my-run-df.log` or W&B.

## Output contract

```text
runs/<run-name>/
├── run.yaml
├── trial_00/
│   ├── df/       # checkpoint, preprocessing, config, metrics
│   ├── phi/      # checkpoint, config, metrics
│   ├── eval/     # persisted JSON/NPZ diagnostics
│   └── plots/    # rendered figures
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
