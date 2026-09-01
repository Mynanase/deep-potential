# Experiment entry points

Each run YAML describes one concrete experiment and one output directory. The
recommended daily interface keeps every stage independently callable:

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.eval_df configs/runs/<run>.yaml
python -m experiments.plot_df configs/runs/<run>.yaml

python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.eval_phi configs/runs/<run>.yaml
python -m experiments.plot_phi configs/runs/<run>.yaml
```

For a foreground DF pipeline, Phi pipeline, or full run, use the thin composite
entry point:

```bash
python -m experiments.run df configs/runs/<run>.yaml
python -m experiments.run phi configs/runs/<run>.yaml
python -m experiments.run all configs/runs/<run>.yaml
```

The composite entry point starts every worker as a fresh subprocess. This keeps
JAX/GPU state isolated between stages. It is fail-fast: the first non-zero
worker exit stops the chain, while artifacts from earlier completed workers are
kept. It does not silently skip completed training. Recover with the relevant
single-stage command; set `execution.resume: true` only to continue a compatible
training checkpoint.

The selected workflow also validates its dependencies before starting:
stage-specific evaluation must be enabled, `phi` requires a completed DF
checkpoint (which may come from `phi.df_run`), and `all` requires Phi to use the
same run's DF. Use the independent stages when a run intentionally disables an
evaluation or mixes artifacts from separate runs.

For detached server execution, use the launcher. Stage names are `df`,
`eval-df`, `plot-df`, `phi`, `eval-phi`, and `plot-phi`; pipeline names are
`df-pipeline`, `phi-pipeline`, and `all`:

```bash
python -m experiments.launch all configs/runs/<run>.yaml
python -m experiments.launch plot-df configs/runs/<run>.yaml
```

The launcher owns process detachment, logs, PID files, and runtime environment
defaults. The worker entry points remain foreground commands that accept only
the run YAML path. A successful launcher return means that the worker was
started; inspect its run-scoped log for completion.

For the same run, active pipelines conflict with covered standalone workers.
Evaluation and plotting workers are also mutually exclusive so plots cannot
read partially written diagnostics or race on the shared manifest and report.

`experiments.run_eval` and `experiments.run_plot --only` remain compatibility
interfaces for callers that need the older aggregate behavior:

```bash
python -m experiments.run_eval configs/runs/<run>.yaml
python -m experiments.run_plot configs/runs/<run>.yaml --only df phi
```

They are not the recommended interface for new scripts. There is no public
`--stage` selector; use `eval_df`, `eval_phi`, `plot_df`, or `plot_phi` instead.

Repository-specific behavior belongs under `experiments.datasets`,
`experiments.workflows`, `experiments.diagnostics`, `experiments.plotting`, and
`experiments.validation`. `dpjax` remains the array-only numerical dependency.

## Halo12 outer-clump cleaning experiment

The Halo12 cleaning branch uses a reversible density-connected selection. It
does not modify the original Gadget/Auriga HDF5. Generate the canonical clean
dataset, exact removed-row artifact, and a static diagnostic with:

```bash
python -m experiments.datasets.clean_outer_clump
```

The defaults search at `r >= 40`, count neighbors within a length of `1`, and
grow the component containing the densest point from particles with at least
`16` neighbors. Every parameter and the source phase-space SHA-256 are stored
with the output. The retained rows preserve their original `source_index`.

Run the paired no-clip control and clean-data experiment. These two configs use
the same model recipes, random seeds, and mass weighting; only the 9,683
explicitly removed rows differ:

```bash
python -m experiments.run all configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run all \
  configs/runs/halo12_clean_outer_clump_v1.yaml
```

Do not use `halo12_static_v1.yaml` as the only causal control for this question:
its global `clip_sigma: 4.5` selection already removes every detected clump
member and also removes unrelated tail particles.

Before training, or after moving the data to another host, revalidate the exact
pair contract:

```bash
python -m experiments.validation.halo12_clump_pair
```

After both DF checkpoints exist, evaluate them on the same clean target rows
and the same randomly selected score points:

```bash
python -m experiments.validation.halo12_clump_pair --evaluate-df
```

## Output contract

```text
runs/<experiment>/
├── run.yaml
├── logs/
├── df/
├── phi/
└── results/
    ├── data/      # flat JSON/NPZ artifacts
    ├── figures/   # official PNG/PDF
    ├── debug/     # notebook output
    ├── manifest.json
    └── report.md
```

Evaluation writes flat, stage-specific artifacts under `results/data/`:

```text
df_config.yaml          phi_config.yaml
df_metrics.json         phi_metrics.json
df_diagnostics.npz      phi_diagnostics.npz
df_samples.npz
```

These are derived artifacts: `eval_df` and `eval_phi` reconstruct them from the
immutable run snapshot, input data, and saved checkpoints, so evaluation can be
rerun without retraining. The specialized plot commands require their stage's
evaluation artifacts and fail clearly when one is missing; they never train or
evaluate as a side effect.

DF artifacts are loaded independently with `load_df_metrics`,
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

Use `notebooks/figure_debug.ipynb` to render one registered figure from saved
artifacts. Jupyter does not train, sample a model, or invoke an evaluation
entry point.

## Optional truth checks

Truth checks are deliberately outside the run YAML and routine evaluation
entry points. Edit the constants in a one-off script and generate the arrays
once:

```bash
python analysis/validate_plummer_truth.py
python analysis/validate_auriga_truth.py
```

They create flat `validation_<kind>_{metrics.json,diagnostics.npz}` artifacts
under `results/data/`. The scripts do not render or save figures.
