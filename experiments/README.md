# Experiment entry points

Each run YAML describes one concrete experiment and one output directory:

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.run_eval configs/runs/<run>.yaml
python -m experiments.run_plot configs/runs/<run>.yaml
python -m experiments.run_plot configs/runs/<run>.yaml --only training df
```

For detached server execution, use the launcher:

```bash
python -m experiments.launch plot configs/runs/<run>.yaml
```

The launcher owns process detachment, logs, PID files, and runtime environment
defaults. The worker entry points remain foreground commands that accept only
the run YAML path.

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
python -m experiments.run_df configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run_df \
  configs/runs/halo12_clean_outer_clump_v1.yaml
python -m experiments.run_phi configs/runs/halo12_raw_no_clip_v1.yaml
python -m experiments.run_phi \
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

Use `notebooks/figure_debug.ipynb` to render one registered figure from saved
artifacts. Jupyter does not train, sample a model, or invoke `run_eval`.

## Optional truth checks

Truth checks are deliberately outside the run YAML and `run_eval.py`. Edit the
constants in a one-off script and generate the arrays once:

```bash
python analysis/validate_plummer_truth.py
python analysis/validate_auriga_truth.py
```

They create flat `validation_<kind>_{metrics.json,diagnostics.npz}` artifacts
under `results/data/`. The scripts do not render or save figures.
