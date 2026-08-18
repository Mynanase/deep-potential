# marimo analysis

Training and evaluation run as standalone processes. The Marimo files in this
directory only read saved `metrics.csv`, JSON, and NPZ artifacts, then call
composable functions from `experiments.plotting`.

```bash
conda activate dp-jax
marimo edit analysis/halo12.py
```

Choose experiment directories with the text inputs. Marimo owns selection,
comparison, layout, and display; it does not implement low-level Matplotlib
logic or launch expensive work.

Optional analytic/simulator truth is generated separately by editing and
running `validate_plummer_truth.py` or `validate_auriga_truth.py`. These are
lightweight one-off scripts with constants at the top, not run-configured
workflows. Marimo only reads their JSON/NPZ output.
