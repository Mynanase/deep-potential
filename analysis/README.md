# marimo analysis

Training and evaluation run as standalone processes.  The marimo files in
this directory only read saved `metrics.csv`, JSON, NPZ, and figure artifacts.

```bash
conda activate dp-jax
marimo edit analysis/halo12.py
```

Set the default run directory near the top of the notebook, or change it with
the text input in the notebook UI.
