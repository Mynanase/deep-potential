# One-off numeric validation

Interactive plotting lives in `notebooks/figure_debug.ipynb`; formal batch
plotting is owned by `python -m experiments.run_plot`.

The scripts retained here perform optional analytic or simulator-truth numeric
checks. They write flat JSON/NPZ artifacts to the selected run's
`results/data/` and do not train models or render figures:

```bash
conda activate dp-jax
python analysis/validate_plummer_truth.py
python analysis/validate_auriga_truth.py
python analysis/check_oracle_score.py
```
