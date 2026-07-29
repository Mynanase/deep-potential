# Scheduler-free server jobs

These Bash scripts run directly on a Linux server and do not require a batch
scheduler. Run them from any directory; they locate the repository from their
own path unless `DEEP_POTENTIAL_ROOT` is set.

The training-job default logger is W&B. Set `LOGGER=csv` for an entirely local
or offline run.
Set `GPU_DEVICES`, for example `GPU_DEVICES=0,1`, to restrict visible GPUs.

The ensemble script runs seeds sequentially so multiple FFJORD processes do not
compete for the same GPU memory. To use separate GPUs concurrently, launch
separate processes with disjoint `SEEDS`, `GPU_DEVICES`, and preferably
different log files.

`eval_halo12_df_ensemble.sh` accepts one or more seeds. One seed produces all
data/model distribution plots and Stein diagnostics; two or more seeds also
produce score-repeatability metrics. The evaluation job renders plots by
default after writing JSON/NPZ outputs; set `SKIP_PLOT=1` to keep evaluation
and rendering separate.

See `docs/auriga_halo12_df_stage.md` for commands and acceptance criteria.
