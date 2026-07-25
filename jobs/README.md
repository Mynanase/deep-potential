# Scheduler-free server jobs

These Bash scripts run directly on a Linux server and do not require a batch
scheduler. Run them from any directory; they locate the repository from their
own path unless `DEEP_POTENTIAL_ROOT` is set.

The default logger is local CSV. Set `LOGGER=wandb` only when W&B is configured.
Set `GPU_DEVICES`, for example `GPU_DEVICES=0,1`, to restrict visible GPUs.

The ensemble script runs seeds sequentially so multiple FFJORD processes do not
compete for the same GPU memory. To use separate GPUs concurrently, launch
separate processes with disjoint `SEEDS`, `GPU_DEVICES`, and preferably
different log files.

See `docs/auriga_halo12_df_stage.md` for commands and acceptance criteria.
