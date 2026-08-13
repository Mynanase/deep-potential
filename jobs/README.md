# Server helpers

Training and evaluation no longer use per-experiment shell wrappers. Use
`python -m experiments.launch <df|phi|eval> <run.yaml>` for detached execution,
run-scoped logs, and PID tracking; see `docs/server_agent_run_guide.md`.

`_common.sh` contains shared environment setup for the remaining data-preparation
helper. `prepare_halo12_df.sh` converts Auriga input into the canonical HDF5
format and is not part of model training.
