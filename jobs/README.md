# Server helpers

Training and evaluation no longer use per-experiment shell wrappers. Launch the
three Python entry points directly with `nohup`, using a run YAML as the only
argument; see `docs/server_agent_run_guide.md`.

`_common.sh` contains shared environment setup for the remaining data-preparation
helper. `prepare_halo12_df.sh` converts Auriga input into the canonical HDF5
format and is not part of model training.
