# Agent notes — dpjax-phase2

This is an OpenResearch project. Use the installed `orx` guide for experiment
orchestration; its workflow does not need another copy here.

Use `README.md` to orient, then read the source relevant to the current task.
Historical documents are reference material, not a checklist for every task.

- Production uses host `gpu` and `/home/qiutao/miniforge3/envs/dp-jax/bin/python`; keep that environment intact. Set `JAX_PLATFORMS=cuda` for production (`cpu` locally) and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Preserve source data and completed run artifacts. `.note/` is private: do not read, commit, or disclose it.
- Simulation truth is evaluation-only. Preserve signed densities and negative-density diagnostics.
- Follow nearby code and keep changes direct. Run checks relevant to the change; numerical, loss, or sampling changes need analytic tests and the CPU test suite.
