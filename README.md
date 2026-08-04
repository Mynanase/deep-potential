# deep-potential

JAX/Flax implementation of a two-stage method for recovering gravitational
potentials from phase-space snapshots:

1. fit a distribution function (DF) with a normalizing flow;
2. freeze the DF and fit a potential network with the collisionless Boltzmann
   equation.

Experiments use one checked-in run YAML and three stable entry points:

```bash
python -m experiments.run_df configs/runs/halo12_static_v1.yaml
python -m experiments.run_phi configs/runs/halo12_static_v1.yaml
python -m experiments.run_eval configs/runs/halo12_static_v1.yaml
```

The expensive stages are standalone processes. Post-training exploration uses
the git-friendly Marimo app at `analysis/halo12.py` and only reads saved
artifacts.

See `README_JAX.md` for setup, `PROJECT_STRUCTURE.md` for architecture, and
`docs/server_agent_run_guide.md` for the server workflow.

Historical TensorFlow/Sonnet code under `archive/legacy_tensorflow/` is not part
of the supported runtime.
