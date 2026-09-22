# AGENTS.md — dpjax / Auriga Halo12 deep-potential (Phase-2 baseline)

This file contains repository-wide non-negotiable rules.
Scientific background and the operating manual live in
`scripts/auriga/README.md`.

## Baseline and references

The accepted Phase-2 baseline is defined in `docs/phase-1-summary.md`.
Treat it as the control configuration; test changes in child experiment nodes
and do not silently alter multiple baseline decisions at once.
See `docs/branches.md` for branch, remote, tag, and archived-project provenance.

## Compute environment

- Production experiments run on host `gpu`, one GPU per run unless the runner
  explicitly supports another allocation.
- On the server always use the existing conda environment
  `/home/qiutao/miniforge3/envs/dp-jax/bin/python`. Do not create new venvs
  and do not install or upgrade packages into that environment; report
  dependency gaps before acting.
- JAX processes always set `JAX_PLATFORMS=cuda` (`JAX_PLATFORMS=cpu` for local
  tests) and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Run contract

- Use the committed runners rather than reconstructing production commands:
  `bash scripts/auriga/run_smoke_baseline.sh` for smoke validation and the
  `scripts/auriga/run_w1024_*.sh` family for production. Runners must preflight
  their inputs, run relevant tests, and verify lineage before reusing data.
- Check the exit code of each of the three training stages before continuing;
  mid-run resumption is not supported. Retraining always uses a fresh run
  directory; never mix in old checkpoints or old gradient files.
- **Strict compute/plot layering**: compute scripts persist arrays/JSON; plot
  scripts only read persisted arrays and never reload models. Generic figure
  scripts live at `scripts/` (`plot_potential_2d.py`, `plot_radial_marginals.py`,
  `plot_enclosed_mass.py`): units from `--input` attrs, figures to `--fig-dir`
  (png), styling on `auriga/orx_figstyle.py`. Retired scripts live in
  `scripts/auriga/archive/` and are wired into nothing (README maps each to
  its replacement).
- Standard acceptance after every experiment round = the particle-truth
  adjudication suite: `validate_enclosed_mass.py` (compute) →
  `plot_pt_adjudication.py` / `plot_shell_error_pairs.py` /
  `plot_pt_generations.py` (figures). Truth-side preprocessing goes through
  `auriga/truth_products.py` (build-grids / build-shell-mass /
  build-radial-hists): lineage-cached and idempotent — a lineage mismatch is
  an error, never an overwrite.
- Never add `--potential-ignore-nobs` to Halo12 runs (it drops the spatial
  density gradient and breaks the full steady-state equation) or
  `--basic-potential-benchmarking-gaia-units` (wrong unit system).
- Experiments are always launched through the project's experiment
  orchestration; do not start long tasks on the server by hand. Run snapshots
  contain committed content only — commit changes first.

## Code style

Follow the upstream (Green/Kalda) style first; the preferences below are
compatible with it and settle the remaining choices.

- Active Python code is linear, imperative research-script style. A stage
  entry is one main function plus its CLI, top to bottom: config, data,
  compute, output. Stage dispatch is plain `if args.x:` blocks, never a
  framework. Full argparse with per-argument help is fine for stage entries
  (`fit_all.py` is the reference); auxiliary scripts keep the CLI short.
- Keep simple calls, variable lists, array selections, merges, renames and
  file reads on one line; break only when clearly too long, multi-layered,
  or hard to read. Long lines are acceptable (upstream routinely exceeds
  88 chars); do not wrap just to satisfy a column limit.
- Prefer direct statements and intermediate variables with physical
  meaning (`benchmarking_r0`, `cylindrical_origin`). Small intuitive
  duplication is allowed (nested `moving_average` in `utils.plot_loss`).
  Do NOT introduce registries, factories, thin wrappers, or new abstraction
  layers to save a few lines.
- DO keep: JAX derivatives/jit, model definitions, batching machinery, and
  numerical/helper functions with genuine reuse (`load_flow`,
  `phi_direct`, `sample_conditional`, `bootstrap_ci`). Large explicit
  parameter lists are fine (upstream passes 14-17).

## Random seeds

- Four seed layers are fixed: data shuffle `--seed 0` (`prepare_data.py`,
  recorded in attrs `shuffle_seed`); DF flow `df.seed=0`; score sampling
  `flow_sampling.seed=1`; potential network `Phi.seed=2`.
- Do not pass `--seed` to `fit_all.py`: it overrides all three training seeds
  at once and breaks the layering above.
- A fixed experiment records the input HDF5, run/options.json, source commit
  hash, and model files together.

## Data

- Training data is frozen: `halo12-clean-smooth.h5` (server DATA_ROOT; seed 0,
  mass weighting, lineage attrs complete). `prepare_data.py` refuses to
  overwrite an existing output; new data gets a new file name; never feed an
  already dimensionless-ized h5 back to the converter; inputs must carry
  kpc / km/s metadata.
- Do not commit generated datasets or run artifacts except explicitly
  designated small fixtures and truth products.
- Default is mass weighting `m/mean(m)`; `--weighting number` is a different
  scientific target — never mix or compare results across the two.

## Physics and units

- Dimensionless coordinates `q = x/L, p = v/V` with L = 10 kpc, V = 100 km/s;
  the saved `dlnf_deta` are derivatives with respect to the input `(q, p)`;
  `Phi_phys = V²φ`.
- The density ρ derived from the potential is the total gravitational source
  density, not a stellar-mass histogram; axis profiles are not spherical-shell
  averages.
- `eta.attrs` `r_in/r_out` delimit the Phi training domain (1 ≤ r ≤ 70 kpc);
  they do not truncate the DF.
- The negative-density penalty `lambda_` only suppresses negative Laplacians
  and does not guarantee a smooth positive density; keep negative densities
  and local fluctuations visible — never clip to zero to hide problems.
  Simulated true potentials / accelerations never enter training outputs.

## Git and the experiment tree

- Experiment branches are prefixed `orx/*`: once a node has a run producing
  results it is frozen — no rewriting; later changes open child nodes. Never
  delete branches; "deletable" markers require manual confirmation.
- Do not push to upstream; `origin` is for visibility only. Use
  `docs/branches.md` for remotes, tags, and archived-project lookup.
- Commit message style: `feat(phi)` / `fix(runner)` / `eval(lambda)` /
  `fig(...)` / `docs:` / `bench:` / `test(phi)` / `chore:`, body states
  motivation and the evidence run.
- Before committing, run applicable lightweight checks: `bash -n` for shell,
  `python -m py_compile` for Python, and
  `env JAX_PLATFORMS=cpu <python> -m pytest tests/ -q` for the CPU test suite.
- The full CPU test suite must pass. Skips are acceptable only for tests that
  explicitly require the uncommitted particle-truth HDF5.
- New loss, sampling, and audit behavior must include analytic tests. Tests
  validate machinery such as algebra, quadrature, and units, not checkpoints.
- Artifacts go to `runs/` (ignored): code is committed, artifacts are not.
  `.note/` holds private research notes — never commit it, never disclose its
  contents.
