# AGENTS.md — dpjax / Auriga Halo12 deep-potential (Phase-2 baseline)

Applies to all coding and run agents working in this repository (Codex, Claude
Code, etc.). This file continues the AGENTS.md of the
`codex/upstream-sync-2026-09-12` branch (0a23194, 2026-09-17), revised for the
phase-2 baseline. For scientific background, unit derivations, and the full
operating manual see `scripts/auriga/README.md`. This file only fixes the
non-negotiable conventions; on conflict, the user's current instruction wins.

## Background and document index

The main line is adjudicated (phase-1, six rounds, 2026-09-17 → 09-22):
**frozen clean+smooth data (n=1,619,615) + S1 stratified sampling +
grid-decoupled negative-density prior + innerA radius-balanced grid +
λ=10**. Decision/evidence/branch mapping: `docs/phase-1-summary.md`; fix and
figure ledgers: `docs/phase2-premerge-survey.md`; branch landscape:
`docs/branches.md`; long-term kept scripts: `scripts/auriga/keep/` (README
records provenance; not yet wired into the main pipeline).

## Compute environment

- Experimental compute runs on the ssh host `gpu` (8×A100-40GB, driver 570 /
  CUDA 12.8), single GPU `CUDA_VISIBLE_DEVICES=0` (session convention: one run
  per GPU; runners allow external override).
- On the server always use the existing conda environment
  `/home/qiutao/miniforge3/envs/dp-jax/bin/python`. Do not create new venvs
  and do not install or upgrade packages into that environment; report
  dependency gaps before acting.
- For local CPU verification use
  `/Users/qttao/Documents/Research/flow-diff/myanase-deep-potential/.venv-halo/bin/python`;
  ⚠️ the local conda `dp-jax` env lacks `e3nn_jax` and cannot run this repo.
- `scripts/auriga/requirements.txt` is the pinned version snapshot, used only
  when an environment rebuild has been explicitly decided.
- JAX processes always set `JAX_PLATFORMS=cuda` (`JAX_PLATFORMS=cpu` for local
  tests) and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Run contract

- Fixed smoke-baseline entry point: from the repo root,
  `bash scripts/auriga/run_smoke_baseline.sh` (three independent processes
  `--flow-training` → `--flow-sampling` → `--potential-training`, then
  `plot_potential.py` + `smoke_summary.py`).
- Production training uses the `scripts/auriga/run_w1024_*.sh` family.
  Template: `set -eo pipefail` + PREFLIGHT (interpreter / data / dependency /
  options assertions) + run the relevant unit tests before training; idempotent
  — if the target dataset exists, verify lineage first, then reuse.
- Check the exit code of each of the three training stages before continuing;
  mid-run resumption is not supported. Retraining always uses a fresh run
  directory; never mix in old checkpoints or old gradient files.
- `fit_all.py` reads parameters from `<run-dir>/options.json`: copy the
  options into the run directory before starting (smoke uses
  `scripts/auriga/options-smoke.json`, production starting point `options.json`).
- **Strict compute/plot layering**: compute scripts persist arrays/JSON; plot
  scripts only read persisted arrays and never reload models. Axis ranges use
  hand-tuned `set_ylim` constants with the data range noted in a comment
  (2026-09-15 convention).
- Standard acceptance after every experiment round = the particle-truth
  adjudication suite: `validate_enclosed_mass.py` (compute) →
  `plot_pt_adjudication.py` / `plot_shell_error_pairs.py` /
  `plot_pt_generations.py` (figures).
- Never add `--potential-ignore-nobs` to Halo12 runs (it drops the spatial
  density gradient and breaks the full steady-state equation) or
  `--basic-potential-benchmarking-gaia-units` (wrong unit system).
- Experiments are always launched through the project's experiment
  orchestration; do not start long tasks on the server by hand. Run snapshots
  contain committed content only — commit changes first.

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
- Only two data files are committed: `data/auriga/halo12-smoke.h5` (512-particle
  smoke) and `data/auriga/halo12_particle_truth_grids.h5` (particle-truth
  grids); the rest of `data/` is ignored.
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
- Do not push to upstream; `origin` is for visibility only. Retrospection:
  the `phase1-archive` remote (local old repo `/Research/dpjax`, all frozen
  phase-1 branches), the five `phase1/*` tags, and run logs via
  `orx logs <runId>` (old project `07d8ee01`, permanently queryable).
- Commit message style: `feat(phi)` / `fix(runner)` / `eval(lambda)` /
  `fig(...)` / `docs:` / `bench:` / `test(phi)` / `chore:`, body states
  motivation and the evidence run.
- Lightweight pre-commit checks: `bash -n` (shell), `python -m py_compile`,
  `env JAX_PLATFORMS=cpu <python> -m pytest tests/ -q` (expect 46 passed,
  3 skipped; the skips are due to the truth HDF5 not being local). Tests
  validate the machinery (algebra / quadrature / units on analytic examples),
  not checkpoints; new loss/sampling/audit logic must come with tests in the
  same style, honoring the `_unpack_phi_batch` 4/5/6/7-tuple batch contract.
- Artifacts go to `runs/` (ignored): code is committed, artifacts are not.
  `.note/` holds private research notes — never commit it, never disclose its
  contents.

## Phase-2 round-1 handover (in progress)

Oscillation-suppression fixes are ready and await cherry-pick (sources on
`phase1-archive`): osc-pair `7de9703` + `aa7b695` + `3d07384`, spectral-norm
`002dcdd`, cosine-anneal `51fd27e`; ⚠️ the osc-pair weight was calibrated in
the λ=1 context and must be recalibrated for the λ=10 baseline. Update this
section once landed.
