# ARC self-reflection

## Provenance

- Repository: current `deep-potential` workspace
- Branch: `experiment/baseline-b`
- Starting commit: `5601a2b0a301704ea0819a3e8f55e2cd388eb449`
- ARC skill: `arc 0.9.4`
- Project: `2205.02244_x_2507.03742`
- Primary run: `auriga_static_potential_20260725_025838`
- Ideas retry: `auriga_static_potential_20260725_025838_retry1`
- Domain IDs:
  - `arXiv_2205_02244_22624df3e4`
  - `arXiv_2507_03742_f8d2975367`
- Verified seeds:
  - Green et al. static method: `arXiv:2205.02244`
  - Kalda & Green update: `arXiv:2507.03742`

## What worked

- Local PDF first pages and ar5iv full text independently established the
  correct paper identifiers. This prevented the INSPIRE/domain lineage
  confusion with the earlier `2011.04673` workshop paper from changing scope.
- Separate domain builds let the 2023 static method remain the core while the
  2025 paper contributed ensemble, selection, uncertainty, and non-stationarity
  practices.
- Repository inspection materially improved the ideas prompt. Every valid ARC
  loop converged on truth-calibrated static-force reliability rather than a
  generic repeat of Deep Potential or a pattern-speed project.
- The deterministic ranking selected the best schema-valid round from each
  loop. The highest-scoring proposal was “Auditable Auriga Force Truth and
  Conformal Staticity Calibration for JAX Deep Potential” at 92/100.
- The first engineering slice was small enough to implement and test:
  canonical Auriga HDF5 data/truth schema, ParticleID/KD-tree alignment,
  source-index provenance, additive-offset-safe potential metrics, vector
  acceleration metrics, a static Halo12 Phi config, and operation-guide
  commands. The full repository test suite passed with 34 tests.

## Anomalies and response

- ARC's local PDF parser lacked `pdftotext`, so ar5iv full text was used and
  local PDFs were cross-checked separately.
- The 2023 domain candidate pack omitted the 2025 update and partly conflated
  the full paper with its workshop ancestor. The two verified papers were kept
  as separate pinned seeds.
- The first ideas run failed because sandboxed workers could not call `ps`.
  Its recovery-only artifacts were preserved and excluded. A new `_retry1`
  run was used rather than overwriting the failed audit trail.
- The retry completed, but 18 late-round outputs were major recoveries after an
  external usage limit was reached. The ranker correctly assigned them zero
  marks and selected earlier valid rounds.
- PDF export first failed on a missing `Noto Sans CJK SC` font. A retry with
  `Source Han Sans CN` was started, but status inspection and the required
  ranked-report export were blocked by the same external usage limit. No
  workaround was attempted; Markdown/JSON/HTML remain authoritative.

## Scientific judgment

The strongest route is not an immediate switch to the newest flow library. The
current uncertainty is dominated by whether a static method is valid and by the
absence of a calibrated Halo12 force truth. A new spline flow or direct score
estimator is useful only after a fixed truth protocol can show that it improves
force bias, coverage, or accepted volume.

The current `power(alpha=0.5)` preprocessing result cannot be carried into the
CBE by silently disabling the guard. Its physical score requires the transform
Jacobian and log-Jacobian derivative, and a transformed potential coordinate
path also requires correct gradient/Laplacian chain rules. The ranked ideas
suggest decoupling the Phi normalizer from nonlinear DF preprocessing as the
cleaner long-term design.

Public AuriGaia force/potential grids cover Au6, Au16, Au21, Au23, Au24, and
Au27, not Halo12. Public raw snapshots expose scalar particle `Potential` but
not a common `Acceleration` field. Therefore the next run must either locate
private Halo12 force truth, build and audit a Halo12 force grid from all mass
components, or calibrate first on a public grid halo.

## Outcome and remaining blocker

The research question, ranked direction, experiment stages, success metrics,
and first repository implementation are complete. The broader goal—successful
application of the modified Deep Potential to real Auriga Halo12 mock data—is
not yet complete because no Auriga HDF5 data, trained v22 DF checkpoint, Phi
checkpoint, or Halo12 acceleration truth is present in the accessible
workspace. The next concrete action is to place or identify those inputs and
run `experiments.prepare_auriga`.
