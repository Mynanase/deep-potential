# Ideas

Abbreviations:

IR=intent relevance, N=novelty, CN=confidence of novelty, SV=scientific value, PL=planning, WD=well-definedness, T=total.

## `domain_idea_001`

Truth-Calibrated Static Deep Potential Trust Regions for Auriga Halo Mocks

| Round | IR | N | CN | SV | PL | WD | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 11 | 12 | 12 | 13 | 12 | 84 |
| 2 | 25 | 12 | 12 | 13 | 14 | 14 | 90 |
| 3 | 25 | 12 | 13 | 13 | 14 | 14 | 91 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## `domain_idea_002`

Auditable Auriga Force Truth and Conformal Staticity Calibration for JAX Deep Potential

| Round | IR | N | CN | SV | PL | WD | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23 | 11 | 11 | 12 | 13 | 12 | 82 |
| 2 | 24 | 12 | 12 | 13 | 14 | 13 | 88 |
| 3 | 25 | 13 | 12 | 14 | 14 | 14 | 92 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## `domain_idea_003`

Auriga Force-Truth Safe-Use Calibration for Static JAX Deep Potential

| Round | IR | N | CN | SV | PL | WD | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23 | 11 | 12 | 12 | 13 | 12 | 83 |
| 2 | 24 | 12 | 12 | 13 | 14 | 13 | 88 |
| 3 | 25 | 12 | 12 | 13 | 14 | 14 | 90 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## `domain_idea_004`

Conformal Staticity Gate for Auriga Deep Potential

| Round | IR | N | CN | SV | PL | WD | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 11 | 12 | 12 | 13 | 12 | 84 |
| 2 | 25 | 12 | 13 | 13 | 14 | 13 | 90 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## `domain_idea_005`

Truth-Calibrated Static-Use Veto Maps for JAX Deep Potential on Auriga Halo Mocks

| Round | IR | N | CN | SV | PL | WD | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 11 | 12 | 12 | 13 | 12 | 84 |
| 2 | 25 | 11 | 13 | 12 | 14 | 14 | 89 |
| 3 | 25 | 11 | 13 | 12 | 15 | 14 | 90 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

# Appendix: Idea Details

### 1. Auditable Auriga Force Truth and Conformal Staticity Calibration for JAX Deep Potential

- Loop: `domain_idea_002`
- Selected round: `3`
- Proposer output: `loops/domain_idea_002/rounds/round_003/proposer_outputs/proposer_001.json`
- Review output: `loops/domain_idea_002/rounds/round_003/reviews/reviewer_001.json`

#### Referee Marks by Round

| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| domain_idea_002 | 1 | 82 | 23 | 11 | 11 | 12 | 13 | 12 |
| domain_idea_002 | 2 | 88 | 24 | 12 | 12 | 13 | 14 | 13 |
| domain_idea_002 | 3 | 92 | 25 | 13 | 12 | 14 | 14 | 14 |
| domain_idea_002 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| domain_idea_002 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### Full Idea Verbatim

Title: Auditable Auriga Force Truth and Conformal Staticity Calibration for JAX Deep Potential

Idea Summary: Build an Auriga staticity transfer function for the current JAX Deep Potential code: a reproducible simulator-force truth layer plus a split-conformal rule that turns internal diagnostics into an accepted-volume mask with a 90% force-error upper bound. The deliverable is not just a recovered $\Phi(\mathbf{x})$, but a calibrated statement of where static CBE recovery is reliable on Auriga halo mocks.

Calculation Plan: First calculation: add a reproducible force-truth layer to the existing Auriga pipeline, then fit a one-dimensional split-conformal accepted-volume rule. Use the canonical local contract already present in `dpjax.datasets.auriga`: HDF5 schema `dpjax.auriga.mock.v1`, dataset `/eta` with columns `[x,y,z,vx,vy,vz]`, optional row-aligned $/particle_{id}$, `/mass`, `/potential`, `/acceleration`, $/source_{index}$, and attrs $length_{unit}$, $velocity_{unit}$, $potential_{unit}$, $acceleration_{unit}$. The preparation step should produce $data/auriga/halo{5,12,23}_static_mock.h5$ through $experiments.prepare_{auriga}$, but with `/acceleration` filled by a separately auditable force-truth calculation. For a query point $\mathbf{q}$ in the centered, disk-aligned physical frame, compute $$\mathbf{a}_{\rm true}(\mathbf{q})=-G\sum_{j\in\mathcal{S}} m_j\frac{\mathbf{q}-\mathbf{x}_j}{\left(\|\mathbf{q}-\mathbf{x}_j\|^2+\epsilon_j^2\right)^{3/2}},$$ with $G=4.30091\times10^{-6}\,{\rm kpc}\,(\mathrm{km/s})^2\,M_\odot^{-1}$, positions in kpc, masses in $M_\odot$, and acceleration in $(\mathrm{km/s})^2/{\rm kpc}$. The source set $\mathcal{S}$ is all documented snapshot particles/cells used for the total gravitational field, normally `PartType0..5` inside a fixed source radius such as 300 kpc around the main halo, with masses read from `Masses` or `Header/MassTable`; all chunks, type choices, unit conversion factors, softening values, tree opening angle, and self-exclusion rule must be written to $force_truth_manifest.json$. For tracer-particle queries, omit the same particle when $particle_{id}$ matches; for grid points, omit nothing. Use a Barnes-Hut or FMM tree for production, and direct-sum 1,000 random query points as an audit. Also compute $\Phi_{\rm tree}=-G\sum_j m_j/\sqrt{\|\mathbf{q}-\mathbf{x}_j\|^2+\epsilon_j^2}$ and compare it with public scalar `Potential` after unit conversion and additive-offset alignment only; scalar `Potential` is a sign/unit audit, not vector force truth. Then train static Deep Potential with $configs/df_halo12_ffjord_v22.yaml$ plus $configs/phi_halo12_static_v1.yaml$ when no-transform FFJORD is stable; if the power-transform `v21` path is needed, first implement and test the nonlinear chain rule currently blocked by $require_physics_compatible_transform$. Evaluate with $experiments.eval_auriga_truth$, using its $auriga_truth_predictions.npz$ fields $row_{index}$, $source_{index}$, `position`, $predicted_{acceleration}$, and $truth_{acceleration}$ as the base truth table. Define cells before seeing errors: radial bins `[5,8,12,16,22,30,40,50]` kpc, polar bins $[0,10,20,30,40,55,90]^\circ$ mirrored north/south, and 8 azimuth wedges. For each cell $c$, target $$E_c=\mathrm{median}_{g\in c}\frac{\|\hat{\mathbf a}_g-\mathbf a_{{\rm true},g}\|}{\|\mathbf a_{{\rm true},g}\|+\epsilon_a}.$$ Fit acceptance using only non-truth features recomputed at the same $row_{index}$: residual $R_c$, ensemble force scatter $U_c$, and ParticleID-parity split discrepancy $D_c$. Define $z_X(c)=[\log(X_c+\epsilon)-\mathrm{median}_{\mathcal F}\log(X+\epsilon)]/[1.4826\,\mathrm{MAD}_{\mathcal F}\log(X+\epsilon)+\epsilon]$ for $X\in\{R,U,D\}$ and $T_c=\max(z_R,z_U,z_D)$. On fit cells $\mathcal F$, fit a monotone one-dimensional base predictor $\hat m(T)$ for $E$ by isotonic pinball loss at $\tau=0.8$. On calibration cells $\mathcal C$, compute $s_c=E_c-\hat m(T_c)$ and $$q_{0.9}=\max\left(0,\operatorname{kth}_{k}\{s_c:c\in\mathcal C\}\right),\quad k=\left\lceil0.9(|\mathcal C|+1)\right\rceil.$$ The conformal upper bound is $U_{0.9}(c)=\hat m(T_c)+q_{0.9}$, and a cell is accepted if $U_{0.9}(c)<0.15$ and it passes hard masks on finite values, $n_{\rm eff}$, score explosions, and negative-density fraction. Calibrate on Auriga 5+23 and test on Auriga 12; a halo-12-only radial/azimuth block split is only a pilot.

### 2. Truth-Calibrated Static Deep Potential Trust Regions for Auriga Halo Mocks

- Loop: `domain_idea_001`
- Selected round: `3`
- Proposer output: `loops/domain_idea_001/rounds/round_003/proposer_outputs/proposer_001.json`
- Review output: `loops/domain_idea_001/rounds/round_003/reviews/reviewer_001.json`

#### Referee Marks by Round

| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| domain_idea_001 | 1 | 84 | 24 | 11 | 12 | 12 | 13 | 12 |
| domain_idea_001 | 2 | 90 | 25 | 12 | 12 | 13 | 14 | 14 |
| domain_idea_001 | 3 | 91 | 25 | 12 | 13 | 13 | 14 | 14 |
| domain_idea_001 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| domain_idea_001 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### Full Idea Verbatim

Title: Truth-Calibrated Static Deep Potential Trust Regions for Auriga Halo Mocks

Idea Summary: Create a truth-calibrated safe-use atlas for static Deep Potential on Auriga halo mocks. The scientific product is $\hat Φ(\mathbf x)$, $\hat{\mathbf a}(\mathbf x)$, and a data-facing trust mask $\mathcal T(\mathbf x)$ that predicts where static force recovery should meet a pre-registered error threshold. The proposal is not a new pattern-speed or generic Deep Potential solver; it is a validation protocol that makes the current JAX pipeline honest about where static equilibrium, DF derivatives, preprocessing, and selection allow reliable Auriga force recovery.

Calculation Plan: First calculation is a pre-registered Auriga truth-gated run, not an open-ended diagnostic search. Truth-source gate: if the current local $data/halo_12_train.h5$ can be paired with $/truth/acc_{true}$ or with local snapshot particles sufficient to recompute accelerations from all matter, use $halo_{id}=12$; otherwise switch the first truth-calibrated run to a public AuriGaia halo with force/potential grids, preferably Au 6, and keep Halo12 as an unscored engineering pilot. The required HDF5 contract is $/tracers/eta[N,6]=[x,y,z,vx,vy,vz]$, $/tracers/selection_{prob}[N]$ optional, $/tracers/subhalo_{id}[N]$ and disk/origin flags for sample construction only, $/tracers/tracer_{weight}[N]$ optional, $/truth/eval_{xyz}[M,3]$, $/truth/acc_{true}[M,3]$, optional $/truth/phi_{true}[M]$ and $/truth/rho_{true}[M]$, and $/metadata/{halo_{id},snapshot,redshift,center,bulk_{velocity},units,truth_{source},selection_{name}}$. Coordinates are centered on the main-halo potential minimum or `SubhaloPos`; velocities subtract the main-subhalo bulk velocity; positions are kpc, velocities km/s, accelerations $(\mathrm{km}/\mathrm{s})^2/\mathrm{kpc}$, and potentials $(\mathrm{km}/\mathrm{s})^2$. Tracers are diffuse stellar-halo particles or mock stars with $5<r<50\,\mathrm{kpc}$; bound satellites and disk particles are removed before training but never used as trust-model features. Code coordinate path: keep raw physical $η=(\mathbf x,\mathbf v)$ alongside DF coordinates. For DF preprocessing $y=T(η)$ and standardization $z=(y-μ)/σ$, compute the physical score used in the CBE as $$s_η=∇_η\ln f(η)=J_{z\leftarrow η}^{T}∇_z\ln q_\varphi(z)+∇_η\ln |\det J_{z\leftarrow η}|.$$ For elementwise transforms, $$s_{η_j}=T'_j(η_j)s_{z_j}/σ_j+T''_j(η_j)/T'_j(η_j).$$ Use `asinh` or $ε$-smoothed signed-power transforms to avoid singular derivatives at zero. Phi coordinate path: decouple the potential from DF preprocessing. The potential network consumes only linearly standardized physical positions $u=(\mathbf x-x_{Φ,0})/s_{Φ}$ saved as $phi_{normalizer}.npz$; it does not consume nonlinear DF-transformed coordinates. Physical acceleration is $$\hat{\mathbf a}(\mathbf x)=-∇_\mathbf{x}Φ_θ=-\operatorname{diag}(s_{Φ}^{-1})∇_uΦ_θ(u).$$ CBE residual is computed in physical units as $$r(η)=\mathbf v\cdot∇_\mathbf{x}\ln f-∇_\mathbf{x}Φ\cdot∇_\mathbf{v}\ln f,$$ with inverse-timescale diagnostic $τ_{\mathrm{CBE}}^{-1}=|r|$; convert using $1\,\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{kpc}^{-1}\simeq1.0227\times10^{-3}\,\mathrm{Myr}^{-1}$. Aggregate $τ_{\mathrm{CBE}}^{-1}$ per spatial cell from held-out tracers in that cell using median and p90. Pre-registration for the first scored run: seeds `[11,23,37,41]`; radial shells `[5,10,20,35,50]` kpc crossed with equal-area angular sectors, adaptively merged to at least 2048 tracers and 128 truth grid points; train trust-rule cells from azimuth sectors with index $k mod 4 in {0,1}$, validation from $k mod 4 == 2$, test from $k mod 4 == 3$; train masks are complete plus radial incompleteness plus one smooth angular cap; held-out masks are stripe/dust-like angular masks and one substructure-correlated censoring mask. Train FFJORD DF variants using v21-style transform, smoothed-asinh transform, and v22 no-transform; freeze each DF; train static $Φ_θ$ with the repo’s stable MSE-first Phi recipe. Labels for calibration are truth errors $e_a=\|\hat{\mathbf a}-\mathbf a_{\mathrm{true}}\|/(\|\mathbf a_{\mathrm{true}}\|+ε)$ plus radial and tangential components. Inputs $D_{\mathrm{obs}}$ to the trust rule are truth-free only: tracer count, radius, validation NLL gap, score p99/max, CBE residual timescale summaries, seed/bootstrap force scatter, north/south and azimuthal split disagreement, transform disagreement, and flow-vs-observed profile mismatch. Fit only a simple logistic or isotonic model for $P(e_a<0.1\mid D_{\mathrm{obs}})$ and report it against baselines: no mask, count/radius-only mask, CBE-only mask, seed-scatter-only mask, transform-disagreement-only mask, v21 versus v22, and optional local CBE inversion on the same cells. Profile mismatch is estimated by drawing inverse-preprocessed flow samples, applying the same observed mask/window, and comparing binned $ρ(r)$, $σ_v(r)$, and $β(r)$ against the observed masked tracer sample; no analytic marginalization or truth density is used in $D_{\mathrm{obs}}$.

### 3. Conformal Staticity Gate for Auriga Deep Potential

- Loop: `domain_idea_004`
- Selected round: `2`
- Proposer output: `loops/domain_idea_004/rounds/round_002/proposer_outputs/proposer_001.json`
- Review output: `loops/domain_idea_004/rounds/round_002/reviews/reviewer_001.json`

#### Referee Marks by Round

| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| domain_idea_004 | 1 | 84 | 24 | 11 | 12 | 12 | 13 | 12 |
| domain_idea_004 | 2 | 90 | 25 | 12 | 13 | 13 | 14 | 13 |
| domain_idea_004 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| domain_idea_004 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| domain_idea_004 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### Full Idea Verbatim

Title: Conformal Staticity Gate for Auriga Deep Potential

Idea Summary: Revise $idea_{004}$ into one concrete experiment: an Auriga truth-calibrated staticity gate for JAX Deep Potential. The method outputs $\hat Φ(x)$, $\hat a(x)=-\nabla Φ$, and $\hat ρ=\nabla^2Φ/(4πG)$, but the paper-level product is the calibrated rule deciding where static recovery is trustworthy. The gate is trained on Auriga truth grid cells using internal diagnostics: CBE residuals, seed/bootstrap scatter, tracer-split disagreement, negative-density fraction, local sample count, selection effective sample size, and a local CBE condition number. Primary gate model: fit a ridge/Huber linear predictor for $\log e_g$, where $e_g=\|\hat a_g-a_{true,g}\|/\|a_{true,g}\|$, then conformalize it on a calibration split. A cell is accepted only if $$\exp(\hat m(q_g)+Q_{0.9})<τ,$$ with $τ=0.1$ for the main force-error target. Add a cheap local baseline on the same cells: for nearby tracers $j$ around grid cell $g$, solve $$M_g a_g^{loc}=-b_g,\quad M_g=\sum_j w_{gj}s_{v,j}s_{v,j}^T+λI,\quad b_g=\sum_jw_{gj}s_{v,j}(v_j\cdot s_{x,j}),$$ and define $κ_g=κ(M_g)$. If local inversion succeeds where global Deep Potential fails, the issue is likely the neural $Φ$ layer or regularization; if both fail, the issue is DF score quality, selection, or halo disequilibrium.

Calculation Plan: First calculation: use raw Auriga $halo_{12}$ star particles at z=0 as the primary tracer set, not AuriGaia or upsampled stars, so the initial result isolates static CBE recovery before survey forward-modeling. Build $data/auriga_halo12_static.h5$ with `/eta` as `(N,6)` float32 in halo-centered, disk-aligned physical coordinates $[x,y,z,v_{x},v_{y},v_{z}]$, using `PartType4` particles with positive stellar formation time as tracers and cuts such as $5 < r < 70 kpc$, $is_bound_main=true$, plus optional substructure masks. Store truth separately in the same file: $/truth/grid_{x}$, $/truth/a_{total}$, $/truth/a_by_component$, $/truth/phi_{optional}$, $/truth/rho_{optional}$, component labels, softening values, halo center, velocity center, rotation matrix, and unit attrs. Compute truth accelerations from snapshot particles/cells, centered on the central Subfind subhalo: $$a_{true}(x_g)=-G\sum_{p\in\{gas,DM,stars,BH\}}m_p\frac{x_g-r_p}{(|x_g-r_p|^2+\epsilon_p^2)^{3/2}},$$ with gas `PartType0`, high-resolution DM `PartType1`, all massive `PartType4`, and BH `PartType5`; use `MassTable[1]` for high-resolution DM and `Masses` for variable-mass components. Exclude low-resolution `PartType2/3` from the nominal inner-halo force but record their contribution as a systematic check. Use Auriga units from the data specs: coordinates are converted from comoving Mpc/h to physical kpc at z=0, velocities to km/s, masses to $M_{sun}$, and $G=4.30091\times10^{-6}\,kpc\,(km/s)^2\,M_\odot^{-1}$. The first repository-facing implementation is then: train v22 no-transform DF/Phi, implement transform-aware CBE so v21 power-transform checkpoints become physically valid, evaluate global Deep Potential force errors, evaluate a cheap local CBE inversion baseline, and train a conformal staticity gate on held-out truth grid cells.

### 4. Auriga Force-Truth Safe-Use Calibration for Static JAX Deep Potential

- Loop: `domain_idea_003`
- Selected round: `3`
- Proposer output: `loops/domain_idea_003/rounds/round_003/proposer_outputs/proposer_001.json`
- Review output: `loops/domain_idea_003/rounds/round_003/reviews/reviewer_001.json`

#### Referee Marks by Round

| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| domain_idea_003 | 1 | 83 | 23 | 11 | 12 | 12 | 13 | 12 |
| domain_idea_003 | 2 | 88 | 24 | 12 | 12 | 13 | 14 | 13 |
| domain_idea_003 | 3 | 90 | 25 | 12 | 12 | 13 | 14 | 14 |
| domain_idea_003 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| domain_idea_003 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### Full Idea Verbatim

Title: Auriga Force-Truth Safe-Use Calibration for Static JAX Deep Potential

Idea Summary: A single concrete research idea: produce an Auriga-calibrated safe-use map for static JAX Deep Potential. The paper-level output is not “we recovered $Φ$ on Auriga,” but a validated rule for where the static force estimate can be trusted: $q_{0.1}(d_c)=P(e_{a,c}<0.1\mid d_c)$ from diagnostics available without truth. This makes the current `experiment/baseline-b` code directly useful for reliable static-potential recovery from Auriga halo 6D snapshots.

Calculation Plan: Use the existing Auriga integration as the execution spine, not a new parallel format. Start from `dpjax/datasets/auriga.py`, $experiments/prepare_{auriga}.py$, $experiments/eval_auriga_truth.py$, `dpjax/evaluation.py`, and $configs/phi_halo12_static_v1.yaml$. The first Φ run should be the existing robust baseline, because that is the branch’s halo static config and includes the negative-density penalty: $python -m experiments.train_{phi} --config configs/phi_halo12_static_v1.yaml --data data/auriga/halo12_static_mock.h5 --df-run-dir runs/halo_{12}/df_ffjord_v22 --run-dir runs/halo_{12}/phi_static_v1$. Immediately add a matched MSE ablation, identical except $train.loss_{type}: mse$, so the MSE-vs-robust decision is empirical and judged by held-out force truth, not CBE residual alone. Extend the canonical Auriga HDF5 contract with an optional $/truth_{grid}$ group rather than replacing it: `position[M,3]`, `acceleration[M,3]`, optional `potential[M]`, $cell_{id}[M]$, $radial_{edges}$, $mu_{edges}$, $phi_{edges}$, plus attrs for $force_{source}$, $force_{sign}$, $acceleration_scale_to_(km/s)^2/kpc$, `G`, $softening_{policy}$, $components_{included}$, $periodic_boundary_policy$, $lowres_boundary_policy$, $center_acceleration_subtracted$, and $self_field_policy$. The tessellation is deterministic and dependency-light: radial edges `[5,8,12,16,22,30,40,50]` kpc, 12 equal-area bins in $\mu=\cos θ$, and 24 uniform bins in $φ$, with grid points at each cell’s volume midpoint plus optional random intra-cell validation points. Define $e_a=\|\mathbf{a}_θ-\mathbf{a}_{true}\|/\max(\|\mathbf{a}_{true}\|,a_{floor})$, with $a_{floor}=0.01\,\mathrm{median}_{cal}\|\mathbf{a}_{true}\|$ recorded in calibration metadata. Truth priority: native snapshot acceleration if present and documented; otherwise recompute a frozen force field over DM+stars+gas+BH using the same centering, softening, component inclusion, boundary/periodic convention, and subtract the center acceleration to compare in the Galactocentric non-inertial frame. Add `dpjax.evaluation` functions for cell metrics and calibration, and extend $experiments/eval_auriga_truth.py$ to emit per-cell diagnostics and a safe-use table. Fit a simple L2-logistic unsafe/safe classifier $q_ε(d_c)=P(e_{a,c}<ε\mid d_c)$ with SciPy/NumPy; class weights are allowed only during classifier fitting, then a separate unweighted calibration/reliability step must recalibrate probabilities on held-out cells.

### 5. Truth-Calibrated Static-Use Veto Maps for JAX Deep Potential on Auriga Halo Mocks

- Loop: `domain_idea_005`
- Selected round: `3`
- Proposer output: `loops/domain_idea_005/rounds/round_003/proposer_outputs/proposer_001.json`
- Review output: `loops/domain_idea_005/rounds/round_003/reviews/reviewer_001.json`

#### Referee Marks by Round

| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| domain_idea_005 | 1 | 84 | 24 | 11 | 12 | 12 | 13 | 12 |
| domain_idea_005 | 2 | 89 | 25 | 11 | 13 | 12 | 14 | 14 |
| domain_idea_005 | 3 | 90 | 25 | 11 | 13 | 12 | 15 | 14 |
| domain_idea_005 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| domain_idea_005 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### Full Idea Verbatim

Title: Truth-Calibrated Static-Use Veto Maps for JAX Deep Potential on Auriga Halo Mocks

Idea Summary: One concrete idea: make JAX Deep Potential on Auriga a truth-calibrated static-use instrument. The deliverable is a recovered static force field plus a calibrated veto map saying where that field can be trusted without looking at truth. Auriga truth is used for calibration labels only; the deployable score uses CBE residuals, ensemble scatter, split-sample disagreement, score conditioning, selection/support diagnostics, and edge distance. This is scientifically distinct from rerunning Deep Potential, adding generic ensembles, or adopting conditional selection: it asks where static CBE recovery is valid in a realistic halo and gives an operational accept/reject rule for future runs.

Calculation Plan: First calculation: build a mandatory Auriga evaluation-support data contract and a low-capacity calibrated veto for one static $halo_{12}$ run. The repository-facing training dataset remains `eta` with shape `(N,6)` in physical Galactocentric order `[x,y,z,vx,vy,vz]`, but the truth file must also contain $x_{eval}$ `(M,3)` and $acc_true_eval$ `(M,3)`. For the first run, $x_{eval}$ is a held-out set of tracer positions, stratified over radial-angular cells and never used in DF/Φ training; arbitrary grid points are allowed later only if the same all-component force solver can label them. Store required metadata: snapshot id, halo center, halo bulk velocity, units, softening model, $truth_{method}$, train/eval particle IDs, and mask definitions. Use z≈0 Auriga $halo_{12}$, centered on the central SUBFIND potential minimum or most-bound particle, with velocities subtracting the central subhalo bulk velocity. The first radial volume is $15\le r_{\rm GC}\le50\,\mathrm{kpc}$ to avoid the disk/bar and outer low-support edge. Define $mask_{halo}$ as PartType4 star particles in the host FoF group within this radial volume, selected as accreted-halo stars when merger-tree/accretion labels exist; otherwise use a recorded fallback cut $|z|>5\,\mathrm{kpc}$ or circularity $|J_z/J_c(E)|<0.7$. Define $mask_{substructure}$ by removing currently bound noncentral SUBFIND subhalos and, when accretion fields exist, recent coherent debris with time since infall <4 Gyr. Truth acceleration is read from the snapshot particle acceleration field if available; otherwise compute it with an all-component tree force using Auriga/AREPO softenings. Reject the truth product unless a recomputed validation subset has <3% median and <5% p95 force-label discrepancy, below the 10% recovery threshold. Train DF/Φ with the current JAX scripts and static CBE residual

$$\mathcal R_θ(η)=\mathbf v\cdot\nabla_x\log f(η)-\nabla Φ_θ(\mathbf x)\cdot\nabla_v\log f(η),\qquad \hat{\mathbf a}_θ=-\nabla Φ_θ.$$

Resolve the current transform blocker explicitly. Preferred path: add $score_physical_apply$ in `dpjax/flows/api.py`, $CoordinateTransform.jac_{diag}$ and $CoordinateTransform.grad_log_abs_det_jac$ in `dpjax/data.py`, and a physical-gradient wrapper for Φ so that nonlinear transforms use

$$\nabla_η\log p_η=(T'(η)/σ)\odot\nabla_z\log p_z+\nabla_η\log|\det T'(η)|.$$

Refactor $residual_{A}$ or add $residual_{physical}$ to consume physical $η$, physical scores, and physical $\nabla Φ$. Test `none`, `asinh`, `log`, and a smoothed signed-power transform; the existing unsmoothed `power` case near zero is fragile and should not be used for CBE unless the derivative is clipped or replaced by a smoothed version. Contingency path: if this implementation is not ready, run a smaller no-transform halo subset using $configs/phi_halo12_static_v1.yaml$ and v22-style DF gates: finite/decreasing validation NLL, $score_{p99}<200$, and $score_max_abs<5000$. Form adaptive cells from radial shells crossed with HEALPix-like angular sectors, merging until each cell has at least 512 training tracers and 64 held-out eval points. Label each cell by $\epsilon_a=\mathrm{median}_{x\in cell}\|\hat{\mathbf a}(x)-\mathbf a_{true}(x)\|/\|\mathbf a_{true}(x)\|$. The first veto estimator is a predeclared monotone severity score from truth-free diagnostics: residual scale, DF-seed force scatter, Φ-seed force scatter, split-mask force disagreement, velocity-score conditioning $κ_x$, DF score tails, selection-weight variance, support density, and edge distance. Fit isotonic calibration on blocked radial-angular folds to estimate $q_j=P(\epsilon_a<0.1\mid S_j)$, not a flexible classifier. Minimal baselines are required: no veto, radius/count/edge-only veto, residual-only veto, ensemble-only veto, and full veto. Minimum viable paper order: 1. Plummer regression plus one $halo_{12}$ snapshot and two masks, high payoff and medium cost. 2. Selection stress test with radial incompleteness and angular holes, high payoff and medium cost. 3. Derivative/score stability ablation with 1, 4, 8, 16 DF seeds, medium-high payoff and high cost. Defer conditional velocity and direct score estimation unless these baselines identify selection or derivative noise as the limiting failure mode. For non-spherical Auriga, report force-equivalent shell mass $M_F(r)=-r^2\langle a_r\rangle_Ω/G$ over accepted cells only, compared to the same statistic from truth acceleration; do not make noisy learned Laplacian density the primary claim.
