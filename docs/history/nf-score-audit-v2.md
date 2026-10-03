# NF score audit v2: minimal execution protocol

This document supersedes the v1 protocol and task-card design. It preserves the
scientific goal and accepted evidence while removing operational overhead: no
token budgets, no repeated Git instructions, and no redundant model-behavior
rules. Existing completed nodes remain immutable.

## Goal

Determine whether outer-region NF first-order score error exists, whether it
affects force inversion, and whether it causally contributes to negative density
and high-frequency oscillation in the Phi network. Report conclusions by spatial
region, velocity coverage, physical scale, and product (score, force, density).

## Accepted state

| Stage | Evidence | Status and implication |
|---|---|---|
| T1 numerical chain | run `f7f2c973-5a1b-4330-ba72-286526b9e3d2`, commit `8948905` | Numerical gates passed 8/8 and the fixed-point cache is certified for downstream use. This certifies derivative machinery, not model correctness. |
| T2 weak Stein test | calibration run `91a17beb-ca44-48e2-98ca-970b3716c000`; final real stage run `6778ac8c-cd19-470c-9746-339740895417`, commit `c6d3457` | Machinery passed calibration. Auriga heldout evidence remains exploratory and dependence-sensitive: angular-cluster maxT p=0.2235 and radial-block p=0.1645 do not by themselves identify score bias. |
| T3 Plummer oracle | successful run `fade7ea7-d38c-4796-8450-8daa0139393e`, commit `00f7049` | Analytic oracle and local inversion machinery passed. A5 local-gradient relative error is p10/median/p90 = 0.112/0.250/0.400, so alpha-star must not be used as a stable Auriga true-force proxy. |

No stage has established a causal claim about Phi density error. The next
decision is therefore mock-based, not another Auriga significance test.

## Minimal route

1. **T4a: mock DF/NF qualification.** Train or otherwise produce a full-data
   Plummer mock NF that is compatible with the T1 score-cache interface. Verify
   data lineage, units, score/log-prob derivatives, checkpoint hashes, and the
   cache contract. Do not use a radial cut or selection variant.
2. **T4: minimum paired experiment.** Run only P00 and P11 initially:
   P00 uses analytic position and velocity scores; P11 uses NF position and
   velocity scores. Keep data, seeds, architecture, optimization, and checkpoint
   selection identical. Persist convergence trajectories and unified readouts.
3. **T5: adjudication.** Compare paired outputs on score/local-force error, Phi
   force error, negative-density volume and mass, shell mass, and fixed-scale
   spectra. Add P10/P01 only if P00 versus P11 differs and locating the score
   component is the next decision.

Do not expand to data-size sweeps, extra seeds, additional mocks, or architecture
variants until T5 identifies a specific unresolved factor.

## Minimal task-card template

```markdown
# Tn: one decision

Question:
Inputs: parent run/commit, manifest, cache or data identity
Fixed facts: units, data identity, seeds, full-data/no-r-cut rule
Allowed execution: local checks or an explicitly authorized compute scope
Forbidden changes: unrelated scientific variants or hidden environment knobs
Acceptance: gates observable in the run log
Deliverables: checkpoint/cache/manifest/report as applicable
Stop: identity mismatch, failed numerical gate, or authorization exceeded
``+

Keep stage handoffs to one short state row plus links:
`stage / run / commit / key gate / conclusion / limitation / next`.

## Git and repair contract

A run receives only the recorded immutable commit. The working tree proposes
edits, the index records the exact tree intended for the next commit, and the
commit becomes the run snapshot. Thus `git add`, `git commit`, a clean status,
and review of the recorded commit are necessary before launch.

Implementation failures before a scientific answer are repaired on the same
provisional node, with a regression check and commit when practical. Once a run
answers a node, its code history is frozen and continuation opens a child.

Git locks are normal transient mutexes and do not require per-operation user
approval. `Operation not permitted` is a permission or sandbox issue: do not
delete locks or bypass Git with temporary indexes, `commit-tree`, or hand-written
refs. Only if Git reports an existing lock, no Git process remains active, and
the lock is stale, quarantine it and retry through normal Git commands.
