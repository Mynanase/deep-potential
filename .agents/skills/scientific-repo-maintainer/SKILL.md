---
name: scientific-repo-maintainer
description: Maintain or refactor a scientific Python/JAX repository when experiment workflows, diagnostics, artifacts, or repository structure must improve without silently changing the scientific model.
---

# Scientific Repository Maintainer

Keep research software reproducible and easy to operate without introducing unnecessary orchestration.

## Establish the contract

- Read the repository instructions, current configs, public entry points, output layout, and tests before proposing structural changes.
- Inspect the branch and dirty worktree. Preserve unrelated tracked edits and all local research material.
- Identify which scientific contracts are fixed: data selection, array order, units, objective, weights, model configuration, random seeds, and artifact schemas.
- Separate verified current behavior, proposed behavior, and validation still requiring the user's GPU or server.

## Choose the smallest durable layer

- Keep reusable numerical functions array-only and independent of files, plotting, and repository paths.
- Put dataset loading, configuration, checkpoints, training, evaluation, plotting, and run layout in the operational layer.
- Treat training health metrics, post-training scientific evaluation, and rendering as separate interfaces. Add a thin process-level orchestrator when users need one daily command.
- Persist expensive or shared derived diagnostics so figures can be regenerated without model loading or repeated score/Hessian computation. Do not persist cheap presentation-only transforms.
- Promote repeated diagnostics into shared artifact readers and figure builders; keep genuinely one-off truth checks in analysis scripts.

## Preserve failure and recovery semantics

- Validate configuration and prerequisites before expensive work.
- Fail fast between stages, but retain valid checkpoints and completed scientific artifacts.
- Do not infer completion from a directory alone; validate the checkpoint or artifact contents required by the next stage.
- Keep standalone commands usable for debugging and recovery. Do not add automatic retries, hidden skipping, or a workflow database without demonstrated need.

## Verify proportionally

- Test public commands, enabled/disabled branches, missing or stale artifacts, resume protection, and fail-fast ordering.
- Run focused tests while editing, then the repository's CPU test suite, lint, compile, lock, CLI-help, and diff checks.
- Render representative scientific figures when plotting changes and inspect the actual output.
- Leave real GPU or multi-host validation explicitly pending when the current environment cannot perform it.
