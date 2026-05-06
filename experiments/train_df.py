from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import optax
import yaml

from dpjax.data import fit_normalizer, iter_batches, load_eta_h5, CoordinateTransform
from dpjax.flows.api import build_flow, init_flow, log_prob_apply, log_prob_reg_apply, score_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.utils.ckpt import create_manager, finalize, restore_latest, save

# ---------------------------------------------------------------------------
# ReduceLROnPlateau state
# ---------------------------------------------------------------------------

class _PlateauState:
    """Simple state machine for ReduceLROnPlateau + early-stopping-on-min-lr.

    After ``warmup_steps``, the learning rate is held constant until the
    validation loss fails to improve by at least *min_delta* for *patience*
    consecutive evaluations.  At that point the lr is multiplied by *factor*
    and the patience counter resets.
    Training terminates when lr drops below *min_lr*.
    """

    def __init__(
        self,
        warmup_steps: int,
        initial_lr: float,
        factor: float = 0.5,
        patience: int = 1024,
        min_delta: float = 0.01,
        min_lr: float = 5e-6,
    ):
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.best_val_loss: float = math.inf
        self.steps_without_improvement: int = 0
        self.current_lr: float = initial_lr
        self.lr_reduction_count: int = 0

    def step(self, global_step: int, val_loss: float | None) -> float:
        """Process one evaluation.  Returns the lr to use for the *next* interval.

        During warmup the caller should use a warmup schedule; after warmup
        the returned value is the (possibly reduced) constant lr.
        """
        if global_step < self.warmup_steps or val_loss is None or math.isnan(val_loss):
            return self.current_lr

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience:
            self.current_lr *= self.factor
            self.lr_reduction_count += 1
            self.steps_without_improvement = 0
            print(
                f"[train_df] ReduceLROnPlateau: reducing lr to {self.current_lr:.2e} "
                f"(reduction #{self.lr_reduction_count})"
            )

        return self.current_lr

    @property
    def should_stop(self) -> bool:
        return self.current_lr < self.min_lr


# ---------------------------------------------------------------------------
# Core training function – callable from both CLI and Jupyter
# ---------------------------------------------------------------------------

def run_df_training(
    config: Dict[str, Any],
    data_path: str | Path,
    run_dir: str | Path,
    *,
    resume: bool = False,
    logger: Optional["ExperimentLogger"] = None,
) -> Dict[str, Any]:
    """Train the DF (RealNVP / FFJORD) normalizing flow.

    Parameters
    ----------
    config : dict
        Full training configuration (typically loaded from a YAML file).
    data_path : str or Path
        Path to the HDF5 data file containing ``eta``.
    run_dir : str or Path
        Directory for checkpoints, metrics, and config snapshots.
    resume : bool
        If ``True``, resume from the latest checkpoint in *run_dir*.

    Returns
    -------
    dict
        ``{"params": ..., "normalizer": ..., "model": ..., "final_step": int}``
    """
    data_path = resolve_path(data_path)
    run_dir = ensure_dir(run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    data_cfg = config.get("data", {})

    eta = load_eta_h5(data_path, dataset=data_cfg.get("dataset", "eta"))

    # Optional coordinate preprocessing (e.g. asinh/log/power) applied BEFORE
    # standardization, so that the normalizer sees a flatter distribution.
    transform_cfg = data_cfg.get("transform")
    coord_transform = None
    if transform_cfg is not None:
        coord_transform = CoordinateTransform.fit(eta, transform_cfg)
        if coord_transform.type != "none":
            eta = coord_transform.transform(eta)
            coord_transform.save_npz(run_dir / "coord_transform.npz")
            print(f"[train_df] Applied coordinate transform {coord_transform.type!r} "
                  f"on dims={coord_transform.dims.tolist()} before standardization.")

    # Optional sigma clipping: remove outlier samples that cause score explosion.
    # Applied AFTER coordinate transform, BEFORE normalizer fitting, so that
    # the normalizer's mean/std reflect the clipped distribution.
    clip_sigma = float(data_cfg.get("clip_sigma", 0.0))
    if clip_sigma > 0.0:
        clip_mean = np.mean(eta, axis=0)
        clip_std = np.std(eta, axis=0)
        clip_std = np.maximum(clip_std, 1e-6)
        mask = np.all(np.abs(eta - clip_mean) < clip_sigma * clip_std, axis=1)
        n_before = eta.shape[0]
        eta = eta[mask]
        n_after = eta.shape[0]
        print(f"[train_df] Sigma-clip at {clip_sigma}σ: removed "
              f"{n_before - n_after}/{n_before} samples "
              f"({100.0 * (n_before - n_after) / n_before:.2f}%), "
              f"kept {n_after}.")

    normalizer = fit_normalizer(eta, eps=float(config.get("normalizer", {}).get("eps", 1.0e-6)))
    normalizer.save_npz(run_dir / "normalizer.npz")

    eta_std = normalizer.transform(eta)

    val_frac = float(data_cfg.get("val_frac", 0.1))
    val_frac = float(np.clip(val_frac, 0.0, 0.5))

    n_total = int(eta_std.shape[0])
    n_val = int(round(n_total * val_frac))
    n_val = min(max(n_val, 0), max(n_total - 1, 0))

    if n_val > 0:
        eta_train = eta_std[:-n_val]
        eta_val = eta_std[-n_val:]
    else:
        eta_train = eta_std
        eta_val = np.empty((0, eta_std.shape[1]), dtype=np.float32)

    flow_cfg = config.get("flow", {})
    model = build_flow(flow_cfg)

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8192))
    epochs = int(train_cfg.get("epochs", 32))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 200))
    max_to_keep = int(train_cfg.get("max_to_keep", 3))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    if max_batches_per_epoch is not None:
        max_batches_per_epoch = int(max_batches_per_epoch)
        if max_batches_per_epoch <= 0:
            raise ValueError("train.max_batches_per_epoch must be positive when provided.")
    stop_score_p99 = train_cfg.get("stop_score_p99", None)
    if stop_score_p99 is not None:
        stop_score_p99 = float(stop_score_p99)
        if stop_score_p99 <= 0:
            raise ValueError("train.stop_score_p99 must be positive when provided.")
    stop_score_max_abs = train_cfg.get("stop_score_max_abs", None)
    if stop_score_max_abs is not None:
        stop_score_max_abs = float(stop_score_max_abs)
        if stop_score_max_abs <= 0:
            raise ValueError("train.stop_score_max_abs must be positive when provided.")
    n_devices = int(jax.local_device_count())
    use_sharding = bool(train_cfg.get("multi_gpu", True)) and n_devices > 1

    if use_sharding:
        if batch_size < n_devices:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= number of devices ({n_devices}) for multi-GPU training."
            )
        if (batch_size % n_devices) != 0:
            adjusted_batch_size = (batch_size // n_devices) * n_devices
            if adjusted_batch_size <= 0:
                raise ValueError(
                    f"batch_size ({batch_size}) is too small for {n_devices} devices in multi-GPU training."
                )
            print(
                f"[train_df] Adjusting batch_size from {batch_size} to {adjusted_batch_size} "
                f"for sharded training over {n_devices} devices."
            )
            batch_size = adjusted_batch_size

    mesh = None
    replicated_sharding = None
    batch_sharding = None
    if use_sharding:
        mesh = Mesh(np.asarray(jax.local_devices()[:n_devices]), axis_names=("batch",))
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        batch_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    seed = int(config.get("seed", 0))
    rng = jax.random.key(seed)

    # Optional data jitter — add Gaussian noise to break exact overlaps
    jitter_std = float(data_cfg.get("jitter_std", 0.0))
    if jitter_std > 0.0:
        rng_jitter = np.random.default_rng(seed=seed + 9999)
        eta_train = eta_train + rng_jitter.normal(0.0, jitter_std, size=eta_train.shape).astype(np.float32)
        print(f"[train_df] Applied jitter with std={jitter_std} to training data.")

    params = init_flow(model, rng, flow_cfg)

    # Compute step counts to configure schedules
    n = int(eta_train.shape[0])
    steps_per_epoch = n // batch_size
    if max_batches_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError(
            f"Not enough training samples ({n}) for batch_size={batch_size}. "
            f"Reduce batch_size or val_frac."
        )
    total_steps = int(epochs) * int(steps_per_epoch)
    print(
        "[train_df] Effective config: "
        f"flow={flow_cfg.get('type')}, batch_size={batch_size}, epochs={epochs}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"grad_clip={grad_clip}, log_every={log_every}, ckpt_every={ckpt_every}, "
        f"stop_score_p99={stop_score_p99}, stop_score_max_abs={stop_score_max_abs}, "
        f"multi_gpu={use_sharding}, n_devices={n_devices}"
    )

    # ── Configure learning rate schedule ────────────────────────────────
    lr_config = train_cfg.get("lr", 1.0e-3)
    lr_schedule_name = "cosine"  # default

    if isinstance(lr_config, dict):
        lr_schedule_name = str(lr_config.get("schedule", "cosine")).lower()
        lr_max = float(lr_config.get("max", 1.0e-3))
        lr_final = float(lr_config.get("final", 5.0e-6))
        warmup_frac = float(lr_config.get("warmup_frac", 0.03))
        warmup_steps = int(warmup_frac * total_steps)
    else:
        # Simple float → constant lr with plain Adam
        lr_max = float(lr_config)
        lr_final = lr_max
        warmup_steps = 0

    if lr_schedule_name == "plateau":
        # ── ReduceLROnPlateau (paper-style: warmup → constant → reduce on plateau) ──
        #
        # Implementation strategy:
        # Phase 1 (warmup): use optax.warmup_cosine_decay_schedule with very long
        # decay so it stays ~constant at lr_max after warmup.
        # Phase 2 (post-warmup): when plateau_state triggers an lr reduction,
        # we rebuild the entire optimizer with the new constant lr.
        # Rebuilding resets Adam momentum, which is correct: momentum from the
        # old lr regime should not persist under the new lr.
        plateau_cfg = lr_config.get("plateau", {})
        plateau_state = _PlateauState(
            warmup_steps=warmup_steps,
            initial_lr=lr_max,
            factor=float(plateau_cfg.get("factor", 0.5)),
            patience=int(plateau_cfg.get("patience", 1024)),
            min_delta=float(plateau_cfg.get("min_delta", 0.01)),
            min_lr=float(plateau_cfg.get("min_lr", lr_final)),
        )

        # Start with warmup schedule; after warmup, effectively constant at lr_max
        # (decay_steps set extremely large so cosine decay is negligible)
        warmup_lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr_max,
            warmup_steps=max(warmup_steps, 1),
            decay_steps=max(warmup_steps + total_steps, warmup_steps + 2),
            end_value=lr_max * 0.999,  # negligible decay
        )
        base_opt = optax.radam(warmup_lr_schedule)
        plateau_phase = "warmup"  # will switch to "plateau" after warmup
        current_plateau_lr = lr_max

        print(
            f"[train_df] LR schedule: ReduceLROnPlateau "
            f"(warmup={warmup_steps} steps, peak={lr_max:.2e}, "
            f"factor={plateau_state.factor}, patience={plateau_state.patience}, "
            f"min_delta={plateau_state.min_delta}, min_lr={plateau_state.min_lr:.2e})"
        )
    elif lr_schedule_name == "cosine":
        # ── Warmup + cosine decay (original schedule) ──────────────────
        _decay_steps = total_steps
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr_max,
            warmup_steps=warmup_steps,
            decay_steps=_decay_steps,
            end_value=lr_final
        )
        base_opt = optax.radam(lr_schedule)
        plateau_state = None
        plateau_phase = None
        current_plateau_lr = lr_max
        print(
            f"[train_df] LR schedule: warmup_cosine_decay "
            f"(warmup={warmup_steps}, peak={lr_max:.2e}, end={lr_final:.2e})"
        )
    else:
        raise ValueError(f"Unknown lr.schedule: {lr_schedule_name!r}. Use 'cosine' or 'plateau'.")

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    opt_state = opt.init(params)
    # Mutable holder so plateau lr rebuilds can swap the optimizer and
    # train_step (a JIT closure) picks up the new one automatically.
    opt_holder = {"opt": opt}
    train_step_fn_holder = [None]  # will be set below

    ckpt_mgr = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    step0 = 0
    if resume:
        restored = restore_latest(ckpt_mgr)
        params = restored["params"]
        restored_opt_state = restored.get("opt_state", None)
        expected_opt_state = opt.init(params)
        if restored_opt_state is None:
            print("[train_df] Resume checkpoint has no opt_state; reinitializing optimizer state.")
            opt_state = expected_opt_state
        else:
            restored_struct = jax.tree_util.tree_structure(restored_opt_state)
            expected_struct = jax.tree_util.tree_structure(expected_opt_state)
            if restored_struct != expected_struct:
                print(
                    "[train_df] Restored opt_state structure is incompatible with current optimizer; "
                    "reinitializing optimizer state and continuing from restored params."
                )
                opt_state = expected_opt_state
            else:
                opt_state = restored_opt_state
        step0 = int(restored.get("step", 0))

    if use_sharding:
        params = jax.device_put(params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)

    def _to_host(tree):
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)

    @jax.jit
    def eval_loss(params, batch):
        lp, reg = log_prob_reg_apply(model, params, batch, flow_cfg)
        return -jnp.mean(lp) + jnp.mean(reg)

    # train_step is NOT jit-decorated here because we need to re-jit it
    # whenever the optimizer is rebuilt (ReduceLROnPlateau).  Instead we
    # manage jit compilation via train_step_fn_holder.
    def _make_train_step(optimizer):
        @jax.jit
        def _train_step(params, opt_state, batch):
            def loss_fn(p):
                lp, reg = log_prob_reg_apply(model, p, batch, flow_cfg)
                return -jnp.mean(lp) + jnp.mean(reg)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state2 = optimizer.update(grads, opt_state, params)
            params2 = optax.apply_updates(params, updates)
            return params2, opt_state2, loss
        return _train_step

    train_step_fn_holder[0] = _make_train_step(opt_holder["opt"])

    metrics_path = run_dir / "metrics.csv"
    expected_header = [
        "step",
        "epoch",
        "loss",
        "score_p50",
        "score_p99",
        "score_max_abs",
        "val_loss",
        "val_score_p99",
        "lr",
    ]
    if resume and metrics_path.exists():
        with metrics_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            rows = list(reader)
        if existing_header != expected_header:
            with metrics_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=expected_header)
                writer.writeheader()
                for row in rows:
                    if row.get((existing_header or [""])[0], "") == (existing_header or [""])[0]:
                        continue
                    writer.writerow({k: row.get(k, "nan") for k in expected_header})

    if resume and metrics_path.exists():
        open_mode, write_header = "a", False
    else:
        open_mode, write_header = "w", True
    with metrics_path.open(open_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(expected_header)

        global_step = step0
        np_rng = np.random.default_rng(seed=seed)
        n_val_eval = min(2048, int(eta_val.shape[0]))
        val_eval = eta_val[:n_val_eval] if n_val_eval > 0 else None

        start_epoch = step0 // max(steps_per_epoch, 1)
        pbar = tqdm(
            total=total_steps,
            initial=min(global_step, total_steps),
            dynamic_ncols=True,
            unit="step",
            mininterval=0.5,
            smoothing=0.1,
        )
        stop_training = False

        for epoch in range(start_epoch, epochs):
            for batch_np in iter_batches(
                eta_train,
                batch_size=batch_size,
                rng=np_rng,
                shuffle=True,
                drop_remainder=True,
                max_batches=max_batches_per_epoch,
            ):
                batch = jnp.asarray(batch_np)
                if use_sharding:
                    batch = jax.device_put(batch, batch_sharding)

                params, opt_state, loss = train_step_fn_holder[0](params, opt_state, batch)
                loss_scalar = float(jax.device_get(loss))

                # ── Compute effective lr for logging ──────────────────────
                if lr_schedule_name == "plateau":
                    if plateau_phase == "warmup":
                        current_effective_lr = float(warmup_lr_schedule(global_step))
                    else:
                        current_effective_lr = current_plateau_lr
                elif lr_schedule_name == "cosine":
                    current_effective_lr = float(lr_schedule(global_step))
                else:
                    current_effective_lr = lr_max

                need_host_state = (global_step % log_every) == 0
                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    need_host_state = True

                if need_host_state and use_sharding:
                    params_host = _to_host(params)
                    opt_state_host = _to_host(opt_state)
                else:
                    params_host = params
                    opt_state_host = opt_state

                x_small = jnp.asarray(batch_np[:1024])

                if (global_step % log_every) == 0:
                    score = score_apply(model, params_host, x_small, flow_cfg)
                    score_abs = jnp.abs(score)
                    p50 = float(jnp.percentile(score_abs, 50.0))
                    p99 = float(jnp.percentile(score_abs, 99.0))
                    smax = float(jnp.max(score_abs))

                    if val_eval is not None:
                        val_batch = jnp.asarray(val_eval)
                        val_loss = float(eval_loss(params_host, val_batch))
                        val_score = score_apply(model, params_host, val_batch, flow_cfg)
                        val_score_p99 = float(jnp.percentile(jnp.abs(val_score), 99.0))
                    else:
                        val_loss = math.nan
                        val_score_p99 = math.nan

                    writer.writerow([global_step, epoch, loss_scalar, p50, p99, smax, val_loss, val_score_p99, current_effective_lr])
                    f.flush()
                    if logger is not None:
                        logger.log_scalars(global_step, {
                            "epoch": epoch, "loss": loss_scalar,
                            "score_p50": p50, "score_p99": p99, "score_max_abs": smax,
                            "val_loss": val_loss, "val_score_p99": val_score_p99,
                            "lr": current_effective_lr,
                        })
                    postfix = dict(nll=loss_scalar, p99=p99, smax=smax, epoch=epoch, lr=current_effective_lr)
                    if not math.isnan(val_loss):
                        postfix["val_nll"] = val_loss
                    pbar.set_postfix(**postfix)

                    # ── ReduceLROnPlateau: check and rebuild optimizer ──
                    if plateau_state is not None:
                        old_lr = plateau_state.current_lr
                        plateau_state.step(global_step, val_loss)

                        # Transition from warmup to plateau phase
                        if plateau_phase == "warmup" and global_step >= plateau_state.warmup_steps:
                            plateau_phase = "plateau"
                            current_plateau_lr = lr_max
                            print(f"[train_df] LR schedule: warmup complete, switching to plateau phase at lr={lr_max:.2e}")

                        # If lr was reduced, rebuild optimizer with new constant lr
                        if plateau_state.current_lr != old_lr:
                            new_lr = plateau_state.current_lr
                            current_plateau_lr = new_lr
                            print(f"[train_df] Rebuilding optimizer with new lr={new_lr:.2e}")
                            new_schedule = optax.constant_schedule(new_lr)
                            new_base_opt = optax.radam(new_schedule)
                            new_opt = optax.chain(optax.clip_by_global_norm(grad_clip), new_base_opt)
                            # Re-init optimizer state (resets momentum for clean transition)
                            if use_sharding:
                                params_for_init = _to_host(params)
                            else:
                                params_for_init = params
                            opt_state = new_opt.init(params_for_init)
                            # Swap optimizer in holder and re-jit train_step
                            opt_holder["opt"] = new_opt
                            train_step_fn_holder[0] = _make_train_step(new_opt)

                        if plateau_state.should_stop:
                            print(
                                f"[train_df] ReduceLROnPlateau: lr ({plateau_state.current_lr:.2e}) "
                                f"dropped below min_lr ({plateau_state.min_lr:.2e}). Stopping."
                            )
                            stop_training = True

                    stop_by_p99 = stop_score_p99 is not None and (
                        p99 >= stop_score_p99
                        or (not math.isnan(val_score_p99) and val_score_p99 >= stop_score_p99)
                    )
                    stop_by_smax = stop_score_max_abs is not None and smax >= stop_score_max_abs
                    if stop_by_p99 or stop_by_smax:
                        stop_training = True
                        print(
                            "[train_df] Early stopping: "
                            f"step={global_step}, epoch={epoch}, "
                            f"score_p99={p99}, val_score_p99={val_score_p99}, "
                            f"score_max_abs={smax}"
                        )

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    save(ckpt_mgr, global_step, {"params": params_host, "opt_state": opt_state_host, "step": global_step})

                global_step += 1
                pbar.update(1)
                if stop_training:
                    break
            if stop_training:
                break

        pbar.close()

    if use_sharding:
        params = _to_host(params)
        opt_state = _to_host(opt_state)

    save(ckpt_mgr, global_step, {"params": params, "opt_state": opt_state, "step": global_step})
    finalize(ckpt_mgr)
    print(f"Saved final checkpoint at step {global_step}.")

    return {
        "params": params,
        "normalizer": normalizer,
        "model": model,
        "final_step": global_step,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train DF (RealNVP / FFJORD) on eta.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--override", type=str, default=None,
        help="JSON string of config overrides, e.g. '{\"train\": {\"epochs\": 64}}'.",
    )
    parser.add_argument("--logger", type=str, default="csv", help="Logger backend: csv, wandb, tensorboard, wandb+tb.")
    parser.add_argument("--project", type=str, default="dp-plummer", help="W&B project name.")
    parser.add_argument("--run-name", type=str, default=None, help="W&B / experiment run name.")
    args = parser.parse_args()

    import json
    import sys
    _repo = Path(__file__).resolve().parents[1]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from dpjax.config import merge_config
    from experiments.logger import ExperimentLogger

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.override:
        cfg = merge_config(cfg, json.loads(args.override))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with ExperimentLogger(
        run_dir, project=args.project, run_name=args.run_name,
        backend=args.logger, config=cfg,
    ) as logger:
        run_df_training(cfg, args.data, run_dir, resume=args.resume, logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())