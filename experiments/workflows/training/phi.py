"""Operational potential training workflow."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm.auto import tqdm

from dpjax.models.potential import (
    PotentialConfig,
    PotentialMLP,
    grad_phi_apply,
    laplacian_phi_apply,
)
from dpjax.physics.cbe import (
    loss_cbe_mse,
    loss_cbe_robust,
    loss_negative_density,
    residual_A,
)
from dpjax.utils.tree import count_parameters, mean_square
from experiments.datasets.phase_space import (
    iter_batches,
    load_df_support_eta,
    require_physics_compatible_transform,
)
from experiments.paths import ensure_dir, resolve_path
from experiments.workflows.artifacts import load_df
from experiments.workflows.checkpoints import (
    create_manager,
    finalize,
    restore_latest,
    save,
)
from experiments.workflows.logging import ExperimentLogger
from experiments.workflows.optimizers import build_optimizer
from experiments.workflows.score_sources import (
    resolve_score_source,
    score_std_batch,
)

# ---------------------------------------------------------------------------
# Operational training workflow – called by the run entry point
# ---------------------------------------------------------------------------

def run_phi_training(
    config: dict[str, Any],
    data_path: str | Path,
    df_run_dir: str | Path,
    run_dir: str | Path,
    *,
    resume: bool = False,
    init_params_dir: str | Path | None = None,
    logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    """Train the potential network Phi with a frozen DF using CBE residual.

    Parameters
    ----------
    config : dict
        Full training configuration (typically loaded from a YAML file).
    data_path : str or Path
        Path to the HDF5 data file containing ``eta``.
    df_run_dir : str or Path
        Directory of a completed DF training run (must contain
        ``config.yaml``, ``normalizer.npz``, and ``ckpt/``).
    run_dir : str or Path
        Directory for checkpoints, metrics, and config snapshots.
    resume : bool
        If ``True``, resume from the latest checkpoint in *run_dir*.
    init_params_dir : str or Path, optional
        If provided, initialize Phi parameters from the latest checkpoint
        found in ``init_params_dir/ckpt`` while creating a fresh optimizer
        state. This is intended for fine-tuning with changed optimizer or
        loss settings.

    Returns
    -------
    dict
        ``{"phi_params": ..., "phi_model": ..., "df_model": ...,
          "df_params": ..., "normalizer": ..., "final_step": int}``
    """
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    run_dir = ensure_dir(run_dir)
    if resume and init_params_dir is not None:
        raise ValueError("--resume and --init-params are mutually exclusive.")
    if init_params_dir is not None:
        init_params_dir = resolve_path(init_params_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    df_model, df_params, normalizer, df_cfg, coord_transform = load_df(df_run_dir)
    require_physics_compatible_transform(
        coord_transform,
        operation="Phi/CBE training",
    )
    flow_cfg = df_cfg.get("flow", {})
    score_source = resolve_score_source(config)

    df_data_cfg = df_cfg.get("data", {})
    df_dataset = str(df_data_cfg.get("dataset", "eta"))
    requested_dataset = str(
        config.get("data", {}).get("dataset", df_dataset)
    )
    if requested_dataset != df_dataset:
        raise ValueError(
            "Phi training must use the same row-aligned dataset as the DF "
            f"({df_dataset!r}); got {requested_dataset!r}."
        )
    support_eta, support_source, source_n = load_df_support_eta(
        data_path,
        df_run_dir,
        df_data_cfg,
        coordinate_transform=coord_transform,
    )
    print(
        "[train_phi] DF support: "
        f"source={support_source}, kept={support_eta.shape[0]}/"
        f"{source_n}"
    )
    eta_std = normalizer.transform(support_eta)

    pot_cfg = config.get("potential", {})
    phi_model = PotentialMLP(PotentialConfig(
        hidden_sizes=tuple(int(x) for x in pot_cfg.get("hidden_sizes", [512, 512, 512, 512])),
        output_scale=float(pot_cfg.get("output_scale", 1.0)),
    ))

    train_cfg = config.get("train", {})
    optimizer_name = str(train_cfg.get("optimizer", "radam"))
    batch_size = int(train_cfg.get("batch_size", 4096))
    epochs = int(train_cfg.get("epochs", 64))
    loss_type = str(train_cfg.get("loss_type", "robust")).lower()
    alpha = float(train_cfg.get("alpha", 1.0))
    beta = float(train_cfg.get("beta", 1.0))
    lambda_mass = float(train_cfg.get("lambda_mass", 1.0))
    l2_reg = float(train_cfg.get("l2_reg", 0.1))
    rw_cfg = train_cfg.get("reweight", {})
    rw_gamma = float(rw_cfg.get("gamma", 0.0))
    rw_r_ref = float(rw_cfg.get("r_ref", 1.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 200))
    max_to_keep = int(train_cfg.get("max_to_keep", 3))
    n_devices = int(jax.local_device_count())
    use_sharding = bool(train_cfg.get("multi_gpu", True)) and n_devices > 1

    if rw_gamma < 0.0:
        raise ValueError(f"train.reweight.gamma must be >= 0, got {rw_gamma}.")
    if rw_r_ref <= 0.0:
        raise ValueError(f"train.reweight.r_ref must be > 0, got {rw_r_ref}.")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"train.beta must be finite and > 0, got {beta}.")
    if not np.isfinite(lambda_mass) or lambda_mass < 0.0:
        raise ValueError(
            "train.lambda_mass must be finite and >= 0, "
            f"got {lambda_mass}."
        )

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
                f"[train_phi] Adjusting batch_size from {batch_size} to {adjusted_batch_size} "
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

    if loss_type not in {"robust", "mse"}:
        raise ValueError(f"Unknown train.loss_type={loss_type!r}; expected 'robust' or 'mse'.")
    mass_batch_size = int(train_cfg.get("mass_batch_size", batch_size))
    if mass_batch_size <= 0:
        raise ValueError(
            "train.mass_batch_size must be positive, "
            f"got {mass_batch_size}."
        )
    mass_uniform_points = int(train_cfg.get("mass_uniform_points", 0))
    mass_uniform_r_max = float(train_cfg.get("mass_uniform_r_max", 75.0))
    mass_uniform_domain = str(
        train_cfg.get("mass_uniform_domain", "cube")
    ).lower()
    mass_uniform_weight = float(train_cfg.get("mass_uniform_weight", 1.0))
    mass_uniform_weight_start = float(
        train_cfg.get("mass_uniform_weight_start", mass_uniform_weight)
    )
    mass_uniform_ramp_frac = float(
        train_cfg.get("mass_uniform_ramp_frac", 0.0)
    )
    if mass_uniform_points < 0:
        raise ValueError(
            "train.mass_uniform_points must be >= 0, "
            f"got {mass_uniform_points}."
        )
    if mass_uniform_points > 0:
        if mass_uniform_r_max <= 0.0:
            raise ValueError(
                "train.mass_uniform_r_max must be > 0, "
                f"got {mass_uniform_r_max}."
            )
        if mass_uniform_weight < 0.0 or mass_uniform_weight_start < 0.0:
            raise ValueError(
                "train.mass_uniform_weight and "
                "train.mass_uniform_weight_start must be >= 0, "
                f"got {mass_uniform_weight} / {mass_uniform_weight_start}."
            )
        if mass_uniform_ramp_frac < 0.0 or mass_uniform_ramp_frac > 1.0:
            raise ValueError(
                "train.mass_uniform_ramp_frac must be in [0, 1], "
                f"got {mass_uniform_ramp_frac}."
            )
        if mass_uniform_domain not in {"sphere", "cube"}:
            raise ValueError(
                "train.mass_uniform_domain must be 'sphere' or 'cube', "
                f"got {mass_uniform_domain!r}."
            )

    seed = int(config.get("seed", 1))
    rng = jax.random.key(seed)

    dummy_x = jnp.zeros((1, 3), dtype=jnp.float32)
    phi_params = phi_model.init(rng, dummy_x)["params"]
    if init_params_dir is not None:
        init_ckpt_mgr = create_manager(Path(init_params_dir) / "ckpt")
        init_restored = restore_latest(init_ckpt_mgr)
        phi_params = init_restored["params"]
        print(
            "[train_phi] restore_mode=weights_only, "
            f"init_params_from={init_params_dir}"
        )
    parameter_count = count_parameters(phi_params)
    (run_dir / "model_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.model-summary.v1",
                "kind": "phi",
                "parameter_count": parameter_count,
                "score_source": score_source,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"[train_phi] parameter_count={parameter_count}")

    # Compute step counts to configure schedules
    n = int(eta_std.shape[0])
    steps_per_epoch = n // batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            f"Not enough training samples ({n}) for batch_size={batch_size}. "
            "Reduce batch_size."
        )
    total_steps = int(epochs) * int(steps_per_epoch)

    device_list = list(jax.local_devices()[:n_devices])
    per_device_batch = (batch_size // n_devices) if use_sharding else batch_size
    print(
        "[train_phi] setup: "
        f"backend={jax.default_backend()}, "
        f"use_sharding={use_sharding}, "
        f"n_devices={n_devices}, "
        f"global_batch_size={batch_size}, "
        f"per_device_batch_size={per_device_batch}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"total_steps={total_steps}"
    )
    print(f"[train_phi] local devices: {device_list}")
    print(f"[train_phi] score_source={score_source}")
    print(
        "[train_phi] reweight: "
        f"gamma={rw_gamma}, r_ref={rw_r_ref}"
    )
    print(
        "[train_phi] mass constraint: "
        f"lambda_mass={lambda_mass}, beta={beta}, "
        f"mass_batch_size={min(mass_batch_size, batch_size)}"
    )
    if mass_uniform_points > 0:
        ramp_info = ""
        if mass_uniform_ramp_frac > 0.0:
            ramp_info = (
                f", ramp={mass_uniform_weight_start}"
                f"->{mass_uniform_weight} over "
                f"{mass_uniform_ramp_frac * 100:.0f}% of steps"
            )
        print(
            "[train_phi] mass uniform probes: "
            f"n={mass_uniform_points}, domain={mass_uniform_domain}, "
            f"r_max={mass_uniform_r_max}, weight={mass_uniform_weight}"
            f"{ramp_info}"
        )

    # Configure learning rate
    lr_config = train_cfg.get("lr", 1.0e-3)
    if isinstance(lr_config, dict):
        lr_max = float(lr_config.get("max", 0.05))
        lr_final = float(lr_config.get("final", 5.0e-6))
        warmup_frac = float(lr_config.get("warmup_frac", 0.1))
        warmup_steps = int(warmup_frac * total_steps)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr_max,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr_final
        )
        base_opt = build_optimizer(optimizer_name, lr_schedule)
    else:
        base_opt = build_optimizer(optimizer_name, float(lr_config))

    opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    opt_state = opt.init(phi_params)

    ckpt_mgr = create_manager(run_dir / "ckpt", max_to_keep=max_to_keep)
    step0 = 0
    if resume:
        print(f"[train_phi] restore_mode=full_resume, run_dir={run_dir}")
        restored = restore_latest(ckpt_mgr)
        phi_params = restored["params"]
        step0 = int(restored.get("step", 0))
        # Try to restore optimizer state; if the checkpoint was saved with a
        # different optimizer config (e.g. changed lr schedule), orbax may
        # deserialise the state as plain dicts which are incompatible.
        try:
            restored_opt = restored["opt_state"]
            # Quick smoke-test: run a dummy update to verify compatibility
            _dummy_grads = jax.tree_util.tree_map(jnp.zeros_like, phi_params)
            opt.update(_dummy_grads, restored_opt, phi_params)
            opt_state = restored_opt
            print(f"[train_phi] restored opt_state from step {step0}")
        except (AttributeError, TypeError, ValueError) as exc:
            print(
                f"[train_phi] WARNING: opt_state incompatible ({exc!r}), "
                "re-initialising optimizer from current params."
            )
            opt_state = opt.init(phi_params)

    def _to_host(tree):
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)

    if use_sharding:
        phi_params = jax.device_put(phi_params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)
        df_params = jax.device_put(df_params, replicated_sharding)
        df_params_host = _to_host(df_params)
    else:
        df_params_host = df_params

    @jax.jit
    def train_step(phi_params, opt_state, eta_std_batch, mass_key, mass_weight_now):
        x_std = eta_std_batch[:, :3]
        std_x = jnp.asarray(normalizer.std[:3], dtype=eta_std_batch.dtype)
        mean_x = jnp.asarray(normalizer.mean[:3], dtype=eta_std_batch.dtype)

        def loss_fn(p):
            score_std = score_std_batch(
                score_source,
                eta_std_batch,
                normalizer,
                df_model=df_model,
                df_params=df_params,
                flow_cfg=flow_cfg,
            )
            grad_phi_std = grad_phi_apply(phi_model, p, x_std)

            # Compute per-sample radial weights
            weights = None
            if rw_gamma > 0.0:
                x_phys = x_std * std_x + mean_x
                r_phys = jnp.sqrt(jnp.sum(x_phys**2, axis=-1) + 1e-12)
                raw_w = (r_phys / rw_r_ref) ** rw_gamma
                weights = raw_w / jnp.mean(raw_w)  # normalize so mean(w)=1

            residual = residual_A(
                eta_std_batch,
                score_std,
                grad_phi_std,
                normalizer,
            )
            if loss_type == "mse":
                residual_loss = loss_cbe_mse(
                    residual,
                    weights=weights,
                )
            else:
                residual_loss = loss_cbe_robust(
                    residual,
                    jnp.zeros_like(residual),
                    alpha=alpha,
                    beta=beta,
                    lambda_mass=0.0,
                    weights=weights,
                )

            mass_loss = jnp.zeros((), dtype=residual.dtype)
            negative_fraction = jnp.zeros((), dtype=residual.dtype)
            negative_fraction_uniform = jnp.zeros((), dtype=residual.dtype)
            if lambda_mass > 0.0:
                n_mass = min(mass_batch_size, x_std.shape[0])
                x_mass = x_std[:n_mass]
                mass_weights = None if weights is None else weights[:n_mass]
                laplacian_phi_phys = laplacian_phi_apply(
                    phi_model,
                    p,
                    x_mass,
                    std_x=std_x,
                )
                mass_loss = loss_negative_density(
                    laplacian_phi_phys,
                    beta=beta,
                    weights=mass_weights,
                )
                negative_fraction = jnp.mean(laplacian_phi_phys < 0.0)

                if mass_uniform_points > 0 and (
                    mass_uniform_weight > 0.0
                    or mass_uniform_weight_start > 0.0
                ):
                    if mass_uniform_domain == "sphere":
                        dirs = jax.random.normal(
                            mass_key,
                            (mass_uniform_points, 3),
                            dtype=x_std.dtype,
                        )
                        dirs = dirs / jnp.sqrt(
                            jnp.sum(dirs**2, axis=-1, keepdims=True)
                            + 1.0e-12
                        )
                        radii = mass_uniform_r_max * jax.random.uniform(
                            mass_key,
                            (mass_uniform_points, 1),
                            dtype=x_std.dtype,
                        ) ** (1.0 / 3.0)
                        x_uniform_phys = dirs * radii
                    else:
                        x_uniform_phys = (
                            2.0
                            * jax.random.uniform(
                                mass_key,
                                (mass_uniform_points, 3),
                                dtype=x_std.dtype,
                            )
                            - 1.0
                        ) * mass_uniform_r_max
                    x_uniform_std = (
                        x_uniform_phys - mean_x[None, :]
                    ) / std_x[None, :]
                    laplacian_uniform = laplacian_phi_apply(
                        phi_model,
                        p,
                        x_uniform_std,
                        std_x=std_x,
                    )
                    mass_loss_uniform = loss_negative_density(
                        laplacian_uniform,
                        beta=beta,
                    )
                    negative_fraction_uniform = jnp.mean(
                        laplacian_uniform < 0.0
                    )
                    mass_loss = (
                        mass_loss
                        + mass_weight_now * mass_loss_uniform
                    ) / (1.0 + mass_weight_now)

            total_loss = (
                residual_loss
                + lambda_mass * mass_loss
                + l2_reg * mean_square(p)
            )
            return total_loss, (
                residual_loss,
                mass_loss,
                negative_fraction,
                negative_fraction_uniform,
            )

        (loss, aux), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(phi_params)
        residual_loss, mass_loss, negative_fraction, negative_fraction_uniform = aux
        updates, opt_state2 = opt.update(grads, opt_state, phi_params)
        phi_params2 = optax.apply_updates(phi_params, updates)
        return (
            phi_params2,
            opt_state2,
            loss,
            residual_loss,
            mass_loss,
            negative_fraction,
            negative_fraction_uniform,
        )

    metrics_path = run_dir / "metrics.csv"
    expected_header = [
        "step",
        "epoch",
        "loss",
        "residual_loss",
        "mass_penalty",
        "negative_laplacian_fraction",
        "negative_laplacian_fraction_uniform",
        "residual_mean",
        "residual_std",
        "residual_p99_abs",
    ]
    if resume and metrics_path.exists():
        with metrics_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            rows = list(reader)
        if existing_header != expected_header:
            with metrics_path.open("w", newline="") as f:
                dict_writer = csv.DictWriter(
                    f,
                    fieldnames=expected_header,
                )
                dict_writer.writeheader()
                for row in rows:
                    dict_writer.writerow(
                        {key: row.get(key, "nan") for key in expected_header}
                    )
    write_header = not metrics_path.exists() or not resume
    open_mode = "a" if resume else "w"
    with metrics_path.open(open_mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(expected_header)

        global_step = step0
        np_rng = np.random.default_rng(seed=seed)
        start_epoch = step0 // steps_per_epoch

        pbar = tqdm(
            total=total_steps,
            initial=min(global_step, total_steps),
            dynamic_ncols=True,
            unit="step",
            mininterval=0.5,
            smoothing=0.1,
        )

        for epoch in range(start_epoch, epochs):
            for batch_np in iter_batches(
                eta_std,
                batch_size=batch_size,
                rng=np_rng,
                shuffle=True,
                drop_remainder=True,
            ):
                eta_b = jnp.asarray(batch_np)
                if use_sharding:
                    eta_b = jax.device_put(eta_b, batch_sharding)
                mass_key = jax.random.fold_in(rng, global_step)
                if mass_uniform_points > 0 and mass_uniform_ramp_frac > 0.0:
                    ramp_steps = max(
                        1, int(mass_uniform_ramp_frac * total_steps)
                    )
                    ramp_t = min(global_step / ramp_steps, 1.0)
                    mass_weight_now = (
                        mass_uniform_weight_start
                        + (
                            mass_uniform_weight
                            - mass_uniform_weight_start
                        )
                        * ramp_t
                    )
                else:
                    mass_weight_now = mass_uniform_weight
                (
                    phi_params,
                    opt_state,
                    loss,
                    residual_loss,
                    mass_loss,
                    negative_fraction,
                    negative_fraction_uniform,
                ) = train_step(
                    phi_params,
                    opt_state,
                    eta_b,
                    mass_key,
                    mass_weight_now,
                )
                loss_scalar = float(jax.device_get(loss))
                residual_loss_scalar = float(jax.device_get(residual_loss))
                mass_loss_scalar = float(jax.device_get(mass_loss))
                negative_fraction_scalar = float(
                    jax.device_get(negative_fraction)
                )
                negative_fraction_uniform_scalar = float(
                    jax.device_get(negative_fraction_uniform)
                )

                need_host_state = (global_step % log_every) == 0
                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    need_host_state = True

                if need_host_state and use_sharding:
                    phi_params_host = _to_host(phi_params)
                    opt_state_host = _to_host(opt_state)
                else:
                    phi_params_host = phi_params
                    opt_state_host = opt_state

                if (global_step % log_every) == 0:
                    eta_small = jnp.asarray(batch_np[:1024])
                    score_small = score_std_batch(
                        score_source,
                        eta_small,
                        normalizer,
                        df_model=df_model,
                        df_params=df_params_host,
                        flow_cfg=flow_cfg,
                    )
                    grad_phi_small = grad_phi_apply(phi_model, phi_params_host, eta_small[:, :3])
                    r = residual_A(eta_small, score_small, grad_phi_small, normalizer)
                    r_mean = float(jnp.mean(r))
                    r_std = float(jnp.std(r))
                    r_p99 = float(jnp.percentile(jnp.abs(r), 99.0))
                    writer.writerow(
                        [
                            global_step,
                            epoch,
                            loss_scalar,
                            residual_loss_scalar,
                            mass_loss_scalar,
                            negative_fraction_scalar,
                            negative_fraction_uniform_scalar,
                            r_mean,
                            r_std,
                            r_p99,
                        ]
                    )
                    f.flush()
                    if logger is not None:
                        logger.log_scalars(global_step, {
                            "epoch": epoch, "loss": loss_scalar,
                            "residual_loss": residual_loss_scalar,
                            "mass_penalty": mass_loss_scalar,
                            "negative_laplacian_fraction": (
                                negative_fraction_scalar
                            ),
                            "negative_laplacian_fraction_uniform": (
                                negative_fraction_uniform_scalar
                            ),
                            "residual_mean": r_mean, "residual_std": r_std,
                            "residual_p99_abs": r_p99,
                        })
                    if mass_uniform_points > 0:
                        pbar.set_postfix(
                            loss=loss_scalar,
                            mass=mass_loss_scalar,
                            neg_frac=negative_fraction_scalar,
                            neg_frac_u=negative_fraction_uniform_scalar,
                            w_u=mass_weight_now,
                            r_std=r_std,
                            r_p99=r_p99,
                            epoch=epoch,
                        )
                    else:
                        pbar.set_postfix(
                            loss=loss_scalar,
                            mass=mass_loss_scalar,
                            neg_frac=negative_fraction_scalar,
                            r_std=r_std,
                            r_p99=r_p99,
                            epoch=epoch,
                        )

                if ckpt_every and (global_step % ckpt_every) == 0 and global_step != step0:
                    save(ckpt_mgr, global_step, {"params": phi_params_host, "opt_state": opt_state_host, "step": global_step})

                global_step += 1
                pbar.update(1)

        pbar.close()

    if use_sharding:
        phi_params = _to_host(phi_params)
        opt_state = _to_host(opt_state)

    save(ckpt_mgr, global_step, {"params": phi_params, "opt_state": opt_state, "step": global_step})
    finalize(ckpt_mgr)
    print(f"Saved final checkpoint at step {global_step}.")

    return {
        "phi_params": phi_params,
        "phi_model": phi_model,
        "df_model": df_model,
        "df_params": df_params_host,
        "normalizer": normalizer,
        "score_source": score_source,
        "final_step": global_step,
    }
