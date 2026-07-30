from __future__ import annotations

import h5py
import numpy as np
import pytest

from dpjax.data import DFDataSelection, Normalizer, phase_space_sha256
from experiments import train_phi


def test_phi_training_reuses_persisted_df_support(tmp_path, monkeypatch):
    eta = np.arange(24, dtype=np.float32).reshape(4, 6)
    data_path = tmp_path / "data.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("eta", data=eta)

    df_run_dir = tmp_path / "df"
    df_run_dir.mkdir()
    DFDataSelection(
        source_size=4,
        dataset="eta",
        source_sha256=phase_space_sha256(eta),
        clip_sigma=4.5,
        split_seed=7,
        support_indices=np.array([0, 2]),
        train_indices=np.array([2]),
        val_indices=np.array([0]),
    ).save_npz(df_run_dir / "data_selection.npz")

    normalizer = Normalizer(
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
    )
    df_config = {
        "data": {"dataset": "eta", "clip_sigma": 4.5},
        "flow": {"type": "realnvp", "dim": 6},
    }
    monkeypatch.setattr(
        train_phi,
        "load_df",
        lambda _: (object(), {}, normalizer, df_config, None),
    )

    config = {
        "seed": 1,
        "data": {"dataset": "eta"},
        "potential": {"hidden_sizes": [2]},
        "train": {
            "batch_size": 3,
            "epochs": 1,
            "loss_type": "mse",
            "lambda_mass": 0.0,
            "multi_gpu": False,
        },
    }

    with pytest.raises(
        ValueError,
        match=r"Not enough training samples \(2\) for batch_size=3",
    ):
        train_phi.run_phi_training(
            config,
            data_path,
            df_run_dir,
            tmp_path / "phi",
        )
