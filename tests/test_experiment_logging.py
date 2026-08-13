from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from experiments.workflows.logging import ExperimentLogger


def test_wandb_backend_receives_mode_and_entity(tmp_path, monkeypatch):
    captured = {}

    class DummyRun:
        def log(self, metrics, *, step):
            pass

        def finish(self):
            pass

    def fake_init(**kwargs):
        captured.update(kwargs)
        return DummyRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    with ExperimentLogger(
        tmp_path,
        backend="wandb",
        project="deep-potential",
        run_name="trial-phi",
        entity="research-team",
        mode="offline",
        config={"seed": 42},
    ):
        pass

    assert captured["project"] == "deep-potential"
    assert captured["name"] == "trial-phi"
    assert captured["entity"] == "research-team"
    assert captured["mode"] == "offline"


def test_wandb_mode_is_validated(tmp_path):
    with pytest.raises(ValueError, match="W&B mode"):
        ExperimentLogger(tmp_path, backend="wandb", mode="invalid")
