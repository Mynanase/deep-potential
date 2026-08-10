from __future__ import annotations

import optax
import pytest

from experiments.workflows.optimizers import build_optimizer


@pytest.mark.parametrize("name", ["adam", "radam"])
def test_build_optimizer_accepts_model_recipe_names(name):
    optimizer = build_optimizer(name, 1.0e-3)
    assert isinstance(optimizer, optax.GradientTransformation)


def test_build_optimizer_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer("not-an-optimizer", 1.0e-3)
