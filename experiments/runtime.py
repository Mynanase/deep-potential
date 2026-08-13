"""Project-level process environment defaults for experiment entry points."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


DEFAULT_RUNTIME_ENV = {
    # Avoid reserving most GPU memory when the JAX backend initializes. Users,
    # containers, and schedulers may override this before launching Python.
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


def configure_runtime_environment(
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Apply non-destructive project defaults and return the target mapping."""
    target = os.environ if env is None else env
    for key, value in DEFAULT_RUNTIME_ENV.items():
        target.setdefault(key, value)
    return target
