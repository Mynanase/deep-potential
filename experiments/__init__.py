"""Operational experiment layer and supported run entry points."""

from experiments.runtime import configure_runtime_environment

# Apply project-safe defaults before any experiment entry point imports JAX.
configure_runtime_environment()
