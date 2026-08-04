# Legacy Jupyter notebooks

New post-training analysis uses the git-friendly marimo entry points under
`analysis/`. Start with:

```bash
marimo edit analysis/halo12.py
```

The `.ipynb` files in this directory are retained for compatibility while
analysis is migrated. Training remains a standalone process.

Interactive Jupyter notebooks for the Deep Potential (JAX) project.

## Prerequisites

```bash
# From the project root, install dpjax as an editable package with notebook extras
pip install -e ".[notebook]"
```

This ensures `dpjax` is importable from any working directory, including `notebooks/`.

## Notebook Index

| Notebook | Description |
|----------|-------------|
| `07_analysis.ipynb` | Post-training analysis of saved DF and potential runs |

Training and evaluation live in the importable CLI modules under
`experiments/`; notebooks do not duplicate the training pipeline.

## Tips

- **CPU debugging**: Set `JAX_PLATFORM_NAME=cpu` before starting Jupyter to avoid GPU compilation overhead during prototyping.
- **Shared GPU**: Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if you share the GPU with other users.
- **JIT caching**: Once a cell with `@jax.jit` functions runs, the compiled code stays in memory. Re-running the cell is nearly instant.
- **Path handling**: All notebooks use `dpjax.paths` for path resolution, so they work regardless of the current working directory.
