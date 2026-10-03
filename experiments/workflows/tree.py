"""Small parameter-tree helper."""

from __future__ import annotations


def count_parameters(tree) -> int:
    import jax
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree) if hasattr(leaf, "size"))
