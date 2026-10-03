"""Deterministic msgpack checkpoints for the minimal T4a workflow."""

from __future__ import annotations

import msgpack
import msgpack_numpy as m

m.patch()


def save(path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(msgpack.packb(item, use_bin_type=True, default=m.encode))
    tmp.rename(path)


def load(path) -> dict:
    return msgpack.unpackb(path.read_bytes(), object_hook=m.decode, strict_map_key=False)
