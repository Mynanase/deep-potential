from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_server_jobs_are_scheduler_free_and_valid_bash():
    legacy_suffix = "." + "s" + "batch"
    assert not [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and path.suffix == legacy_suffix
    ]

    scripts = sorted((PROJECT_ROOT / "jobs").glob("*.sh"))
    assert scripts
    for script in scripts:
        assert os.access(script, os.X_OK)
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )

    active_paths = [
        PROJECT_ROOT / "PROJECT_STRUCTURE.md",
        PROJECT_ROOT / "README_JAX.md",
        PROJECT_ROOT / "docs" / "auriga_halo12_df_stage.md",
        PROJECT_ROOT / "docs" / "operations_maintenance_guide.md",
        *scripts,
    ]
    active_text = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in active_paths
    )
    assert ("sl" + "urm") not in active_text
    assert ("s" + "batch") not in active_text
