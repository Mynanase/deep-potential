from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_dpjax_workflows_never_import_experiments():
    workflow_files = (PROJECT_ROOT / "dpjax" / "workflows").rglob("*.py")
    violations = {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            module
            for module in _imported_modules(path)
            if module == "experiments" or module.startswith("experiments.")
        )
        for path in workflow_files
    }
    assert not {path: modules for path, modules in violations.items() if modules}


def test_run_entrypoints_never_import_legacy_experiment_modules():
    for name in ("run_df.py", "run_phi.py", "run_eval.py"):
        path = PROJECT_ROOT / "experiments" / name
        assert not {
            module
            for module in _imported_modules(path)
            if module == "experiments" or module.startswith("experiments.")
        }


def test_experiments_package_keeps_a_small_supported_surface():
    expected = {
        "__init__.py",
        "gendata_plummer.py",
        "inspect_data.py",
        "prepare_auriga.py",
        "run_df.py",
        "run_eval.py",
        "run_phi.py",
        "smoke_dpjax.py",
    }
    actual = {
        path.name
        for path in (PROJECT_ROOT / "experiments").glob("*.py")
    }
    assert actual == expected
