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


def test_dpjax_core_never_imports_operational_layers():
    core_files = (PROJECT_ROOT / "dpjax").rglob("*.py")
    banned_roots = {
        "csv",
        "h5py",
        "matplotlib",
        "orbax",
        "pathlib",
        "wandb",
        "yaml",
    }
    violations = {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            module
            for module in _imported_modules(path)
            if module.split(".", maxsplit=1)[0] in banned_roots
            or module == "experiments"
            or module.startswith("experiments.")
        )
        for path in core_files
    }
    assert not {path: modules for path, modules in violations.items() if modules}


def test_dpjax_has_no_dataset_workflow_or_plotting_packages():
    forbidden = {
        PROJECT_ROOT / "dpjax" / "config.py",
        PROJECT_ROOT / "dpjax" / "data.py",
        PROJECT_ROOT / "dpjax" / "datasets",
        PROJECT_ROOT / "dpjax" / "evaluation.py",
        PROJECT_ROOT / "dpjax" / "paths.py",
        PROJECT_ROOT / "dpjax" / "plotting",
        PROJECT_ROOT / "dpjax" / "workflows",
    }
    remaining = []
    for path in forbidden:
        if path.is_file() or path.is_dir() and any(path.rglob("*.py")):
            remaining.append(path)
    assert not remaining


def test_experiments_package_keeps_a_small_supported_surface():
    expected = {
        "__init__.py",
        "gendata_plummer.py",
        "eval_df.py",
        "eval_phi.py",
        "inspect_data.py",
        "launch.py",
        "list_runs.py",
        "paths.py",
        "plot_df.py",
        "plot_phi.py",
        "prepare_auriga.py",
        "run.py",
        "run_df.py",
        "run_eval.py",
        "run_phi.py",
        "run_plot.py",
        "runtime.py",
        "smoke_dpjax.py",
    }
    actual = {
        path.name
        for path in (PROJECT_ROOT / "experiments").glob("*.py")
    }
    assert actual == expected


def test_active_plotting_has_no_machine_or_case_specific_paths():
    plotting_files = (PROJECT_ROOT / "experiments" / "plotting").glob("*.py")
    forbidden = ("/localdisk/", "halo12", "plummer")
    violations = {
        str(path.relative_to(PROJECT_ROOT)): token
        for path in plotting_files
        for token in forbidden
        if token in path.read_text(encoding="utf-8").lower()
    }
    assert not violations
