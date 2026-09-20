"""Structural checks for the haloscope package layer."""

import ast
from pathlib import Path

import pytest

HALOSCOPE_PACKAGE = Path(__file__).resolve().parents[2] / "density_field_properties" / "haloscope"
FORBIDDEN_IMPORT_PREFIXES = (
    "density_field_properties.preprocessing",
    "density_field_properties.read_data",
    "density_field_properties.validation",
    "density_field_properties.pipelines",
)
CANONICAL_MODULES = (
    "bins.py",
    "training.py",
    "predict.py",
    "model.py",
    "__init__.py",
)


def _module_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


@pytest.mark.parametrize("module_name", CANONICAL_MODULES)
def test_haloscope_canonical_modules_avoid_forbidden_dependencies(module_name):
    """
    Haloscope must remain isolated from preprocessing, read_data, and validation.
    """
    imports = _module_imports(HALOSCOPE_PACKAGE / module_name)
    forbidden = [
        name
        for name in imports
        if any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    ]
    assert not forbidden, f"{module_name} imports forbidden modules: {forbidden}"
