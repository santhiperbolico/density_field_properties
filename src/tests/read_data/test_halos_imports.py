"""Import checks for read_data.halos and legacy halo_catalog shims."""

import importlib
import warnings

import pytest


@pytest.mark.parametrize(
    "module_path,symbol_names",
    [
        (
            "density_field_properties.read_data",
            [
                "HaloCatalogData",
                "HaloCatalogReader",
                "RockstarCatalogReader",
            ],
        ),
        (
            "density_field_properties.read_data.halos",
            [
                "HaloCatalogData",
                "HaloCatalogReader",
                "RockstarCatalogReader",
            ],
        ),
    ],
)
def test_read_data_public_symbols_are_importable(module_path, symbol_names):
    """
    Each listed symbol must resolve on the canonical read_data modules.
    """
    module = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(module, name), f"{module_path} missing {name}"


def test_registry_symbols_are_importable():
    """
    Registry helpers must import without optional FastPM dependencies.
    """
    module = importlib.import_module("density_field_properties.read_data.halos.registry")
    assert hasattr(module, "get_halo_catalog_reader")
    assert hasattr(module, "HaloCatalogError")


def test_legacy_halo_catalog_shim_emits_deprecation_warning():
    """
    Legacy halo_catalog imports must warn but remain callable.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("density_field_properties.halo_catalog.utils")
        assert any(
            issubclass(item.category, DeprecationWarning) for item in caught
        ), "Expected DeprecationWarning from legacy halo_catalog shim"
        assert callable(module.get_halo_catalog_reader)
