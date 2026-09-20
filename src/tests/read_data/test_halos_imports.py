"""Import checks for read_data.halos modules."""

import importlib


def test_read_data_public_symbols_are_importable():
    """
    Canonical read_data halo symbols must resolve without legacy shims.
    """
    module = importlib.import_module("density_field_properties.read_data")
    for name in ("HaloCatalogData", "HaloCatalogReader", "RockstarCatalogReader"):
        assert hasattr(module, name), f"read_data missing {name}"


def test_read_data_halos_symbols_are_importable():
    """
    read_data.halos must expose the halo reader API.
    """
    module = importlib.import_module("density_field_properties.read_data.halos")
    for name in ("HaloCatalogData", "HaloCatalogReader", "RockstarCatalogReader"):
        assert hasattr(module, name), f"read_data.halos missing {name}"


def test_registry_symbols_are_importable():
    """
    Registry helpers must import without optional FastPM dependencies.
    """
    module = importlib.import_module("density_field_properties.read_data.halos.registry")
    assert hasattr(module, "get_halo_catalog_reader")
    assert hasattr(module, "HaloCatalogError")
