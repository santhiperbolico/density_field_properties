import pytest

from density_field_properties.read_data.halos.registry import (
    HaloCatalogError,
    get_halo_catalog_reader,
)
from density_field_properties.read_data.halos.rockstar import RockstarCatalogReader


def test_get_catalog_reader_rockstar():
    assert get_halo_catalog_reader("rockstar") is RockstarCatalogReader


def test_get_catalog_reader_fastpm():
    from density_field_properties.read_data.halos.fastpm import FastPMCatalogReader

    assert get_halo_catalog_reader("fastpm") is FastPMCatalogReader


def test_get_catalog_reader_invalid():
    with pytest.raises(HaloCatalogError):
        get_halo_catalog_reader("invalid_name")
