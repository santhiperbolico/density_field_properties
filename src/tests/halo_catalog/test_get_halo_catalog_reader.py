import pytest

from density_field_properties.halo_catalog.fastpm import FastPMCatalogReader
from density_field_properties.halo_catalog.rockstar import RockstarCatalogReader
from density_field_properties.halo_catalog.utils import HaloCatalogError, get_halo_catalog_reader


def test_get_catalog_reader_valid():
    assert get_halo_catalog_reader("fastpm") is FastPMCatalogReader
    assert get_halo_catalog_reader("rockstar") is RockstarCatalogReader


def test_get_catalog_reader_invalid():
    with pytest.raises(HaloCatalogError):
        get_halo_catalog_reader("invalid_name")
