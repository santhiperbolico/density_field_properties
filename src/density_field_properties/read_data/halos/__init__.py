"""Halo catalog readers for Rockstar and FastPM formats."""

from density_field_properties.read_data.halos.base import HaloCatalogData, HaloCatalogReader
from density_field_properties.read_data.halos.rockstar import RockstarCatalogReader

__all__ = [
    "FastPMCatalogReader",
    "HaloCatalogData",
    "HaloCatalogError",
    "HaloCatalogReader",
    "RockstarCatalogReader",
    "get_halo_catalog_reader",
]


def __getattr__(name):
    """
    Lazily import FastPM and registry symbols so Rockstar-only workflows avoid bigfile.
    """
    if name == "FastPMCatalogReader":
        from density_field_properties.read_data.halos.fastpm import FastPMCatalogReader

        return FastPMCatalogReader
    if name == "HaloCatalogError":
        from density_field_properties.read_data.halos.registry import HaloCatalogError

        return HaloCatalogError
    if name == "get_halo_catalog_reader":
        from density_field_properties.read_data.halos.registry import get_halo_catalog_reader

        return get_halo_catalog_reader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
