"""Read-data layer for raw DM particles and halo catalogs."""

from density_field_properties.read_data.halos import (
    HaloCatalogData,
    HaloCatalogReader,
    RockstarCatalogReader,
)

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
    Re-export optional halo reader symbols without importing bigfile at package load.
    """
    if name in {
        "FastPMCatalogReader",
        "HaloCatalogError",
        "get_halo_catalog_reader",
    }:
        from density_field_properties.read_data import halos

        return getattr(halos, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
