"""Deprecated shim — use read_data.halos.rockstar."""

import warnings

from density_field_properties.read_data.halos.rockstar import (
    ROCKSTAR_HALO_COLUMNS_POSITION,
    RockstarCatalogReader,
    read_rockstar_cosmology_header,
)

warnings.warn(
    "halo_catalog.rockstar is deprecated; use density_field_properties.read_data.halos.rockstar",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ROCKSTAR_HALO_COLUMNS_POSITION",
    "RockstarCatalogReader",
    "read_rockstar_cosmology_header",
]
