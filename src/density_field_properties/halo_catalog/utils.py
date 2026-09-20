"""Deprecated shim — use read_data.halos.registry."""

import warnings

from density_field_properties.read_data.halos.registry import (
    HaloCatalogError,
    get_halo_catalog_reader,
)

warnings.warn(
    "halo_catalog.utils is deprecated; use density_field_properties.read_data.halos.registry",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HaloCatalogError", "get_halo_catalog_reader"]
