"""Deprecated shim — use read_data.halos.base."""

import warnings

from density_field_properties.read_data.halos.base import HaloCatalogData, HaloCatalogReader

warnings.warn(
    "halo_catalog.halo_catalog is deprecated; use density_field_properties.read_data.halos.base",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HaloCatalogData", "HaloCatalogReader"]
