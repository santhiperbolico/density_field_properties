"""Deprecated shim — use read_data.halos.fastpm."""

import warnings

from density_field_properties.read_data.halos.fastpm import (
    FASTPM_HALO_COLUMNS_POSITION,
    MSUN_G,
    FastPMCatalogReader,
    read_fastpm_cosmology_header,
)

warnings.warn(
    "halo_catalog.fastpm is deprecated; use density_field_properties.read_data.halos.fastpm",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FASTPM_HALO_COLUMNS_POSITION",
    "MSUN_G",
    "FastPMCatalogReader",
    "read_fastpm_cosmology_header",
]
