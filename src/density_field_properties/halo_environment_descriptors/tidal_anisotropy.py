"""Deprecated shim — use environment_properties.tidal_anisotropy."""

import warnings

from density_field_properties.environment_properties.tidal_anisotropy.tidal_anisotropy import (
    ANISOTROPY_PATH,
    _tidal_anisotropy_and_overdensity_from_halo_calaog_batches,
    _tidal_anisotropy_and_overdensity_from_halo_calaog_complete,
    format_halo_catalog,
    tidal_anisotropy_and_overdensity_from_halo_calaog,
)

warnings.warn(
    "halo_environment_descriptors.tidal_anisotropy is deprecated; "
    "use density_field_properties.environment_properties.tidal_anisotropy",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ANISOTROPY_PATH",
    "_tidal_anisotropy_and_overdensity_from_halo_calaog_batches",
    "_tidal_anisotropy_and_overdensity_from_halo_calaog_complete",
    "format_halo_catalog",
    "tidal_anisotropy_and_overdensity_from_halo_calaog",
]
