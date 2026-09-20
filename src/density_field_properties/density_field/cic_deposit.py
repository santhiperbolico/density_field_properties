"""Deprecated shim — use environment_properties.cic.cic_deposit."""

import warnings

from density_field_properties.environment_properties.cic.cic_deposit import (
    delta_field_from_dm_particles,
    delta_field_from_saved_cic,
    density_field_cic_main,
    get_delta_density,
    load_density_field_cic,
    mass_field_cic,
    overdensity_from_cic_grid,
    save_density_field_cic,
    weighted_field_cic,
)

warnings.warn(
    "density_field.cic_deposit is deprecated; "
    "use density_field_properties.environment_properties.cic.cic_deposit",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "delta_field_from_dm_particles",
    "delta_field_from_saved_cic",
    "density_field_cic_main",
    "get_delta_density",
    "load_density_field_cic",
    "mass_field_cic",
    "overdensity_from_cic_grid",
    "save_density_field_cic",
    "weighted_field_cic",
]
