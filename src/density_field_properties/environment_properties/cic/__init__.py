"""CIC density deposition and grid metadata."""

from density_field_properties.environment_properties.cic.utils import (
    DensityFieldInfo,
    gaussian_filter,
    get_grid_cell,
)

__all__ = [
    "DensityFieldInfo",
    "delta_field_from_dm_particles",
    "delta_field_from_saved_cic",
    "density_field_cic_main",
    "gaussian_filter",
    "get_delta_density",
    "get_grid_cell",
    "load_density_field_cic",
    "mass_field_cic",
    "overdensity_from_cic_grid",
    "save_density_field_cic",
    "weighted_field_cic",
]


def __getattr__(name):
    """
    Lazily import CIC deposit helpers so tidal-only workflows avoid particle I/O deps.
    """
    cic_deposit_symbols = {
        "delta_field_from_dm_particles",
        "delta_field_from_saved_cic",
        "density_field_cic_main",
        "get_delta_density",
        "load_density_field_cic",
        "mass_field_cic",
        "overdensity_from_cic_grid",
        "save_density_field_cic",
        "weighted_field_cic",
    }
    if name in cic_deposit_symbols:
        from density_field_properties.environment_properties.cic import cic_deposit

        return getattr(cic_deposit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
