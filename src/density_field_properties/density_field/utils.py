"""Deprecated shim — use environment_properties.cic.utils."""

import warnings

from density_field_properties.environment_properties.cic.utils import (
    DensityFieldInfo,
    gaussian_filter,
    get_grid_cell,
)

warnings.warn(
    "density_field.utils is deprecated; "
    "use density_field_properties.environment_properties.cic.utils",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DensityFieldInfo", "gaussian_filter", "get_grid_cell"]
