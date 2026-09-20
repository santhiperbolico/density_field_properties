"""Deprecated shim — use environment_properties.fourier.fourrier_transformations."""

import warnings

from density_field_properties.environment_properties.fourier.fourrier_transformations import kgrid

warnings.warn(
    "density_field.fourrier_transformations is deprecated; "
    "use density_field_properties.environment_properties.fourier.fourrier_transformations",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["kgrid"]
