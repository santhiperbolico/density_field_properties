"""Deprecated shim — use density_field_properties.haloscope.model."""

import warnings

from density_field_properties.haloscope.model import ConditionalMultiVariateGaussian

warnings.warn(
    "haloscope.haloscope is deprecated; use density_field_properties.haloscope.model",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ConditionalMultiVariateGaussian"]
