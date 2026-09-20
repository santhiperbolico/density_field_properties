"""Deprecated shim — use density_field_properties.preprocessing."""

import warnings

from density_field_properties.preprocessing.filters import filter_finite_input_features
from density_field_properties.preprocessing.tidal_join import (
    attach_tidal_anisotropy,
    load_halo_environment_descriptor_table,
)

warnings.warn(
    "haloscope.sim_to_fastpm.tidal_features is deprecated; "
    "use density_field_properties.preprocessing.input_features.tidal_anisotropy",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "attach_tidal_anisotropy",
    "filter_finite_input_features",
    "load_halo_environment_descriptor_table",
]
