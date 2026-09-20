"""Deprecated shim — use density_field_properties.preprocessing.mass_calibration."""

import warnings

from density_field_properties.preprocessing.mass_calibration import abundance_match_mass

warnings.warn(
    "haloscope.sim_to_fastpm.mass_matching is deprecated; "
    "use density_field_properties.preprocessing.mass_calibration",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["abundance_match_mass"]
