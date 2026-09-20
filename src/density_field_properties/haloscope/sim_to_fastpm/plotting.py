"""Deprecated shim — use density_field_properties.utils and validation."""

import warnings

from density_field_properties.utils.plotting import compare_2d_contours
from density_field_properties.utils.stats import (
    bin_midpoints,
    confidence_intervals,
    median_property_vs_mass,
)
from density_field_properties.validation.marginals import corner_plot_sim_validation
from density_field_properties.validation.plots import plot_assembly_bias_env_panel

warnings.warn(
    "haloscope.sim_to_fastpm.plotting is deprecated; "
    "use density_field_properties.utils and validation",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "bin_midpoints",
    "compare_2d_contours",
    "confidence_intervals",
    "corner_plot_sim_validation",
    "median_property_vs_mass",
    "plot_assembly_bias_env_panel",
]
