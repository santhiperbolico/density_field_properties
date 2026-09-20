"""Deprecated shim — use preprocessing.input_features.n_halos_env."""

import warnings

from density_field_properties.preprocessing.environment_join import local_environment

warnings.warn(
    "haloscope.sim_to_fastpm.environment is deprecated; "
    "use density_field_properties.preprocessing.input_features.n_halos_env",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["local_environment"]
