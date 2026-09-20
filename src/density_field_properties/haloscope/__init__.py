"""Haloscope CMVG model and fit/predict helpers.

The ``ConditionalMultiVariateGaussian`` implementation is vendored from
https://github.com/computationalAstroUAM/haloscope (see ``model.py``).
"""

from density_field_properties.haloscope.bins import default_mass_bin_edges, mask_mass_bin
from density_field_properties.haloscope.model import ConditionalMultiVariateGaussian
from density_field_properties.haloscope.predict import enrich_fastpm_catalog, predict_models
from density_field_properties.haloscope.training import fit_models, holdout_validate_sim_bins

__all__ = [
    "ConditionalMultiVariateGaussian",
    "default_mass_bin_edges",
    "enrich_fastpm_catalog",
    "fit_models",
    "holdout_validate_sim_bins",
    "mask_mass_bin",
    "predict_models",
]
