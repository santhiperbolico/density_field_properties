"""Deprecated shim — use density_field_properties.pipelines.haloscope_enrichment."""

import warnings

from density_field_properties.pipelines.haloscope_enrichment import (
    run_haloscope_enrichment_pipeline,
)

warnings.warn(
    "haloscope.sim_to_fastpm.pipeline is deprecated; "
    "use density_field_properties.pipelines.haloscope_enrichment",
    DeprecationWarning,
    stacklevel=2,
)

run_sim_to_fastpm_haloscope_pipeline = run_haloscope_enrichment_pipeline

__all__ = ["run_sim_to_fastpm_haloscope_pipeline"]
