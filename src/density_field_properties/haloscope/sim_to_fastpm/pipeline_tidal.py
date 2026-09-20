"""Deprecated shim — use density_field_properties.pipelines.haloscope_enrichment."""

import warnings

from density_field_properties.pipelines.haloscope_enrichment_tidal import (
    run_haloscope_enrichment_tidal_pipeline,
    write_tidal_assembly_bias_panel,
)

warnings.warn(
    "haloscope.sim_to_fastpm.pipeline_tidal is deprecated; "
    "use density_field_properties.pipelines.haloscope_enrichment",
    DeprecationWarning,
    stacklevel=2,
)

run_sim_to_fastpm_haloscope_tidal_pipeline = run_haloscope_enrichment_tidal_pipeline

__all__ = [
    "run_sim_to_fastpm_haloscope_tidal_pipeline",
    "write_tidal_assembly_bias_panel",
]
