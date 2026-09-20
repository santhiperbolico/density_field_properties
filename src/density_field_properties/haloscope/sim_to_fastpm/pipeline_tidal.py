"""Deprecated shim — use pipelines.haloscope_enrichment with JSON config."""

import warnings
from pathlib import Path
from typing import Optional, Union

from density_field_properties.pipelines.config import (
    HaloscopeEnrichmentConfig,
    load_haloscope_enrichment_config,
    tidal_production_config,
    tidal_smoke_config,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    HaloscopeEnrichmentRun,
    run_haloscope_enrichment_pipeline,
)
from density_field_properties.validation.assembly_bias_panel import (
    write_tidal_assembly_bias_panel,
)

warnings.warn(
    "haloscope.sim_to_fastpm.pipeline_tidal is deprecated; "
    "use pipelines/run_haloscope_enrichment.py with a tidal JSON config",
    DeprecationWarning,
    stacklevel=2,
)


def run_sim_to_fastpm_haloscope_tidal_pipeline(
    config: Optional[HaloscopeEnrichmentConfig] = None,
    *,
    config_path: Optional[Path] = None,
    smoke: bool = False,
) -> Union[Path, HaloscopeEnrichmentRun]:
    """
    Run the Haloscope enrichment pipeline with tidal INPUT features.

    Parameters
    ----------
    config : Optional[HaloscopeEnrichmentConfig], optional
        Explicit run configuration. When omitted, ``config_path`` or the
        tidal preset defaults are used.
    config_path : Optional[Path], optional
        JSON configuration file. Ignored when ``config`` is provided.
    smoke : bool, optional
        When no explicit config is given, select the tidal smoke preset.

    Returns
    -------
    Path or HaloscopeEnrichmentRun
        Enriched Parquet path, or a run object when tables are collected.
    """
    if config is not None:
        return run_haloscope_enrichment_pipeline(config)

    if config_path is not None:
        return run_haloscope_enrichment_pipeline(load_haloscope_enrichment_config(config_path))

    if smoke:
        return run_haloscope_enrichment_pipeline(tidal_smoke_config())

    return run_haloscope_enrichment_pipeline(tidal_production_config())


__all__ = [
    "run_sim_to_fastpm_haloscope_tidal_pipeline",
    "write_tidal_assembly_bias_panel",
]
