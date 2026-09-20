"""Pipeline orchestration for Haloscope SIM-to-FastPM enrichment."""

from density_field_properties.pipelines.config import (
    DEFAULT_ENV_PRODUCTION_CONFIG,
    DEFAULT_ENV_SMOKE_CONFIG,
    DEFAULT_TIDAL_PRODUCTION_CONFIG,
    DEFAULT_TIDAL_SMOKE_CONFIG,
    HaloscopeEnrichmentConfig,
    load_haloscope_enrichment_config,
)

__all__ = [
    "DEFAULT_ENV_PRODUCTION_CONFIG",
    "DEFAULT_ENV_SMOKE_CONFIG",
    "DEFAULT_TIDAL_PRODUCTION_CONFIG",
    "DEFAULT_TIDAL_SMOKE_CONFIG",
    "HaloscopeEnrichmentConfig",
    "HaloscopeEnrichmentRun",
    "load_haloscope_enrichment_config",
    "run_haloscope_enrichment_pipeline",
    "write_tidal_assembly_bias_panel",
]


def __getattr__(name: str):
    """
    Lazily import pipeline entrypoints and validation helpers.

    Parameters
    ----------
    name : str
        Requested attribute name.

    Returns
    -------
    object
        Exported pipeline symbol.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public API.
    """
    if name == "HaloscopeEnrichmentRun":
        from density_field_properties.pipelines.haloscope_enrichment import (
            HaloscopeEnrichmentRun,
        )

        return HaloscopeEnrichmentRun
    if name == "run_haloscope_enrichment_pipeline":
        from density_field_properties.pipelines.haloscope_enrichment import (
            run_haloscope_enrichment_pipeline,
        )

        return run_haloscope_enrichment_pipeline
    if name == "write_tidal_assembly_bias_panel":
        from density_field_properties.validation.assembly_bias_panel import (
            write_tidal_assembly_bias_panel,
        )

        return write_tidal_assembly_bias_panel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
