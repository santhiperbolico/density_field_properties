"""Deprecated shim — use density_field_properties.pipelines.haloscope_enrichment."""

import warnings
from pathlib import Path
from typing import Optional, Union

from density_field_properties.haloscope.sim_to_fastpm.config import (
    ENRICHED_TIDAL_PARQUET_NAME,
    OUTPUT_DIR_TIDAL,
    TIDAL_DENSITY_N_GRID,
    TIDAL_INPUT_FEATURES,
    default_fastpm_tidal_descriptors_dir,
    default_unit_tidal_descriptors_dir,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    HaloscopeEnrichmentRun,
    run_haloscope_enrichment_pipeline,
    write_tidal_assembly_bias_panel,
)

warnings.warn(
    "pipelines.haloscope_enrichment_tidal is deprecated; "
    "use density_field_properties.pipelines.haloscope_enrichment",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HaloscopeEnrichmentRun",
    "run_haloscope_enrichment_tidal_pipeline",
    "write_tidal_assembly_bias_panel",
]


def run_haloscope_enrichment_tidal_pipeline(
    sim_hlist_path: Union[str, Path],
    fastpm_list_path: Union[str, Path],
    repo_root: Union[str, Path],
    max_sim_halos: Optional[int] = None,
    max_fastpm_halos: Optional[int] = None,
    max_descriptor_batch_files: Optional[int] = None,
    output_dir: Optional[Path] = None,
    min_bin_size: int = 10,
    run_holdout_validation: bool = True,
    unit_descriptors_dir: Optional[Path] = None,
    fastpm_descriptors_dir: Optional[Path] = None,
    n_grid: int = TIDAL_DENSITY_N_GRID,
    run_assembly_bias_plot: bool = False,
    assembly_bias_n_grid: int = 128,
) -> Path:
    """
    Deprecated wrapper around ``run_haloscope_enrichment_pipeline``.

    Parameters
    ----------
    sim_hlist_path : str or Path
        Path to the UNIT consistent-trees hlist.
    fastpm_list_path : str or Path
        Path to the FastPM Rockstar ``out_*.list`` catalog.
    repo_root : str or Path
        Repository root used to resolve relative descriptor directories.
    max_sim_halos : Optional[int], optional
        Cap rows read from the UNIT hlist.
    max_fastpm_halos : Optional[int], optional
        Cap halos read from the FastPM ``.list`` file.
    max_descriptor_batch_files : Optional[int], optional
        Smoke cap on tidal descriptor batch files per simulation.
    output_dir : Optional[Path], optional
        Output directory for the enriched Parquet file.
    min_bin_size : int, optional
        Minimum halos per mass bin for fit and validation.
    run_holdout_validation : bool, optional
        If True, run SIM hold-out validation before enrichment.
    unit_descriptors_dir : Optional[Path], optional
        UNIT tidal descriptor directory.
    fastpm_descriptors_dir : Optional[Path], optional
        FastPM tidal descriptor directory.
    n_grid : int, optional
        Grid resolution used when tidal descriptors were computed.
    run_assembly_bias_plot : bool, optional
        If True, write ``assembly_bias_tidal_input.pdf`` after enrichment.
    assembly_bias_n_grid : int, optional
        Grid resolution for the Paranjape assembly-bias diagnostic.

    Returns
    -------
    Path
        Path to the enriched FastPM Parquet file.
    """
    warnings.warn(
        "run_haloscope_enrichment_tidal_pipeline is deprecated; "
        "call run_haloscope_enrichment_pipeline with tidal input_features",
        DeprecationWarning,
        stacklevel=2,
    )
    root = Path(repo_root)
    unit_descriptor_path = (
        default_unit_tidal_descriptors_dir()
        if unit_descriptors_dir is None
        else Path(unit_descriptors_dir)
    )
    fastpm_descriptor_path = (
        default_fastpm_tidal_descriptors_dir()
        if fastpm_descriptors_dir is None
        else Path(fastpm_descriptors_dir)
    )
    out_dir = OUTPUT_DIR_TIDAL if output_dir is None else Path(output_dir)

    result = run_haloscope_enrichment_pipeline(
        sim_hlist_path=sim_hlist_path,
        fastpm_list_path=fastpm_list_path,
        max_sim_halos=max_sim_halos,
        max_fastpm_halos=max_fastpm_halos,
        output_dir=out_dir,
        min_bin_size=min_bin_size,
        run_holdout_validation=run_holdout_validation,
        input_features=TIDAL_INPUT_FEATURES,
        repo_root=root,
        unit_descriptors_dir=unit_descriptor_path,
        fastpm_descriptors_dir=fastpm_descriptor_path,
        tidal_n_grid=n_grid,
        max_descriptor_batch_files=max_descriptor_batch_files,
        enriched_parquet_name=ENRICHED_TIDAL_PARQUET_NAME,
        run_assembly_bias_plot=run_assembly_bias_plot,
        assembly_bias_n_grid=assembly_bias_n_grid,
        collect_tables=False,
    )
    if isinstance(result, HaloscopeEnrichmentRun):
        return result.output_path
    return result
