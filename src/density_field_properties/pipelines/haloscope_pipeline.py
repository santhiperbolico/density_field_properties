"""Multi-stage Haloscope pipeline orchestration."""

import gc
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    HaloscopeEnrichmentConfig,
    TidalAnisotropyTargetConfig,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    HaloscopeEnrichmentRun,
    build_haloscope_feature_tables,
    run_haloscope_enrichment_from_tables,
    run_haloscope_enrichment_pipeline,
)
from density_field_properties.pipelines.tidal_anisotropy_run import (
    run_tidal_anisotropy_pipeline_for_target,
    tidal_anisotropy_output_dir,
)
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    TIDAL_ANISOTROPY_FEATURE_NAME,
)


def _target_settings(
    config: HaloscopeEnrichmentConfig,
    target_name: str,
) -> TidalAnisotropyTargetConfig:
    """
    Return tidal-anisotropy settings for a pipeline target.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.
    target_name : str
        Canonical target name (``unit`` or ``fastpm``).

    Returns
    -------
    TidalAnisotropyTargetConfig
        Target-specific tidal settings.
    """
    stage = config.pipeline_stages.tidal_anisotropy
    if target_name == PIPELINE_TARGET_UNIT:
        return stage.unit
    if target_name == PIPELINE_TARGET_FASTPM:
        return stage.fastpm
    raise ValueError(f"Unsupported tidal-anisotropy target: {target_name}")


def _halo_catalog_path(config: HaloscopeEnrichmentConfig, target_name: str) -> Path:
    """
    Resolve the halo catalog path for a tidal-anisotropy target.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.
    target_name : str
        Canonical target name (``unit`` or ``fastpm``).

    Returns
    -------
    Path
        Halo catalog path for descriptor generation.
    """
    if target_name == PIPELINE_TARGET_UNIT:
        return Path(config.sim_hlist_path)
    return Path(config.fastpm_list_path)


def _apply_computed_descriptor_dirs(config: HaloscopeEnrichmentConfig) -> None:
    """
    Point descriptor directories to pipeline work directories when unset.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration updated in place when descriptor dirs are unset.
    """
    unit_work_dir = config.pipeline_stages.tidal_anisotropy.unit.work_dir
    fastpm_work_dir = config.pipeline_stages.tidal_anisotropy.fastpm.work_dir
    if config.unit_descriptors_dir is None and unit_work_dir is not None:
        config.unit_descriptors_dir = tidal_anisotropy_output_dir(unit_work_dir)
    if config.fastpm_descriptors_dir is None and fastpm_work_dir is not None:
        config.fastpm_descriptors_dir = tidal_anisotropy_output_dir(fastpm_work_dir)


def _resolve_descriptor_dirs_from_pipeline(config: HaloscopeEnrichmentConfig) -> None:
    """
    Resolve tidal descriptor directories from pipeline target work dirs.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration updated in place when tidal features are requested.
    """
    if TIDAL_ANISOTROPY_FEATURE_NAME not in config.input_features:
        return
    _apply_computed_descriptor_dirs(config)


def run_tidal_anisotropy_stage(config: HaloscopeEnrichmentConfig) -> None:
    """
    Run the tidal-anisotropy stage for the configured simulation targets.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration with ``pipeline_stages.tidal_anisotropy`` enabled.
    """
    stage = config.pipeline_stages.tidal_anisotropy
    box_size = config.box_size_mpc_h
    if box_size is None:
        raise ValueError("box_size_mpc_h must be set when running the tidal-anisotropy stage.")

    for target_name in stage.targets:
        target = _target_settings(config, target_name)
        if target.dm_particles_file is None:
            raise ValueError(f"dm_particles_file is required for tidal target '{target_name}'.")
        if target.work_dir is None:
            raise ValueError(f"work_dir is required for tidal target '{target_name}'.")

        halo_catalog_path = _halo_catalog_path(config, target_name)
        max_halos = (
            config.max_sim_halos
            if target_name == PIPELINE_TARGET_UNIT
            else config.max_fastpm_halos
        )
        run_tidal_anisotropy_pipeline_for_target(
            target_name=target_name,
            dm_particles_file=Path(target.dm_particles_file),
            halo_catalog_path=halo_catalog_path,
            work_dir=Path(target.work_dir),
            box_size=float(box_size),
            n_grid=config.tidal_n_grid,
            mass_particle=stage.mass_particle,
            catalog_layout=target.catalog_layout,
            cic_batch_size=stage.cic_batch_size,
            descriptor_batch_size=stage.descriptor_batch_size,
            max_halos=max_halos,
            read_tensor_from_disk=target.read_tensor_from_disk,
        )
        gc.collect()


def _needs_feature_tables(config: HaloscopeEnrichmentConfig) -> bool:
    """
    Return whether HR/LR feature tables must be built for this run.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.

    Returns
    -------
    bool
        True when preprocess, Haloscope, or assembly-bias stages need tables.
    """
    stages = config.pipeline_stages
    return stages.preprocess.enabled or stages.haloscope.enabled or config.run_assembly_bias_plot


def _save_preprocessed_tables(
    config: HaloscopeEnrichmentConfig,
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Persist preprocessed HR/LR tables to Parquet files.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.
    halos_sim : pd.DataFrame
        HR table with attached INPUT features.
    halos_fastpm : pd.DataFrame
        LR table with attached INPUT features.

    Returns
    -------
    tuple[Path, Path]
        ``(hr_parquet, lr_parquet)`` output paths.
    """
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hr_path = out_dir / config.pipeline_stages.preprocess.hr_parquet_name
    lr_path = out_dir / config.pipeline_stages.preprocess.lr_parquet_name
    halos_sim.to_parquet(hr_path, index=False)
    halos_fastpm.to_parquet(lr_path, index=False)
    logging.info("Preprocessed HR table written to %s", hr_path)
    logging.info("Preprocessed LR table written to %s", lr_path)
    return hr_path, lr_path


def run_haloscope_pipeline(
    config: HaloscopeEnrichmentConfig,
) -> Union[Path, HaloscopeEnrichmentRun, tuple[Path, Path], None]:
    """
    Execute the configured multi-stage Haloscope pipeline.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    Path, HaloscopeEnrichmentRun, tuple[Path, Path], or None
        Enriched Parquet path, run object, preprocess outputs, or ``None`` when
        only the tidal-anisotropy stage ran.
    """
    stages = config.pipeline_stages
    if stages.tidal_anisotropy.enabled:
        run_tidal_anisotropy_stage(config)

    _resolve_descriptor_dirs_from_pipeline(config)

    tables: Optional[tuple[pd.DataFrame, pd.DataFrame, str]] = None
    preprocess_paths: Optional[tuple[Path, Path]] = None

    if _needs_feature_tables(config):
        halos_sim, halos_fastpm, mass_col_fastpm = build_haloscope_feature_tables(config)
        tables = (halos_sim, halos_fastpm, mass_col_fastpm)
        if stages.preprocess.enabled:
            preprocess_paths = _save_preprocessed_tables(config, halos_sim, halos_fastpm)

    if not stages.haloscope.enabled:
        if config.run_assembly_bias_plot and tables is not None:
            return run_haloscope_enrichment_from_tables(
                config,
                tables[0],
                tables[1],
                tables[2],
                run_enrichment=False,
            )
        return preprocess_paths

    if tables is None:
        return run_haloscope_enrichment_pipeline(config)

    return run_haloscope_enrichment_from_tables(
        config,
        tables[0],
        tables[1],
        tables[2],
        run_enrichment=True,
    )
