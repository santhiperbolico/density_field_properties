"""Build and persist preprocessed HR/LR feature tables for the Haloscope pipeline."""

import logging
from pathlib import Path

import pandas as pd

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    HaloscopeEnrichmentConfig,
    mass_correction_settings_from_preprocessing,
    resolve_descriptor_dir_for_target,
    tidal_target_by_name,
)
from density_field_properties.pipelines.run_defaults import ENV_RADIUS_MPC_H, INPUT_FEATURES
from density_field_properties.preprocessing.catalog_loaders import (
    load_fastpm_target_catalog,
    load_unit_sim_training_catalog,
)
from density_field_properties.preprocessing.context import build_preprocessing_context
from density_field_properties.preprocessing.feature_table import build_feature_tables
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    input_features_need_tidal_descriptors,
)


def _resolve_tidal_descriptor_dirs(
    config: HaloscopeEnrichmentConfig,
) -> tuple[Path | None, Path | None]:
    """
    Resolve UNIT and FastPM tidal descriptor directories for preprocessing.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    tuple[Path | None, Path | None]
        UNIT and FastPM descriptor directories.
    """
    unit_dir = resolve_descriptor_dir_for_target(config, PIPELINE_TARGET_UNIT)
    fastpm_dir = resolve_descriptor_dir_for_target(config, PIPELINE_TARGET_FASTPM)
    if unit_dir is None:
        unit_target = tidal_target_by_name(config, PIPELINE_TARGET_UNIT)
        if unit_target is not None and unit_target.work_dir is not None:
            unit_dir = unit_target.work_dir / "tidal_anisotropy"
    if fastpm_dir is None:
        fastpm_target = tidal_target_by_name(config, PIPELINE_TARGET_FASTPM)
        if fastpm_target is not None and fastpm_target.work_dir is not None:
            fastpm_dir = fastpm_target.work_dir / "tidal_anisotropy"
    return unit_dir, fastpm_dir


def build_haloscope_feature_tables(
    config: HaloscopeEnrichmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load catalogs, attach INPUT features, and apply LR mass correction.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, str]
        HR table, LR table, and LR mass column used for bin assignment.
    """
    sim_path = Path(config.sim_hlist_path)
    fastpm_path = Path(config.fastpm_list_path)
    if not sim_path.is_file():
        raise FileNotFoundError(f"SIM hlist not found: {sim_path}")
    if not fastpm_path.is_file():
        raise FileNotFoundError(f"FastPM catalog not found: {fastpm_path}")

    feature_names = list(config.input_features) if config.input_features else list(INPUT_FEATURES)
    root = Path(config.repo_root)
    unit_descriptor_path, fastpm_descriptor_path = _resolve_tidal_descriptor_dirs(config)
    if input_features_need_tidal_descriptors(feature_names):
        config.general.unit_descriptors_dir = unit_descriptor_path
        config.general.fastpm_descriptors_dir = fastpm_descriptor_path

    sample = config.pipeline.preprocessing.sample
    box_size = float(sample.box_size_mpc_h)
    halos_sim = load_unit_sim_training_catalog(sim_path, max_halos=sample.max_sim_halos)
    halos_fastpm = load_fastpm_target_catalog(
        fastpm_path,
        max_halos=sample.max_fastpm_halos,
    )
    mass_correction = mass_correction_settings_from_preprocessing(config.pipeline.preprocessing)
    preprocessing_context = build_preprocessing_context(
        repo_root=root,
        mass_correction=mass_correction,
        sim_boxsize_mpc_h=box_size,
        fastpm_boxsize_mpc_h=box_size,
        env_radius_mpc_h=ENV_RADIUS_MPC_H,
        unit_descriptors_dir=unit_descriptor_path,
        fastpm_descriptors_dir=fastpm_descriptor_path,
        tidal_n_grid=config.tidal_n_grid,
        max_descriptor_batch_files=None,
        mass_filter=config.mass_filter,
    )
    return build_feature_tables(
        halos_sim,
        halos_fastpm,
        feature_names,
        preprocessing_context,
    )


def save_preprocessed_tables(
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
    prep = config.pipeline.preprocessing
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hr_path = out_dir / prep.hr_parquet_name
    lr_path = out_dir / prep.lr_parquet_name
    halos_sim.to_parquet(hr_path, index=False)
    halos_fastpm.to_parquet(lr_path, index=False)
    logging.info("Preprocessed HR table written to %s", hr_path)
    logging.info("Preprocessed LR table written to %s", lr_path)
    return hr_path, lr_path
