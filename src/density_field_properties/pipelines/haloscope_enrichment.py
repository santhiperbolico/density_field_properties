"""Haloscope SIM-to-FastPM enrichment orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from density_field_properties.haloscope import (
    default_mass_bin_edges,
    enrich_fastpm_catalog,
    holdout_validate_sim_bins,
)
from density_field_properties.pipelines.config import HaloscopeEnrichmentConfig
from density_field_properties.pipelines.run_defaults import (
    CALIBRATE_MASS,
    ENV_RADIUS_MPC_H,
    FASTPM_BOXSIZE_MPC_H,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    SIM_BOXSIZE_MPC_H,
    default_fastpm_tidal_descriptors_dir,
    default_unit_tidal_descriptors_dir,
)
from density_field_properties.preprocessing.catalog_loaders import (
    load_fastpm_target_catalog,
    load_unit_sim_training_catalog,
)
from density_field_properties.preprocessing.context import build_preprocessing_context
from density_field_properties.preprocessing.feature_table import build_feature_tables
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    TIDAL_ANISOTROPY_FEATURE_NAME,
)
from density_field_properties.validation.assembly_bias_panel import (
    write_tidal_assembly_bias_panel,
)


@dataclass
class HaloscopeEnrichmentRun:
    """
    Result of a Haloscope enrichment pipeline execution.

    Parameters
    ----------
    output_path : Path
        Path to the enriched LR Parquet file.
    hr_table : Optional[pd.DataFrame]
        Preprocessed HR table when ``collect_tables=True``.
    enriched_lr_table : Optional[pd.DataFrame]
        Enriched LR table when ``collect_tables=True``.
    mass_column_fastpm : Optional[str]
        LR mass column used for bin assignment when ``collect_tables=True``.
    """

    output_path: Path
    hr_table: Optional[pd.DataFrame] = None
    enriched_lr_table: Optional[pd.DataFrame] = None
    mass_column_fastpm: Optional[str] = None


def _resolve_tidal_descriptor_dirs(
    feature_names: Sequence[str],
    unit_descriptors_dir: Optional[Path],
    fastpm_descriptors_dir: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Apply config defaults for tidal descriptor directories when needed.

    Parameters
    ----------
    feature_names : Sequence[str]
        Requested Haloscope INPUT feature names.
    unit_descriptors_dir : Optional[Path]
        UNIT descriptor directory override.
    fastpm_descriptors_dir : Optional[Path]
        FastPM descriptor directory override.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        Resolved UNIT and FastPM descriptor directories.
    """
    if TIDAL_ANISOTROPY_FEATURE_NAME not in feature_names:
        return unit_descriptors_dir, fastpm_descriptors_dir

    unit_path = unit_descriptors_dir
    fastpm_path = fastpm_descriptors_dir
    if unit_path is None:
        unit_path = default_unit_tidal_descriptors_dir()
    if fastpm_path is None:
        fastpm_path = default_fastpm_tidal_descriptors_dir()
    return unit_path, fastpm_path


def run_haloscope_enrichment_pipeline(
    config: HaloscopeEnrichmentConfig,
) -> Union[Path, HaloscopeEnrichmentRun]:
    """
    Run Haloscope enrichment on FastPM using a high-resolution SIM as training data.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    Path or HaloscopeEnrichmentRun
        Enriched Parquet path, or a run object when tables are collected.
    """
    sim_path = Path(config.sim_hlist_path)
    fastpm_path = Path(config.fastpm_list_path)
    if not sim_path.is_file():
        raise FileNotFoundError(f"SIM hlist not found: {sim_path}")
    if not fastpm_path.is_file():
        raise FileNotFoundError(f"FastPM catalog not found: {fastpm_path}")

    feature_names = config.input_features if config.input_features else INPUT_FEATURES
    root = Path(config.repo_root)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_descriptor_path, fastpm_descriptor_path = _resolve_tidal_descriptor_dirs(
        feature_names,
        config.unit_descriptors_dir,
        config.fastpm_descriptors_dir,
    )
    collect_run_tables = config.collect_tables or config.run_assembly_bias_plot

    halos_sim = load_unit_sim_training_catalog(sim_path, max_halos=config.max_sim_halos)
    halos_fastpm = load_fastpm_target_catalog(
        fastpm_path,
        max_halos=config.max_fastpm_halos,
    )
    preprocessing_context = build_preprocessing_context(
        repo_root=root,
        calibrate_mass=CALIBRATE_MASS,
        sim_boxsize_mpc_h=SIM_BOXSIZE_MPC_H,
        fastpm_boxsize_mpc_h=FASTPM_BOXSIZE_MPC_H,
        env_radius_mpc_h=ENV_RADIUS_MPC_H,
        unit_descriptors_dir=unit_descriptor_path,
        fastpm_descriptors_dir=fastpm_descriptor_path,
        tidal_n_grid=config.tidal_n_grid,
        max_descriptor_batch_files=config.max_descriptor_batch_files,
    )
    halos_sim, halos_fastpm, mass_col_fastpm = build_feature_tables(
        halos_sim,
        halos_fastpm,
        feature_names,
        preprocessing_context,
    )

    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))
    output_features = list(OUTPUT_FEATURES)
    if config.run_holdout_validation:
        holdout_validate_sim_bins(
            halos_sim,
            bin_edges,
            input_features=feature_names,
            output_features=output_features,
            min_bin_size=config.min_bin_size,
        )

    enriched, _ = enrich_fastpm_catalog(
        halos_sim,
        halos_fastpm,
        bin_edges,
        mass_column_fastpm=mass_col_fastpm,
        input_features=feature_names,
        output_features=output_features,
        min_bin_size=config.min_bin_size,
    )
    predicted = enriched[list(output_features)].notna().all(axis=1).sum()
    if predicted == 0:
        raise RuntimeError(
            "No FastPM halos received Haloscope predictions; "
            "increase subset size or lower min_bin_size."
        )

    out_parquet = out_dir / config.enriched_parquet_name
    enriched.to_parquet(out_parquet, index=False)

    if collect_run_tables:
        run_result = HaloscopeEnrichmentRun(
            output_path=out_parquet,
            hr_table=halos_sim,
            enriched_lr_table=enriched,
            mass_column_fastpm=mass_col_fastpm,
        )
        if config.run_assembly_bias_plot:
            write_tidal_assembly_bias_panel(
                run_result.hr_table,
                run_result.enriched_lr_table,
                root,
                out_dir,
                input_features=feature_names,
                mass_column_fastpm=run_result.mass_column_fastpm,
                assembly_bias_n_grid=config.assembly_bias_n_grid,
            )
        return run_result
    return out_parquet
