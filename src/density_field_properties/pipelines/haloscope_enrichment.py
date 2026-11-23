"""Haloscope SIM-to-FastPM enrichment orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from density_field_properties.haloscope import enrich_fastpm_catalog
from density_field_properties.haloscope.mass_bin_config import resolve_mass_bin_edges
from density_field_properties.pipelines.config import HaloscopeEnrichmentConfig
from density_field_properties.pipelines.preprocessing_run import build_haloscope_feature_tables
from density_field_properties.pipelines.run_defaults import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    default_fastpm_tidal_descriptors_dir,
    default_unit_tidal_descriptors_dir,
)
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    input_features_need_tidal_descriptors,
)
from density_field_properties.validation.assembly_bias_panel import (
    write_tidal_assembly_bias_panel,
)
from density_field_properties.validation.holdout_corner_plots import (
    write_sim_holdout_corner_plots,
)
from density_field_properties.validation.memory import release_validation_memory


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
    if not input_features_need_tidal_descriptors(feature_names):
        return unit_descriptors_dir, fastpm_descriptors_dir

    unit_path = unit_descriptors_dir
    fastpm_path = fastpm_descriptors_dir
    if unit_path is None:
        unit_path = default_unit_tidal_descriptors_dir()
    if fastpm_path is None:
        fastpm_path = default_fastpm_tidal_descriptors_dir()
    return unit_path, fastpm_path


def run_haloscope_enrichment_from_tables(
    config: HaloscopeEnrichmentConfig,
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    mass_col_fastpm: str,
    run_enrichment: bool = True,
    write_assembly_bias: bool = True,
) -> Union[Path, HaloscopeEnrichmentRun, None]:
    """
    Run Haloscope enrichment and optional assembly-bias plots from prebuilt tables.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.
    halos_sim : pd.DataFrame
        HR table with attached INPUT features.
    halos_fastpm : pd.DataFrame
        LR table with attached INPUT features.
    mass_col_fastpm : str
        LR mass column used for bin assignment.
    run_enrichment : bool, optional
        If False, skip model fitting and only write assembly-bias diagnostics.
    write_assembly_bias : bool, optional
        When False, skip the assembly-bias PDF even if ``validation.enabled``.

    Returns
    -------
    Path, HaloscopeEnrichmentRun, or None
        Enriched Parquet path, run object, or ``None`` when enrichment is skipped.
    """
    feature_names = config.input_features if config.input_features else INPUT_FEATURES
    root = Path(config.repo_root)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_assembly = write_assembly_bias and config.run_assembly_bias_plot
    collect_run_tables = config.collect_tables or run_assembly

    if not run_enrichment:
        if not run_assembly:
            return None
        enriched = halos_fastpm
        out_parquet = out_dir / config.enriched_parquet_name
        run_result = HaloscopeEnrichmentRun(
            output_path=out_parquet,
            hr_table=halos_sim,
            enriched_lr_table=enriched,
            mass_column_fastpm=mass_col_fastpm,
        )
        write_tidal_assembly_bias_panel(
            run_result.hr_table,
            run_result.enriched_lr_table,
            root,
            out_dir,
            input_features=feature_names,
            mass_column_fastpm=run_result.mass_column_fastpm,
            assembly_bias_n_grid=config.assembly_bias_n_grid,
            config=config,
        )
        release_validation_memory()
        return run_result

    bin_edges = resolve_mass_bin_edges(
        np.log10(halos_sim["M200b"].max()),
        halos_fastpm[mass_col_fastpm].to_numpy(),
        config.mass_bin,
    )
    output_features = list(OUTPUT_FEATURES)
    if config.run_holdout_validation or config.run_holdout_corner_plots:
        write_sim_holdout_corner_plots(
            halos_sim,
            bin_edges,
            out_dir,
            input_features=feature_names,
            output_features=output_features,
            min_bin_size=config.min_bin_size,
            write_plots=config.run_holdout_corner_plots,
        )
        release_validation_memory()

    enriched, fitted_models = enrich_fastpm_catalog(
        halos_sim,
        halos_fastpm,
        bin_edges,
        mass_column_fastpm=mass_col_fastpm,
        input_features=feature_names,
        output_features=output_features,
        min_bin_size=config.min_bin_size,
    )
    del fitted_models
    release_validation_memory()

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
        if run_assembly:
            write_tidal_assembly_bias_panel(
                run_result.hr_table,
                run_result.enriched_lr_table,
                root,
                out_dir,
                input_features=feature_names,
                mass_column_fastpm=run_result.mass_column_fastpm,
                assembly_bias_n_grid=config.assembly_bias_n_grid,
                config=config,
            )
            release_validation_memory()
        return run_result
    if run_assembly:
        write_tidal_assembly_bias_panel(
            halos_sim,
            enriched,
            root,
            out_dir,
            input_features=feature_names,
            mass_column_fastpm=mass_col_fastpm,
            assembly_bias_n_grid=config.assembly_bias_n_grid,
            config=config,
        )
        release_validation_memory()
    return out_parquet


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
    halos_sim, halos_fastpm, mass_col_fastpm = build_haloscope_feature_tables(config)
    return run_haloscope_enrichment_from_tables(
        config,
        halos_sim,
        halos_fastpm,
        mass_col_fastpm,
        run_enrichment=True,
    )
