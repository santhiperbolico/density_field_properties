"""Haloscope SIM-to-FastPM enrichment orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from density_field_properties.haloscope.sim_to_fastpm.config import (
    CALIBRATE_MASS,
    ENRICHED_PARQUET_NAME,
    ENV_RADIUS_MPC_H,
    FASTPM_BOXSIZE_MPC_H,
    INPUT_FEATURES,
    OUTPUT_DIR,
    SIM_BOXSIZE_MPC_H,
    default_fastpm_tidal_descriptors_dir,
    default_unit_tidal_descriptors_dir,
)
from density_field_properties.haloscope.sim_to_fastpm.load_catalogs import (
    load_fastpm_target_catalog,
    load_unit_sim_training_catalog,
)
from density_field_properties.haloscope.sim_to_fastpm.training import (
    default_mass_bin_edges,
    enrich_fastpm_catalog,
    holdout_validate_sim_bins,
)
from density_field_properties.preprocessing.context import build_preprocessing_context
from density_field_properties.preprocessing.feature_table import build_feature_tables
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    TIDAL_ANISOTROPY_FEATURE_NAME,
)

ASSEMBLY_BIAS_TIDAL_PDF_NAME = "assembly_bias_tidal_input.pdf"
DEFAULT_ASSEMBLY_BIAS_N_GRID = 128
DEFAULT_ASSEMBLY_LOG_MASS_MIN = 11.5
DEFAULT_ASSEMBLY_LOG_MASS_BINS = 10


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


def write_tidal_assembly_bias_panel(
    halos_sim: pd.DataFrame,
    halos_fastpm_enriched: pd.DataFrame,
    repo_root: Union[str, Path],
    output_dir: Union[str, Path],
    mass_column_fastpm: str = "M200b",
    assembly_bias_n_grid: int = DEFAULT_ASSEMBLY_BIAS_N_GRID,
) -> Path:
    """
    Build and save the HALOSCOPE-style assembly-bias panel for tidal inputs.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        UNIT training catalog with true output properties.
    halos_fastpm_enriched : pd.DataFrame
        FastPM catalog with predicted ``OUTPUT_FEATURES``.
    repo_root : str or Path
        Repository root for resolving saved CIC / DM paths.
    output_dir : str or Path
        Directory for the output PDF.
    mass_column_fastpm : str, optional
        Mass column used for FastPM binning.
    assembly_bias_n_grid : int, optional
        Grid resolution for matter ``delta`` and Paranjape bias.

    Returns
    -------
    Path
        Path to the saved PDF.
    """
    from density_field_properties.haloscope.sim_to_fastpm.assembly_bias import (
        assembly_bias_curves_for_catalog,
        attach_paranjape_bias,
        load_fastpm_matter_overdensity,
        load_sim_matter_overdensity,
        property_matrix_from_frame,
    )
    from density_field_properties.haloscope.sim_to_fastpm.config import (
        ASSEMBLY_BIAS_DM_BATCH_SIZE,
        DM_MASS_PARTICLE_MSUN_H,
        OUTPUT_FEATURES,
        default_fastpm_dm_particles_path,
    )
    from density_field_properties.haloscope.sim_to_fastpm.plotting import (
        plot_assembly_bias_env_panel,
    )

    root = Path(repo_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_mass_bins = np.linspace(
        DEFAULT_ASSEMBLY_LOG_MASS_MIN,
        np.log10(halos_sim["M200b"].max()),
        DEFAULT_ASSEMBLY_LOG_MASS_BINS,
    )

    delta_sim, sim_delta_mode = load_sim_matter_overdensity(
        root,
        SIM_BOXSIZE_MPC_H,
        assembly_bias_n_grid,
        DM_MASS_PARTICLE_MSUN_H,
        dm_batch_size=ASSEMBLY_BIAS_DM_BATCH_SIZE,
    )
    delta_fastpm, fp_delta_mode = load_fastpm_matter_overdensity(
        root,
        FASTPM_BOXSIZE_MPC_H,
        assembly_bias_n_grid,
        DM_MASS_PARTICLE_MSUN_H,
        default_fastpm_dm_particles_path(),
        dm_batch_size=ASSEMBLY_BIAS_DM_BATCH_SIZE,
    )
    if delta_sim is None:
        sim_delta_mode = "halo CIC"
    if delta_fastpm is None:
        fp_delta_mode = "halo CIC"

    halos_sim = halos_sim.copy()
    halos_fastpm_enriched = halos_fastpm_enriched.copy()
    halos_sim["b1"] = attach_paranjape_bias(
        halos_sim,
        SIM_BOXSIZE_MPC_H,
        n_grid=assembly_bias_n_grid,
        matter_delta_field=delta_sim,
    )

    fp_for_bias = halos_fastpm_enriched.dropna(subset=list(OUTPUT_FEATURES)).copy()
    halos_fastpm_enriched.loc[fp_for_bias.index, "b1"] = attach_paranjape_bias(
        fp_for_bias,
        FASTPM_BOXSIZE_MPC_H,
        n_grid=assembly_bias_n_grid,
        mass_column=mass_column_fastpm,
        matter_delta_field=delta_fastpm,
    )
    fp_for_bias = halos_fastpm_enriched.dropna(subset=["b1"] + list(OUTPUT_FEATURES)).copy()
    if len(fp_for_bias) == 0:
        raise RuntimeError(
            "No FastPM halos with finite b1 and predicted properties for assembly bias plot."
        )

    hr_props = property_matrix_from_frame(halos_sim, OUTPUT_FEATURES)
    lr_props = property_matrix_from_frame(fp_for_bias, OUTPUT_FEATURES)
    hr_curves = assembly_bias_curves_for_catalog(
        halos_sim["M200b"].to_numpy(),
        halos_sim["b1"].to_numpy(),
        hr_props,
        log_mass_bins,
    )
    lr_curves = assembly_bias_curves_for_catalog(
        fp_for_bias[mass_column_fastpm].to_numpy(),
        fp_for_bias["b1"].to_numpy(),
        lr_props,
        log_mass_bins,
    )

    output_path = out_dir / ASSEMBLY_BIAS_TIDAL_PDF_NAME
    plot_assembly_bias_env_panel(
        *hr_curves,
        lr_curves[0],
        lr_curves[1],
        lr_curves[3],
        lr_curves[4],
        title=(
            "input: T/|U| + tidal anisotropy "
            f"(HR δ: {sim_delta_mode}, FastPM δ: {fp_delta_mode})"
        ),
        output_path=str(output_path),
    )
    return output_path


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
    sim_hlist_path: Union[str, Path],
    fastpm_list_path: Union[str, Path],
    max_sim_halos: Optional[int] = None,
    max_fastpm_halos: Optional[int] = None,
    output_dir: Optional[Path] = None,
    min_bin_size: int = 10,
    run_holdout_validation: bool = True,
    input_features: Optional[Sequence[str]] = None,
    repo_root: Optional[Path] = None,
    unit_descriptors_dir: Optional[Path] = None,
    fastpm_descriptors_dir: Optional[Path] = None,
    tidal_n_grid: int = 512,
    max_descriptor_batch_files: Optional[int] = None,
    enriched_parquet_name: str = ENRICHED_PARQUET_NAME,
    run_assembly_bias_plot: bool = False,
    assembly_bias_n_grid: int = DEFAULT_ASSEMBLY_BIAS_N_GRID,
    collect_tables: bool = False,
) -> Union[Path, HaloscopeEnrichmentRun]:
    """
    Run Haloscope enrichment on FastPM using a high-resolution SIM as training data.

    Parameters
    ----------
    sim_hlist_path : str or Path
        Path to the SIM (e.g. UNIT consistent-trees) ``hlist_*.list`` or ``.bz2`` file.
    fastpm_list_path : str or Path
        Path to the FastPM Rockstar ``out_*.list`` catalog to enrich.
    max_sim_halos : Optional[int], optional
        Cap rows read from the SIM hlist; ``None`` reads the full catalog.
    max_fastpm_halos : Optional[int], optional
        Cap halos read from the FastPM ``.list`` file.
    output_dir : Optional[Path], optional
        Directory for Parquet output; defaults to ``config.OUTPUT_DIR``.
    min_bin_size : int, optional
        Minimum halos per mass bin for fit and validation.
    run_holdout_validation : bool, optional
        If True, run SIM train/test validation (no files written for plots).
    input_features : Optional[Sequence[str]], optional
        Haloscope INPUT columns to attach; defaults to ``config.INPUT_FEATURES``.
    repo_root : Optional[Path], optional
        Repository root for resolving relative tidal descriptor directories.
    unit_descriptors_dir : Optional[Path], optional
        UNIT tidal descriptor directory.
    fastpm_descriptors_dir : Optional[Path], optional
        FastPM tidal descriptor directory.
    tidal_n_grid : int, optional
        Grid resolution used when tidal descriptors were computed.
    max_descriptor_batch_files : Optional[int], optional
        Smoke cap on tidal descriptor batch files per simulation.
    enriched_parquet_name : str, optional
        Output Parquet filename inside ``output_dir``.
    run_assembly_bias_plot : bool, optional
        If True, write ``assembly_bias_tidal_input.pdf`` after enrichment.
    assembly_bias_n_grid : int, optional
        Grid resolution for the Paranjape assembly-bias diagnostic.
    collect_tables : bool, optional
        If True, return a ``HaloscopeEnrichmentRun`` with HR/LR tables. Forced
        when ``run_assembly_bias_plot`` is True.

    Returns
    -------
    Path or HaloscopeEnrichmentRun
        Enriched Parquet path, or a run object when tables are collected.
    """
    sim_path = Path(sim_hlist_path)
    fastpm_path = Path(fastpm_list_path)
    if not sim_path.is_file():
        raise FileNotFoundError(f"SIM hlist not found: {sim_path}")
    if not fastpm_path.is_file():
        raise FileNotFoundError(f"FastPM catalog not found: {fastpm_path}")

    feature_names = tuple(input_features) if input_features is not None else INPUT_FEATURES
    root = Path.cwd() if repo_root is None else Path(repo_root)
    out_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_descriptor_path, fastpm_descriptor_path = _resolve_tidal_descriptor_dirs(
        feature_names,
        unit_descriptors_dir,
        fastpm_descriptors_dir,
    )
    collect_run_tables = collect_tables or run_assembly_bias_plot

    halos_sim = load_unit_sim_training_catalog(sim_path, max_halos=max_sim_halos)
    halos_fastpm = load_fastpm_target_catalog(fastpm_path, max_halos=max_fastpm_halos)
    preprocessing_context = build_preprocessing_context(
        repo_root=root,
        calibrate_mass=CALIBRATE_MASS,
        sim_boxsize_mpc_h=SIM_BOXSIZE_MPC_H,
        fastpm_boxsize_mpc_h=FASTPM_BOXSIZE_MPC_H,
        env_radius_mpc_h=ENV_RADIUS_MPC_H,
        unit_descriptors_dir=unit_descriptor_path,
        fastpm_descriptors_dir=fastpm_descriptor_path,
        tidal_n_grid=tidal_n_grid,
        max_descriptor_batch_files=max_descriptor_batch_files,
    )
    halos_sim, halos_fastpm, mass_col_fastpm = build_feature_tables(
        halos_sim,
        halos_fastpm,
        feature_names,
        preprocessing_context,
    )

    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))
    if run_holdout_validation:
        holdout_validate_sim_bins(
            halos_sim,
            bin_edges,
            min_bin_size=min_bin_size,
            input_features=feature_names,
        )

    enriched, _ = enrich_fastpm_catalog(
        halos_sim,
        halos_fastpm,
        bin_edges,
        mass_column_fastpm=mass_col_fastpm,
        min_bin_size=min_bin_size,
        input_features=feature_names,
    )
    predicted = enriched[["cv", "Spin", "ca", "ba"]].notna().all(axis=1).sum()
    if predicted == 0:
        raise RuntimeError(
            "No FastPM halos received Haloscope predictions; "
            "increase subset size or lower min_bin_size."
        )

    out_parquet = out_dir / enriched_parquet_name
    enriched.to_parquet(out_parquet, index=False)

    if collect_run_tables:
        run_result = HaloscopeEnrichmentRun(
            output_path=out_parquet,
            hr_table=halos_sim,
            enriched_lr_table=enriched,
            mass_column_fastpm=mass_col_fastpm,
        )
        if run_assembly_bias_plot:
            write_tidal_assembly_bias_panel(
                run_result.hr_table,
                run_result.enriched_lr_table,
                root,
                out_dir,
                mass_column_fastpm=run_result.mass_column_fastpm,
                assembly_bias_n_grid=assembly_bias_n_grid,
            )
        return run_result
    return out_parquet
