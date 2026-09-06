"""SIM-to-FastPM Haloscope pipeline with T/|U| and tidal anisotropy inputs."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from density_field_properties.haloscope.sim_to_fastpm.assembly_bias import (
    assembly_bias_curves_for_catalog,
    attach_paranjape_bias,
    load_fastpm_matter_overdensity,
    load_sim_matter_overdensity,
    property_matrix_from_frame,
)
from density_field_properties.haloscope.sim_to_fastpm.config import (
    ASSEMBLY_BIAS_DM_BATCH_SIZE,
    CALIBRATE_MASS,
    DM_MASS_PARTICLE_MSUN_H,
    ENRICHED_TIDAL_PARQUET_NAME,
    FASTPM_BOXSIZE_MPC_H,
    OUTPUT_DIR_TIDAL,
    OUTPUT_FEATURES,
    SIM_BOXSIZE_MPC_H,
    TIDAL_DENSITY_N_GRID,
    TIDAL_INPUT_FEATURES,
    default_fastpm_dm_particles_path,
    default_fastpm_tidal_descriptors_dir,
    default_unit_tidal_descriptors_dir,
)
from density_field_properties.haloscope.sim_to_fastpm.load_catalogs import (
    load_fastpm_target_catalog,
    load_unit_sim_training_catalog,
)
from density_field_properties.haloscope.sim_to_fastpm.mass_matching import abundance_match_mass
from density_field_properties.haloscope.sim_to_fastpm.plotting import plot_assembly_bias_env_panel
from density_field_properties.haloscope.sim_to_fastpm.tidal_features import (
    attach_tidal_anisotropy,
    filter_finite_input_features,
)
from density_field_properties.haloscope.sim_to_fastpm.training import (
    default_mass_bin_edges,
    enrich_fastpm_catalog,
    holdout_validate_sim_bins,
)

ASSEMBLY_BIAS_TIDAL_PDF_NAME = "assembly_bias_tidal_input.pdf"
DEFAULT_ASSEMBLY_BIAS_N_GRID = 128
DEFAULT_ASSEMBLY_LOG_MASS_MIN = 11.5
DEFAULT_ASSEMBLY_LOG_MASS_BINS = 10


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


def run_sim_to_fastpm_haloscope_tidal_pipeline(
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
    assembly_bias_n_grid: int = DEFAULT_ASSEMBLY_BIAS_N_GRID,
) -> Path:
    """
    Run Haloscope enrichment using Rockstar ``T/|U|`` and tidal anisotropy inputs.

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
        UNIT tidal descriptor directory; defaults to ``config`` path.
    fastpm_descriptors_dir : Optional[Path], optional
        FastPM tidal descriptor directory; defaults to ``config`` path.
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
    sim_path = Path(sim_hlist_path)
    fastpm_path = Path(fastpm_list_path)
    root = Path(repo_root)
    if not sim_path.is_file():
        raise FileNotFoundError(f"SIM hlist not found: {sim_path}")
    if not fastpm_path.is_file():
        raise FileNotFoundError(f"FastPM catalog not found: {fastpm_path}")

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
    if not unit_descriptor_path.is_absolute():
        unit_descriptor_path = root / unit_descriptor_path
    if not fastpm_descriptor_path.is_absolute():
        fastpm_descriptor_path = root / fastpm_descriptor_path

    out_dir = OUTPUT_DIR_TIDAL if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    halos_sim = load_unit_sim_training_catalog(sim_path, max_halos=max_sim_halos)
    halos_fastpm = load_fastpm_target_catalog(fastpm_path, max_halos=max_fastpm_halos)
    if len(halos_sim) == 0 or len(halos_fastpm) == 0:
        raise ValueError("Empty SIM or FastPM catalog after loading and host filtering.")

    halos_sim = attach_tidal_anisotropy(
        halos_sim,
        unit_descriptor_path,
        SIM_BOXSIZE_MPC_H,
        n_grid,
        max_batch_files=max_descriptor_batch_files,
    )
    halos_fastpm = attach_tidal_anisotropy(
        halos_fastpm,
        fastpm_descriptor_path,
        FASTPM_BOXSIZE_MPC_H,
        n_grid,
        max_batch_files=max_descriptor_batch_files,
    )
    halos_sim = filter_finite_input_features(halos_sim, TIDAL_INPUT_FEATURES)
    halos_fastpm = filter_finite_input_features(halos_fastpm, TIDAL_INPUT_FEATURES)
    if len(halos_sim) == 0 or len(halos_fastpm) == 0:
        raise ValueError(
            "No halos left after attaching tidal features; "
            "check descriptor directories and batch limits."
        )

    if CALIBRATE_MASS:
        halos_fastpm["M200b_cal"] = abundance_match_mass(
            halos_fastpm["M200b"].to_numpy(), halos_sim["M200b"].to_numpy()
        )
        mass_col_fastpm = "M200b_cal"
    else:
        mass_col_fastpm = "M200b"

    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))
    if run_holdout_validation:
        holdout_validate_sim_bins(
            halos_sim,
            bin_edges,
            min_bin_size=min_bin_size,
            input_features=TIDAL_INPUT_FEATURES,
        )

    enriched, _ = enrich_fastpm_catalog(
        halos_sim,
        halos_fastpm,
        bin_edges,
        mass_column_fastpm=mass_col_fastpm,
        min_bin_size=min_bin_size,
        input_features=TIDAL_INPUT_FEATURES,
    )
    predicted = enriched[["cv", "Spin", "ca", "ba"]].notna().all(axis=1).sum()
    if predicted == 0:
        raise RuntimeError(
            "No FastPM halos received Haloscope predictions; "
            "increase subset size or lower min_bin_size."
        )

    out_parquet = out_dir / ENRICHED_TIDAL_PARQUET_NAME
    enriched.to_parquet(out_parquet, index=False)

    if run_assembly_bias_plot:
        assembly_bias_path = write_tidal_assembly_bias_panel(
            halos_sim,
            enriched,
            root,
            out_dir,
            mass_column_fastpm=mass_col_fastpm,
            assembly_bias_n_grid=assembly_bias_n_grid,
        )
        print(f"Assembly bias panel saved to {assembly_bias_path}")

    return out_parquet
