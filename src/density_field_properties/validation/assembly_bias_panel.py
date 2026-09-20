"""Assembly-bias diagnostic panel for Haloscope inputs."""

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

from density_field_properties.haloscope.sim_to_fastpm.config import (
    ASSEMBLY_BIAS_DM_BATCH_SIZE,
    DM_MASS_PARTICLE_MSUN_H,
    FASTPM_BOXSIZE_MPC_H,
    OUTPUT_FEATURES,
    SIM_BOXSIZE_MPC_H,
    default_fastpm_dm_particles_path,
    default_fastpm_saved_cic_density_paths,
    default_sim_dm_particles_path,
    default_sim_saved_cic_density_paths,
)
from density_field_properties.pipelines.config import (
    ASSEMBLY_BIAS_TIDAL_PDF_NAME,
    DEFAULT_ASSEMBLY_BIAS_N_GRID,
)
from density_field_properties.validation.assembly_bias import (
    assembly_bias_curves_for_catalog,
    attach_paranjape_bias,
    load_fastpm_matter_overdensity,
    load_sim_matter_overdensity,
    property_matrix_from_frame,
)
from density_field_properties.validation.plots import plot_assembly_bias_env_panel

DEFAULT_ASSEMBLY_LOG_MASS_MIN = 11.5
DEFAULT_ASSEMBLY_LOG_MASS_BINS = 10

INPUT_FEATURE_PLOT_LABELS = {
    "env": "env",
    "t_over_u": "T/|U|",
    "tidal_anisotropy": "tidal anisotropy",
}


def format_assembly_bias_panel_title(
    input_features: Sequence[str],
    sim_delta_mode: str,
    fastpm_delta_mode: str,
) -> str:
    """
    Build the assembly-bias panel title from Haloscope INPUT features.

    Parameters
    ----------
    input_features : Sequence[str]
        Haloscope INPUT column names used for the enrichment run.
    sim_delta_mode : str
        Label for the HR matter overdensity source.
    fastpm_delta_mode : str
        Label for the FastPM matter overdensity source.

    Returns
    -------
    str
        Plot title including INPUT feature labels and delta-field modes.
    """
    labels = [
        INPUT_FEATURE_PLOT_LABELS.get(feature_name, feature_name)
        for feature_name in input_features
    ]
    feature_text = " + ".join(labels)
    return f"input: {feature_text} " f"(HR δ: {sim_delta_mode}, FastPM δ: {fastpm_delta_mode})"


def write_tidal_assembly_bias_panel(
    halos_sim: pd.DataFrame,
    halos_fastpm_enriched: pd.DataFrame,
    repo_root: Union[str, Path],
    output_dir: Union[str, Path],
    input_features: Sequence[str],
    mass_column_fastpm: str = "M200b",
    assembly_bias_n_grid: int = DEFAULT_ASSEMBLY_BIAS_N_GRID,
) -> Path:
    """
    Build and save the HALOSCOPE-style assembly-bias panel.

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
    input_features : Sequence[str]
        Haloscope INPUT columns used for the enrichment run.
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

    sim_density_path, sim_density_info_path = default_sim_saved_cic_density_paths()
    fastpm_density_path, fastpm_density_info_path = default_fastpm_saved_cic_density_paths()
    delta_sim, sim_delta_mode = load_sim_matter_overdensity(
        root,
        SIM_BOXSIZE_MPC_H,
        assembly_bias_n_grid,
        DM_MASS_PARTICLE_MSUN_H,
        dm_batch_size=ASSEMBLY_BIAS_DM_BATCH_SIZE,
        dm_particles_path=default_sim_dm_particles_path(),
        saved_density_path=sim_density_path,
        saved_density_info_path=sim_density_info_path,
    )
    delta_fastpm, fp_delta_mode = load_fastpm_matter_overdensity(
        root,
        FASTPM_BOXSIZE_MPC_H,
        assembly_bias_n_grid,
        DM_MASS_PARTICLE_MSUN_H,
        default_fastpm_dm_particles_path(),
        dm_batch_size=ASSEMBLY_BIAS_DM_BATCH_SIZE,
        saved_density_path=fastpm_density_path,
        saved_density_info_path=fastpm_density_info_path,
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
        title=format_assembly_bias_panel_title(
            input_features,
            sim_delta_mode,
            fp_delta_mode,
        ),
        output_path=str(output_path),
    )
    return output_path
