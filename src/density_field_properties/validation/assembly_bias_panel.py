"""Assembly-bias diagnostic panel for Haloscope inputs."""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from density_field_properties.haloscope.bins import default_mass_bin_edges
from density_field_properties.pipelines.config import (
    ASSEMBLY_BIAS_TIDAL_PDF_NAME,
    DEFAULT_ASSEMBLY_BIAS_N_GRID,
    HaloscopeEnrichmentConfig,
    resolve_dm_mass_particle_msun_h,
    resolve_fastpm_boxsize_mpc_h,
    resolve_fastpm_dm_particles_path,
    resolve_sim_boxsize_mpc_h,
    resolve_sim_dm_particles_path,
)
from density_field_properties.pipelines.run_defaults import (
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
from density_field_properties.pipelines.tidal_anisotropy_run import (
    try_resolve_work_dir_cic_density_paths,
)
from density_field_properties.validation.assembly_bias import (
    assembly_bias_curves_for_catalog,
    attach_paranjape_bias,
    load_fastpm_matter_overdensity,
    load_sim_matter_overdensity,
    property_matrix_from_frame,
)
from density_field_properties.validation.memory import release_validation_memory
from density_field_properties.validation.plots import plot_assembly_bias_env_panel

INPUT_FEATURE_PLOT_LABELS = {
    "env": "env",
    "t_over_u": "T/|U|",
    "tidal_anisotropy": "tidal anisotropy",
    "overdensity": "overdensity",
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


def _resolve_assembly_bias_context(
    config: Optional[HaloscopeEnrichmentConfig],
) -> tuple[float, float, Optional[Path], Optional[Path], float]:
    """
    Resolve box sizes, DM paths, and particle mass for assembly-bias diagnostics.

    Parameters
    ----------
    config : Optional[HaloscopeEnrichmentConfig]
        Run configuration. When ``None``, production defaults from ``run_defaults`` apply.

    Returns
    -------
    tuple[float, float, Optional[Path], Optional[Path], float]
        SIM box size, FastPM box size, SIM DM path, FastPM DM path, and DM mass.
    """
    if config is None:
        return (
            SIM_BOXSIZE_MPC_H,
            FASTPM_BOXSIZE_MPC_H,
            default_sim_dm_particles_path(),
            default_fastpm_dm_particles_path(),
            DM_MASS_PARTICLE_MSUN_H,
        )
    return (
        resolve_sim_boxsize_mpc_h(config),
        resolve_fastpm_boxsize_mpc_h(config),
        resolve_sim_dm_particles_path(config),
        resolve_fastpm_dm_particles_path(config),
        resolve_dm_mass_particle_msun_h(config),
    )


def _resolve_saved_cic_density_paths(
    repo_root: Path,
    config: Optional[HaloscopeEnrichmentConfig],
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """
    Resolve saved CIC density paths for HR and FastPM assembly-bias diagnostics.

    When a run config defines tidal-anisotropy ``work_dir`` values with a unique
    density artifact pair, those paths take precedence over ``run_defaults``.

    Parameters
    ----------
    repo_root : Path
        Repository root for resolving relative work directories.
    config : Optional[HaloscopeEnrichmentConfig]
        Haloscope run configuration.

    Returns
    -------
    tuple[tuple[Path, Path], tuple[Path, Path]]
        ``((sim_density, sim_info), (fastpm_density, fastpm_info))`` paths.
    """
    sim_density_path, sim_density_info_path = default_sim_saved_cic_density_paths()
    fastpm_density_path, fastpm_density_info_path = default_fastpm_saved_cic_density_paths()
    if config is None:
        return (sim_density_path, sim_density_info_path), (
            fastpm_density_path,
            fastpm_density_info_path,
        )

    from density_field_properties.pipelines.config import (
        PIPELINE_TARGET_FASTPM,
        PIPELINE_TARGET_UNIT,
        tidal_target_by_name,
    )

    unit_target = tidal_target_by_name(config, PIPELINE_TARGET_UNIT)
    if unit_target is not None and unit_target.work_dir is not None:
        unit_work_dir = unit_target.work_dir
        unit_dir = unit_work_dir if unit_work_dir.is_absolute() else repo_root / unit_work_dir
        unit_paths = try_resolve_work_dir_cic_density_paths(unit_dir)
        if unit_paths is not None:
            sim_density_path, sim_density_info_path = unit_paths

    fastpm_target = tidal_target_by_name(config, PIPELINE_TARGET_FASTPM)
    if fastpm_target is not None and fastpm_target.work_dir is not None:
        fastpm_work_dir = fastpm_target.work_dir
        fastpm_dir = (
            fastpm_work_dir if fastpm_work_dir.is_absolute() else repo_root / fastpm_work_dir
        )
        fastpm_paths = try_resolve_work_dir_cic_density_paths(fastpm_dir)
        if fastpm_paths is not None:
            fastpm_density_path, fastpm_density_info_path = fastpm_paths

    return (sim_density_path, sim_density_info_path), (
        fastpm_density_path,
        fastpm_density_info_path,
    )


def write_tidal_assembly_bias_panel(
    halos_sim: pd.DataFrame,
    halos_fastpm_enriched: pd.DataFrame,
    repo_root: Union[str, Path],
    output_dir: Union[str, Path],
    input_features: Sequence[str],
    mass_column_fastpm: str = "M200b",
    assembly_bias_n_grid: int = DEFAULT_ASSEMBLY_BIAS_N_GRID,
    config: Optional[HaloscopeEnrichmentConfig] = None,
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
    config : Optional[HaloscopeEnrichmentConfig], optional
        Run configuration supplying box size and DM particle paths.

    Returns
    -------
    Path
        Path to the saved PDF.
    """
    root = Path(repo_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_boxsize_mpc_h, fastpm_boxsize_mpc_h, sim_dm_path, fastpm_dm_path, dm_mass_msun_h = (
        _resolve_assembly_bias_context(config)
    )

    log_mass_bins = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))

    (sim_density_path, sim_density_info_path), (
        fastpm_density_path,
        fastpm_density_info_path,
    ) = _resolve_saved_cic_density_paths(root, config)
    delta_sim, sim_delta_mode = load_sim_matter_overdensity(
        root,
        sim_boxsize_mpc_h,
        assembly_bias_n_grid,
        dm_mass_msun_h,
        dm_batch_size=ASSEMBLY_BIAS_DM_BATCH_SIZE,
        dm_particles_path=sim_dm_path,
        saved_density_path=sim_density_path,
        saved_density_info_path=sim_density_info_path,
    )
    delta_fastpm, fp_delta_mode = load_fastpm_matter_overdensity(
        root,
        fastpm_boxsize_mpc_h,
        assembly_bias_n_grid,
        dm_mass_msun_h,
        fastpm_dm_path,
        dm_batch_size=ASSEMBLY_BIAS_DM_BATCH_SIZE,
        saved_density_path=fastpm_density_path,
        saved_density_info_path=fastpm_density_info_path,
    )
    if delta_sim is None:
        raise RuntimeError(
            "SIM matter overdensity is required for halo-by-halo b1; "
            "configure saved CIC or DM particles for the UNIT catalog."
        )
    if delta_fastpm is None:
        raise RuntimeError(
            "FastPM matter overdensity is required for halo-by-halo b1; "
            "configure saved CIC or DM particles for the FastPM catalog."
        )

    halos_sim = halos_sim.copy()
    halos_fastpm_enriched = halos_fastpm_enriched.copy()
    halos_sim["b1"] = attach_paranjape_bias(
        halos_sim,
        sim_boxsize_mpc_h,
        n_grid=assembly_bias_n_grid,
        matter_delta_field=delta_sim,
    )
    del delta_sim
    release_validation_memory()

    fp_for_bias = halos_fastpm_enriched.dropna(subset=list(OUTPUT_FEATURES)).copy()
    halos_fastpm_enriched.loc[fp_for_bias.index, "b1"] = attach_paranjape_bias(
        fp_for_bias,
        fastpm_boxsize_mpc_h,
        n_grid=assembly_bias_n_grid,
        matter_delta_field=delta_fastpm,
    )
    del delta_fastpm
    release_validation_memory()
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
        threshold_mass=halos_sim["M200b"].to_numpy(),
        threshold_property_matrix=hr_props,
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
    del hr_props, lr_props, hr_curves, lr_curves, fp_for_bias
    release_validation_memory()
    return output_path
