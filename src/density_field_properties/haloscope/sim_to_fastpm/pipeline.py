"""End-to-end SIM-to-FastPM Haloscope pipeline (subset-friendly)."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from density_field_properties.haloscope.sim_to_fastpm.config import (
    CALIBRATE_MASS,
    ENRICHED_PARQUET_NAME,
    ENV_RADIUS_MPC_H,
    FASTPM_BOXSIZE_MPC_H,
    OUTPUT_DIR,
    SIM_BOXSIZE_MPC_H,
)
from density_field_properties.haloscope.sim_to_fastpm.environment import local_environment
from density_field_properties.haloscope.sim_to_fastpm.load_catalogs import (
    load_fastpm_target_catalog,
    load_unit_sim_training_catalog,
)
from density_field_properties.haloscope.sim_to_fastpm.mass_matching import abundance_match_mass
from density_field_properties.haloscope.sim_to_fastpm.training import (
    default_mass_bin_edges,
    enrich_fastpm_catalog,
    holdout_validate_sim_bins,
)


def run_sim_to_fastpm_haloscope_pipeline(
    sim_hlist_path: Union[str, Path],
    fastpm_list_path: Union[str, Path],
    max_sim_halos: Optional[int] = None,
    max_fastpm_halos: Optional[int] = None,
    output_dir: Optional[Path] = None,
    min_bin_size: int = 10,
    run_holdout_validation: bool = True,
) -> Path:
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

    Returns
    -------
    Path
        Path to the enriched FastPM Parquet file.
    """
    sim_path = Path(sim_hlist_path)
    fastpm_path = Path(fastpm_list_path)
    if not sim_path.is_file():
        raise FileNotFoundError(f"SIM hlist not found: {sim_path}")
    if not fastpm_path.is_file():
        raise FileNotFoundError(f"FastPM catalog not found: {fastpm_path}")

    out_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    halos_sim = load_unit_sim_training_catalog(sim_path, max_halos=max_sim_halos)
    halos_fastpm = load_fastpm_target_catalog(fastpm_path, max_halos=max_fastpm_halos)
    if len(halos_sim) == 0 or len(halos_fastpm) == 0:
        raise ValueError("Empty SIM or FastPM catalog after loading and host filtering.")

    if CALIBRATE_MASS:
        halos_fastpm["M200b_cal"] = abundance_match_mass(
            halos_fastpm["M200b"].to_numpy(), halos_sim["M200b"].to_numpy()
        )
        mass_col_fastpm = "M200b_cal"
    else:
        mass_col_fastpm = "M200b"

    halos_sim["env"] = local_environment(
        halos_sim[["x", "y", "z"]].to_numpy(),
        SIM_BOXSIZE_MPC_H,
        radius_mpc_h=ENV_RADIUS_MPC_H,
    )
    halos_fastpm["env"] = local_environment(
        halos_fastpm[["x", "y", "z"]].to_numpy(),
        FASTPM_BOXSIZE_MPC_H,
        radius_mpc_h=ENV_RADIUS_MPC_H,
    )

    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))
    if run_holdout_validation:
        holdout_validate_sim_bins(
            halos_sim,
            bin_edges,
            min_bin_size=min_bin_size,
        )

    enriched, _ = enrich_fastpm_catalog(
        halos_sim,
        halos_fastpm,
        bin_edges,
        mass_column_fastpm=mass_col_fastpm,
        min_bin_size=min_bin_size,
    )
    predicted = enriched[["cv", "Spin", "ca", "ba"]].notna().all(axis=1).sum()
    if predicted == 0:
        raise RuntimeError(
            "No FastPM halos received Haloscope predictions; "
            "increase subset size or lower min_bin_size."
        )

    out_parquet = out_dir / ENRICHED_PARQUET_NAME
    enriched.to_parquet(out_parquet, index=False)
    return out_parquet
