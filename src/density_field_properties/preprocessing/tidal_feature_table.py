"""Deprecated shim — use preprocessing.feature_table.build_feature_tables."""

import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from density_field_properties.preprocessing.context import build_preprocessing_context
from density_field_properties.preprocessing.feature_table import build_feature_tables

warnings.warn(
    "preprocessing.tidal_feature_table is deprecated; use preprocessing.feature_table",
    DeprecationWarning,
    stacklevel=2,
)


def build_tidal_feature_tables(
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    unit_descriptors_dir: Path,
    fastpm_descriptors_dir: Path,
    calibrate_mass: bool,
    sim_boxsize_mpc_h: float,
    fastpm_boxsize_mpc_h: float,
    n_grid: int,
    input_features: Sequence[str],
    max_descriptor_batch_files: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Deprecated wrapper around ``build_feature_tables`` for tidal runs.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        High-resolution training catalog.
    halos_fastpm : pd.DataFrame
        Low-resolution target catalog.
    unit_descriptors_dir : Path
        UNIT tidal descriptor directory.
    fastpm_descriptors_dir : Path
        FastPM tidal descriptor directory.
    calibrate_mass : bool
        Whether to abundance-match LR masses to HR.
    sim_boxsize_mpc_h : float
        SIM periodic box side in Mpc/h.
    fastpm_boxsize_mpc_h : float
        FastPM periodic box side in Mpc/h.
    n_grid : int
        Grid resolution used for tidal descriptors.
    input_features : Sequence[str]
        Haloscope INPUT feature column names.
    max_descriptor_batch_files : Optional[int], optional
        Smoke cap on descriptor batch files per simulation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, str]
        HR table, LR table, and LR mass column for bin assignment.
    """
    context = build_preprocessing_context(
        repo_root=Path.cwd(),
        calibrate_mass=calibrate_mass,
        sim_boxsize_mpc_h=sim_boxsize_mpc_h,
        fastpm_boxsize_mpc_h=fastpm_boxsize_mpc_h,
        env_radius_mpc_h=5.0,
        unit_descriptors_dir=unit_descriptors_dir,
        fastpm_descriptors_dir=fastpm_descriptors_dir,
        tidal_n_grid=n_grid,
        max_descriptor_batch_files=max_descriptor_batch_files,
    )
    return build_feature_tables(halos_sim, halos_fastpm, input_features, context)
