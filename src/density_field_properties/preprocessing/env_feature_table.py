"""Deprecated shim — use preprocessing.feature_table.build_feature_tables."""

import warnings
from typing import Sequence

import pandas as pd

from density_field_properties.preprocessing.context import (
    PreprocessingContext,
    SimulationRunContext,
)
from density_field_properties.preprocessing.feature_table import build_feature_tables

warnings.warn(
    "preprocessing.env_feature_table is deprecated; use preprocessing.feature_table",
    DeprecationWarning,
    stacklevel=2,
)


def build_env_feature_tables(
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    calibrate_mass: bool,
    sim_boxsize_mpc_h: float,
    fastpm_boxsize_mpc_h: float,
    env_radius_mpc_h: float,
    input_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Deprecated wrapper around ``build_feature_tables`` for env-only runs.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        High-resolution training catalog.
    halos_fastpm : pd.DataFrame
        Low-resolution target catalog.
    calibrate_mass : bool
        Whether to abundance-match LR masses to HR.
    sim_boxsize_mpc_h : float
        SIM periodic box side in Mpc/h.
    fastpm_boxsize_mpc_h : float
        FastPM periodic box side in Mpc/h.
    env_radius_mpc_h : float
        Environment search radius in Mpc/h.
    input_features : Sequence[str]
        Haloscope INPUT feature column names.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, str]
        HR table, LR table, and LR mass column for bin assignment.
    """
    context = PreprocessingContext(
        sim=SimulationRunContext(
            boxsize_mpc_h=sim_boxsize_mpc_h,
            env_radius_mpc_h=env_radius_mpc_h,
        ),
        fastpm=SimulationRunContext(
            boxsize_mpc_h=fastpm_boxsize_mpc_h,
            env_radius_mpc_h=env_radius_mpc_h,
        ),
        calibrate_mass=calibrate_mass,
    )
    return build_feature_tables(halos_sim, halos_fastpm, input_features, context)
