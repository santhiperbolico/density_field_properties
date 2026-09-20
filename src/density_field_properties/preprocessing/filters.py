"""Scientific sample filters for Haloscope HR/LR tables."""

from typing import Sequence

import numpy as np
import pandas as pd


def filter_host_training_halos(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep UNIT host halos with positive mass and finite concentration.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw consistent-trees table with ``pid``, ``M200b``, and ``cv``.

    Returns
    -------
    pd.DataFrame
        Filtered host halos.
    """
    mask = (frame["pid"] == -1) & (frame["M200b"] > 0) & np.isfinite(frame["cv"])
    return frame.loc[mask].copy()


def filter_positive_mass(frame: pd.DataFrame, mass_column: str = "M200b") -> pd.DataFrame:
    """
    Keep halos with strictly positive mass.

    Parameters
    ----------
    frame : pd.DataFrame
        Halo catalog table.
    mass_column : str, optional
        Mass column name.

    Returns
    -------
    pd.DataFrame
        Filtered table with a reset index.
    """
    filtered = frame.loc[frame[mass_column] > 0].copy()
    return filtered.reset_index(drop=True)


def filter_min_m200b(
    frame: pd.DataFrame,
    min_mass_msun_h: float,
    mass_column: str = "M200b",
) -> pd.DataFrame:
    """
    Keep halos above a minimum ``M200b`` threshold.

    Parameters
    ----------
    frame : pd.DataFrame
        Halo catalog table.
    min_mass_msun_h : float
        Minimum halo mass in Msun/h.
    mass_column : str, optional
        Mass column name.

    Returns
    -------
    pd.DataFrame
        Filtered table with a reset index.
    """
    filtered = frame.loc[frame[mass_column] >= min_mass_msun_h].copy()
    return filtered.reset_index(drop=True)


def filter_finite_input_features(
    catalog: pd.DataFrame,
    input_features: Sequence[str],
) -> pd.DataFrame:
    """
    Drop rows with non-finite values in Haloscope input feature columns.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table.
    input_features : Sequence[str]
        Column names required for Haloscope ``fit`` / ``predict``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with a reset index.
    """
    mask = np.ones(len(catalog), dtype=bool)
    for column in input_features:
        mask &= np.isfinite(catalog[column].to_numpy(dtype=np.float64))
    return catalog.loc[mask].reset_index(drop=True)
