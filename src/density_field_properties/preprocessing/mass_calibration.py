"""Mass calibration between LR and HR halo catalogs."""

import numpy as np
import pandas as pd
from scipy import stats


def abundance_match_mass(mass_target: np.ndarray, mass_reference: np.ndarray) -> np.ndarray:
    """
    Monotonic abundance matching of ``mass_target`` onto the mass function of ``mass_reference``.

    Parameters
    ----------
    mass_target : np.ndarray
        Masses to remap (e.g. FastPM M200b).
    mass_reference : np.ndarray
        Reference mass sample (e.g. UNIT SIM M200b).

    Returns
    -------
    np.ndarray
        Masses on the reference cumulative distribution, same shape as ``mass_target``.
    """
    quantile_target = stats.rankdata(mass_target) / (len(mass_target) + 1)
    reference_sorted = np.sort(mass_reference)
    quantile_reference = (np.arange(len(reference_sorted)) + 1) / (len(reference_sorted) + 1)
    return np.interp(quantile_target, quantile_reference, reference_sorted)


def calibrate_lr_mass(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    calibrate_mass: bool,
    lr_mass_column: str = "M200b",
    hr_mass_column: str = "M200b",
    calibrated_column: str = "M200b_cal",
) -> tuple[pd.DataFrame, str]:
    """
    Optionally abundance-match LR masses to the HR mass function.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog (FastPM).
    hr_catalog : pd.DataFrame
        High-resolution training catalog (UNIT).
    calibrate_mass : bool
        If True, write ``calibrated_column`` on ``lr_catalog``.
    lr_mass_column : str, optional
        Raw LR mass column.
    hr_mass_column : str, optional
        HR reference mass column.
    calibrated_column : str, optional
        Output calibrated mass column on LR.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The same ``lr_catalog`` instance and the mass column for bin assignment.
    """
    if not calibrate_mass:
        return lr_catalog, lr_mass_column

    lr_catalog[calibrated_column] = abundance_match_mass(
        lr_catalog[lr_mass_column].to_numpy(),
        hr_catalog[hr_mass_column].to_numpy(),
    )
    return lr_catalog, calibrated_column
