"""Mass calibration between LR and HR halo catalogs."""

import numpy as np
import pandas as pd
from scipy import stats

from density_field_properties.preprocessing.halo_matching import (
    UNMATCHED_INDEX,
    hr_masses_for_matched_lr,
    match_hr_to_lr_one_to_one,
)
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
    MassCalibrationConfig,
    MatchingHyperparameters,
)


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


def _apply_abundance_matching(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
) -> pd.DataFrame:
    """
    Write abundance-matched masses on the LR catalog.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog.
    hr_catalog : pd.DataFrame
        High-resolution reference catalog.
    config : MassCalibrationConfig
        Calibration settings including column names.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with ``config.calibrated_column`` populated.

    Raises
    ------
    ValueError
        If either mass sample is empty.
    """
    lr_masses = lr_catalog[config.matching.lr_mass_column].to_numpy()
    hr_masses = hr_catalog[config.matching.hr_mass_column].to_numpy()
    if len(lr_masses) == 0 or len(hr_masses) == 0:
        raise ValueError("Empty HR or LR mass sample for abundance matching.")
    lr_catalog[config.calibrated_column] = abundance_match_mass(lr_masses, hr_masses)
    lr_catalog["mass_calib_method"] = MASS_CALIBRATION_METHOD_ABUNDANCE
    return lr_catalog


def _apply_matching_1to1(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
    box_size_mpc_h: float,
) -> pd.DataFrame:
    """
    Assign HR masses to matched LR halos via spatial 1:1 matching.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog.
    hr_catalog : pd.DataFrame
        High-resolution reference catalog.
    config : MassCalibrationConfig
        Calibration settings including matching hyperparameters.
    box_size_mpc_h : float
        Periodic box side length in Mpc/h.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with calibrated mass and optional trace columns.

    Raises
    ------
    ValueError
        If every LR calibrated mass is NaN after matching.
    """
    matching = match_hr_to_lr_one_to_one(
        hr_catalog,
        lr_catalog,
        box_size_mpc_h=box_size_mpc_h,
        params=config.matching,
    )
    hr_mass_column = config.matching.hr_mass_column
    hr_masses = hr_catalog[hr_mass_column].to_numpy(dtype=float)
    calibrated = hr_masses_for_matched_lr(matching, hr_masses)

    lr_catalog[config.calibrated_column] = calibrated
    matched_mask = matching.lr_to_hr_index != UNMATCHED_INDEX
    lr_catalog["mass_calib_matched"] = matched_mask
    lr_catalog["mass_calib_method"] = MASS_CALIBRATION_METHOD_MATCHING_1TO1

    if "id" in hr_catalog.columns:
        hr_ids = hr_catalog["id"].to_numpy()
        partner_ids = np.full(len(lr_catalog), np.nan, dtype=float)
        partner_ids[matched_mask] = hr_ids[matching.lr_to_hr_index[matched_mask]]
        lr_catalog["mass_calib_hr_id"] = partner_ids

    if not np.any(np.isfinite(calibrated)):
        raise ValueError("All LR calibrated masses are NaN after matching_1to1.")

    return lr_catalog


def apply_mass_calibration(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
    box_size_mpc_h: float,
) -> tuple[pd.DataFrame, str]:
    """
    Calibrate LR halo masses for Haloscope bin assignment.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog (FastPM).
    hr_catalog : pd.DataFrame
        High-resolution training catalog (UNIT).
    config : MassCalibrationConfig
        Method key and hyperparameters.
    box_size_mpc_h : float
        Periodic box side in Mpc/h (used by spatial matching methods).

    Returns
    -------
    tuple[pd.DataFrame, str]
        The same ``lr_catalog`` instance and the mass column for bin assignment.

    Raises
    ------
    ValueError
        If the method is unknown or not yet implemented.
    """
    lr_mass_column = config.matching.lr_mass_column
    if not config.enabled:
        return lr_catalog, lr_mass_column

    if config.method == MASS_CALIBRATION_METHOD_ABUNDANCE:
        _ = box_size_mpc_h
        _apply_abundance_matching(lr_catalog, hr_catalog, config)
        return lr_catalog, config.calibrated_column

    if config.method == MASS_CALIBRATION_METHOD_MATCHING_1TO1:
        _apply_matching_1to1(lr_catalog, hr_catalog, config, box_size_mpc_h)
        return lr_catalog, config.calibrated_column

    if config.method == MASS_CALIBRATION_METHOD_MATCHING_ML:
        raise ValueError(
            "Mass calibration method 'matching_ml' is not implemented yet; "
            "use 'abundance_matching' or wait for Phase 3."
        )

    raise ValueError(f"Unknown mass calibration method: {config.method}")


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
    config = MassCalibrationConfig(
        enabled=calibrate_mass,
        method=MASS_CALIBRATION_METHOD_ABUNDANCE,
        calibrated_column=calibrated_column,
        matching=MatchingHyperparameters(
            lr_mass_column=lr_mass_column,
            hr_mass_column=hr_mass_column,
        ),
    )
    return apply_mass_calibration(lr_catalog, hr_catalog, config, box_size_mpc_h=0.0)
