import numpy as np
import pandas as pd

from density_field_properties.preprocessing.mass_correction.config import (
    MASS_CORRECTION_METHOD_ABUNDANCE,
    MassCorrectionSettings,
)


def abundance_match_mass(
    mass_target: np.ndarray,
    mass_reference: np.ndarray,
    volume_reference: float | int | None = None,
    volume_target: float | int | None = None,
) -> np.ndarray:
    """
    Map target cumulative densities onto reference masses via log-log interpolation.

    Parameters
    ----------
    mass_target : np.ndarray
        Masses to remap (e.g. FastPM M200b).
    mass_reference : np.ndarray
        Reference mass sample (e.g. UNIT SIM M200b).
    volume_reference : float or int, optional
        Simulation volume in (Mpc/h)^3 for the reference sample.
    volume_target : float or int, optional
        Simulation volume in (Mpc/h)^3 for the target sample.

    Returns
    -------
    np.ndarray
        Calibrated masses with the same shape as ``mass_target``.
    """
    mass_reference_sorted = np.sort(mass_reference)[::-1]
    mass_target_argsort = np.argsort(mass_target)[::-1]

    n_reference_density = np.arange(len(mass_reference_sorted), dtype=np.float64) + 1.0
    n_target_density = np.arange(len(mass_target), dtype=np.float64) + 1.0

    condition_volume = int(volume_reference is None) + int(volume_target is None)
    if condition_volume == 1:
        raise ValueError("volume_reference and volume_target must both be set or both omitted.")
    elif condition_volume == 0:
        n_reference_density /= float(volume_reference)
        n_target_density /= float(volume_target)

    log_density_reference = np.log10(n_reference_density)
    log_density_target = np.log10(n_target_density)
    log_mass_reference = np.log10(mass_reference_sorted)
    log_mass_new = np.interp(
        log_density_target,
        log_density_reference,
        log_mass_reference,
    )
    new_masses = np.empty_like(mass_target)
    new_masses[mass_target_argsort] = np.power(10.0, log_mass_new)
    return new_masses


def apply_abundance_matching(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCorrectionSettings,
    box_size_mpc_h: float,
) -> pd.DataFrame:
    """
    Write abundance-matched masses on the LR catalog.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog.
    hr_catalog : pd.DataFrame
        High-resolution reference catalog.
    config : MassCorrectionSettings
        Calibration settings including column names.
    box_size_mpc_h : float
        Periodic box side in Mpc/h for number-density normalization.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with ``config.corrected_column`` populated.

    Raises
    ------
    ValueError
        If either mass sample is empty.
    """
    volume = None
    if box_size_mpc_h > 0.0:
        volume = float(box_size_mpc_h) ** 3

    lr_masses = lr_catalog[config.matching.lr_mass_column].to_numpy()
    hr_masses = hr_catalog[config.matching.hr_mass_column].to_numpy()
    if len(lr_masses) == 0 or len(hr_masses) == 0:
        raise ValueError("Empty HR or LR mass sample for abundance matching.")
    lr_catalog[config.corrected_column] = abundance_match_mass(
        lr_masses, hr_masses, volume, volume
    )
    lr_catalog["mass_corr_method"] = MASS_CORRECTION_METHOD_ABUNDANCE
    return lr_catalog
