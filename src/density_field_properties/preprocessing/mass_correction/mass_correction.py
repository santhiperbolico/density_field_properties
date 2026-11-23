"""LR mass correction dispatch for Haloscope preprocessing."""

import numpy as np
import pandas as pd

from density_field_properties.preprocessing.mass_correction.abundance_correction import (
    apply_abundance_matching,
)
from density_field_properties.preprocessing.mass_correction.config import (
    MASS_CORRECTION_METHOD_ABUNDANCE,
    MASS_CORRECTION_METHOD_MATCHING_1TO1,
    MASS_CORRECTION_METHOD_MATCHING_ML,
    MASS_CORRECTION_METHODS,
    MassCorrectionSettings,
    MatchingHyperparameters,
)
from density_field_properties.preprocessing.mass_correction.hmf_reliable_mass import (
    estimate_fastpm_hmf_elbow_min_mass_msun_h,
)
from density_field_properties.preprocessing.mass_correction.matching_correction import (
    apply_matching_1to1,
)
from density_field_properties.preprocessing.mass_correction.ml_correction import apply_matching_ml


def resolve_lr_correction_mass_floor_msun_h(
    config: MassCorrectionSettings,
    lr_catalog: pd.DataFrame,
    box_size_mpc_h: float,
) -> float | None:
    """
    Resolve the raw LR mass floor below which correction is not applied.

    Parameters
    ----------
    config : MassCorrectionSettings
        Mass correction settings.
    lr_catalog : pd.DataFrame
        FastPM target catalog with raw masses.
    box_size_mpc_h : float
        Periodic box side in Mpc/h.

    Returns
    -------
    float or None
        Mass floor in Msun/h, or ``None`` when no floor is configured.
    """
    if config.lr_correction_min_mass_msun_h is not None:
        return float(config.lr_correction_min_mass_msun_h)
    if not config.auto_lr_correction_min_mass_from_hmf:
        return None
    if box_size_mpc_h <= 0.0:
        return None
    lr_masses = lr_catalog[config.matching.lr_mass_column].to_numpy(dtype=float)
    volume = float(box_size_mpc_h) ** 3

    return estimate_fastpm_hmf_elbow_min_mass_msun_h(
        lr_masses,
        volume,
        min_halo_count_per_bin=config.hmf_lr_floor_min_count,
    )


def mask_corrected_mass_below_lr_floor(
    lr_catalog: pd.DataFrame,
    config: MassCorrectionSettings,
    mass_floor_msun_h: float | None,
) -> None:
    """
    Clear corrected masses for LR halos below the raw-mass reliability floor.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution catalog with raw and corrected mass columns.
    config : MassCorrectionSettings
        Correction settings including column names.
    mass_floor_msun_h : float or None
        Minimum raw LR mass for which correction is trusted.
    """
    if mass_floor_msun_h is None:
        return
    raw_col = config.matching.lr_mass_column
    corrected_col = config.corrected_column
    if corrected_col not in lr_catalog.columns:
        return
    raw_masses = lr_catalog[raw_col].to_numpy(dtype=float)
    below_floor = raw_masses < float(mass_floor_msun_h)
    if not np.any(below_floor):
        return
    corrected = lr_catalog[corrected_col].to_numpy(dtype=float, copy=True)
    corrected[below_floor] = np.nan
    lr_catalog[corrected_col] = corrected


def apply_mass_correction(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCorrectionSettings,
    box_size_mpc_h: float,
) -> tuple[pd.DataFrame, str]:
    """
    Correct LR halo masses for Haloscope bin assignment.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog (FastPM).
    hr_catalog : pd.DataFrame
        High-resolution training catalog (UNIT).
    config : MassCorrectionSettings
        Method key and hyperparameters.
    box_size_mpc_h : float
        Periodic box side in Mpc/h for spatial and abundance matching.

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

    mass_floor = resolve_lr_correction_mass_floor_msun_h(
        config,
        lr_catalog,
        box_size_mpc_h,
    )

    if config.method == MASS_CORRECTION_METHOD_ABUNDANCE:
        apply_abundance_matching(lr_catalog, hr_catalog, config, box_size_mpc_h)
        mask_corrected_mass_below_lr_floor(lr_catalog, config, mass_floor)
        return lr_catalog, config.corrected_column

    if config.method == MASS_CORRECTION_METHOD_MATCHING_1TO1:
        apply_matching_1to1(lr_catalog, hr_catalog, config, box_size_mpc_h)
        mask_corrected_mass_below_lr_floor(lr_catalog, config, mass_floor)
        return lr_catalog, config.corrected_column

    if config.method == MASS_CORRECTION_METHOD_MATCHING_ML:
        apply_matching_ml(lr_catalog, hr_catalog, config, box_size_mpc_h)
        mask_corrected_mass_below_lr_floor(lr_catalog, config, mass_floor)
        return lr_catalog, config.corrected_column

    raise ValueError(f"Unknown mass correction method: {config.method}")


def correct_lr_mass(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    correct_mass: bool,
    lr_mass_column: str = "M200b",
    hr_mass_column: str = "M200b",
    corrected_column: str = "M200b_cal",
) -> tuple[pd.DataFrame, str]:
    """
    Optionally abundance-match LR masses to the HR mass function.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog (FastPM).
    hr_catalog : pd.DataFrame
        High-resolution training catalog (UNIT).
    correct_mass : bool
        If True, write ``corrected_column`` on ``lr_catalog``.
    lr_mass_column : str, optional
        Raw LR mass column.
    hr_mass_column : str, optional
        HR reference mass column.
    corrected_column : str, optional
        Output corrected mass column on LR.

    Returns
    -------
    tuple[pd.DataFrame, str]
        The same ``lr_catalog`` instance and the mass column for bin assignment.
    """
    settings = MassCorrectionSettings(
        enabled=correct_mass,
        method=MASS_CORRECTION_METHOD_ABUNDANCE,
        corrected_column=corrected_column,
        matching=MatchingHyperparameters(
            lr_mass_column=lr_mass_column,
            hr_mass_column=hr_mass_column,
        ),
    )
    return apply_mass_correction(lr_catalog, hr_catalog, settings, box_size_mpc_h=0.0)


__all__ = [
    "MASS_CORRECTION_METHODS",
    "MassCorrectionSettings",
    "apply_mass_correction",
    "correct_lr_mass",
    "mask_corrected_mass_below_lr_floor",
]
