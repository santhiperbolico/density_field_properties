"""Spatial 1:1 HR–LR mass correction via halo matching."""

import numpy as np
import pandas as pd

from density_field_properties.preprocessing.halo_matching import (
    UNMATCHED_INDEX,
    hr_masses_for_matched_lr,
    match_hr_to_lr_one_to_one,
)
from density_field_properties.preprocessing.mass_correction.config import (
    MASS_CORRECTION_METHOD_MATCHING_1TO1,
    MassCorrectionSettings,
)


def apply_matching_1to1(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCorrectionSettings,
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
    config : MassCorrectionSettings
        Mass correction settings including matching hyperparameters.
    box_size_mpc_h : float
        Periodic box side length in Mpc/h.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with corrected mass and optional trace columns.

    Raises
    ------
    ValueError
        If every LR corrected mass is NaN after matching.
    """
    matching = match_hr_to_lr_one_to_one(
        hr_catalog,
        lr_catalog,
        box_size_mpc_h=box_size_mpc_h,
        params=config.matching,
    )
    hr_mass_column = config.matching.hr_mass_column
    hr_masses = hr_catalog[hr_mass_column].to_numpy(dtype=float)
    corrected = hr_masses_for_matched_lr(matching, hr_masses)

    lr_catalog[config.corrected_column] = corrected
    matched_mask = matching.lr_to_hr_index != UNMATCHED_INDEX
    lr_catalog["mass_corr_matched"] = matched_mask
    lr_catalog["mass_corr_method"] = MASS_CORRECTION_METHOD_MATCHING_1TO1

    if "id" in hr_catalog.columns:
        hr_ids = hr_catalog["id"].to_numpy()
        partner_ids = np.full(len(lr_catalog), np.nan, dtype=float)
        partner_ids[matched_mask] = hr_ids[matching.lr_to_hr_index[matched_mask]]
        lr_catalog["mass_corr_hr_id"] = partner_ids

    if not np.any(np.isfinite(corrected)):
        raise ValueError("All LR corrected masses are NaN after matching_1to1.")

    return lr_catalog
