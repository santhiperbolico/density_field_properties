"""Tests for LR mass correction wired through the pipeline config."""

import numpy as np
import pandas as pd

from density_field_properties.pipelines.config import (
    MassCorrectionConfig,
    PreprocessingConfig,
    SampleConfig,
    mass_correction_settings_from_preprocessing,
)
from density_field_properties.preprocessing.mass_correction.mass_correction import (
    apply_mass_correction,
    mask_corrected_mass_below_lr_floor,
)


def _preprocessing_config(get_min_mass_lr_from_hmf: bool) -> PreprocessingConfig:
    """
    Build a minimal preprocessing config for mass-correction tests.

    Parameters
    ----------
    get_min_mass_lr_from_hmf : bool
        Whether to derive the LR correction floor from the HMF knee.

    Returns
    -------
    PreprocessingConfig
        Preprocessing settings with abundance matching enabled.
    """
    return PreprocessingConfig(
        enabled=True,
        hr_parquet_name="hr.parquet",
        lr_parquet_name="lr.parquet",
        sample=SampleConfig(
            box_size_mpc_h=200.0,
            min_mass_msun_h=1.0e11,
            get_min_mass_lr_from_hmf=get_min_mass_lr_from_hmf,
            max_mass_msun_h=None,
            max_sim_halos=None,
            max_fastpm_halos=None,
        ),
        mass_correction=MassCorrectionConfig(
            method="abundance_matching",
            correction_column="M200b_cal",
            params_method={},
        ),
    )


def test_mass_correction_settings_from_preprocessing_sets_hmf_flag():
    """
    Sample ``get_min_mass_lr_from_hmf`` should map onto ``MassCorrectionSettings``.
    """
    prep = _preprocessing_config(get_min_mass_lr_from_hmf=True)
    correction = mass_correction_settings_from_preprocessing(prep)
    assert correction.auto_lr_correction_min_mass_from_hmf is True
    assert correction.corrected_column == "M200b_cal"


def test_mask_corrected_mass_below_lr_floor_clears_low_mass_rows():
    """
    LR halos below the raw-mass floor should get NaN calibrated masses.
    """
    prep = _preprocessing_config(get_min_mass_lr_from_hmf=False)
    correction = mass_correction_settings_from_preprocessing(prep)
    lr = pd.DataFrame({"M200b": [1.0e11, 5.0e11], "M200b_cal": [2.0e11, 6.0e11]})
    mask_corrected_mass_below_lr_floor(lr, correction, mass_floor_msun_h=2.0e11)
    assert np.isnan(lr.loc[0, "M200b_cal"])
    assert lr.loc[1, "M200b_cal"] == 6.0e11


def test_apply_mass_correction_abundance_writes_calibrated_column():
    """
    Abundance matching should populate the configured correction column.
    """
    prep = _preprocessing_config(get_min_mass_lr_from_hmf=False)
    correction = mass_correction_settings_from_preprocessing(prep)
    hr = pd.DataFrame({"M200b": np.logspace(11.5, 13.0, 50)})
    lr = pd.DataFrame({"M200b": np.logspace(11.0, 12.5, 50)})
    _, mass_col = apply_mass_correction(lr, hr, correction, box_size_mpc_h=200.0)
    assert mass_col == "M200b_cal"
    assert "M200b_cal" in lr.columns
    assert lr["M200b_cal"].notna().all()
