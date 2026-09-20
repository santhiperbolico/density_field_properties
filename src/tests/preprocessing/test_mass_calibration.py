"""Tests for LR mass calibration."""

import numpy as np
import pandas as pd

from density_field_properties.preprocessing.mass_calibration import (
    abundance_match_mass,
    calibrate_lr_mass,
)


def test_abundance_match_mass_preserves_order():
    """
    Abundance matching returns the same number of masses as the target sample.
    """
    target = np.array([1.0e11, 5.0e11, 1.0e12])
    reference = np.linspace(1.0e11, 1.0e12, 20)
    calibrated = abundance_match_mass(target, reference)
    assert calibrated.shape == target.shape
    assert np.all(np.diff(calibrated) >= 0)


def test_mass_calibration_applies_only_to_lr():
    """
    LR receives M200b_cal while HR keeps the original M200b column.
    """
    hr = pd.DataFrame({"M200b": np.logspace(11.0, 12.5, 50)})
    lr = pd.DataFrame({"M200b": np.logspace(10.8, 12.2, 40)})
    hr_m200b = hr["M200b"].copy()
    calibrated_lr, mass_column = calibrate_lr_mass(lr, hr, calibrate_mass=True)
    assert calibrated_lr is lr
    assert mass_column == "M200b_cal"
    assert "M200b_cal" in calibrated_lr.columns
    assert "M200b_cal" not in hr.columns
    pd.testing.assert_series_equal(hr["M200b"], hr_m200b)


def test_mass_calibration_disabled_keeps_lr_mass_column():
    """
    When calibration is disabled, LR bin assignment uses raw M200b.
    """
    hr = pd.DataFrame({"M200b": [1.0e12, 2.0e12]})
    lr = pd.DataFrame({"M200b": [5.0e11, 6.0e11]})
    calibrated_lr, mass_column = calibrate_lr_mass(lr, hr, calibrate_mass=False)
    assert mass_column == "M200b"
    assert "M200b_cal" not in calibrated_lr.columns
