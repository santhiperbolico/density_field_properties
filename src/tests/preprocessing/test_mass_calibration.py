"""Tests for LR mass calibration."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.mass_calibration import (
    abundance_match_mass,
    apply_mass_calibration,
    calibrate_lr_mass,
)
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MassCalibrationConfig,
    default_mass_calibration_config,
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


def test_apply_mass_calibration_dispatch_matches_legacy_abundance_matching():
    """
    Dispatcher with AM config should match ``calibrate_lr_mass`` output.
    """
    hr = pd.DataFrame({"M200b": np.logspace(11.0, 12.5, 50)})
    lr = pd.DataFrame({"M200b": np.logspace(10.8, 12.2, 40)})
    legacy_lr, legacy_column = calibrate_lr_mass(lr.copy(), hr, calibrate_mass=True)
    dispatched_lr, dispatched_column = apply_mass_calibration(
        lr.copy(),
        hr,
        default_mass_calibration_config(enabled=True),
        box_size_mpc_h=1000.0,
    )
    assert legacy_column == dispatched_column
    pd.testing.assert_series_equal(legacy_lr["M200b_cal"], dispatched_lr["M200b_cal"])


def test_apply_mass_calibration_rejects_unimplemented_matching_1to1():
    """
    ``matching_1to1`` should raise until Phase 2 is implemented.
    """
    hr = pd.DataFrame({"M200b": [1.0e12], "x": [1.0], "y": [1.0], "z": [1.0]})
    lr = pd.DataFrame({"M200b": [9.0e11], "x": [2.0], "y": [2.0], "z": [2.0]})
    config = MassCalibrationConfig(
        enabled=True,
        method=MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    )
    with pytest.raises(ValueError, match="matching_1to1"):
        apply_mass_calibration(lr, hr, config, box_size_mpc_h=1000.0)


def test_mass_calibration_disabled_keeps_lr_mass_column():
    """
    When calibration is disabled, LR bin assignment uses raw M200b.
    """
    hr = pd.DataFrame({"M200b": [1.0e12, 2.0e12]})
    lr = pd.DataFrame({"M200b": [5.0e11, 6.0e11]})
    calibrated_lr, mass_column = calibrate_lr_mass(lr, hr, calibrate_mass=False)
    assert mass_column == "M200b"
    assert "M200b_cal" not in calibrated_lr.columns
