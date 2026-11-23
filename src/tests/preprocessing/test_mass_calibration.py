"""Tests for LR mass calibration."""

import numpy as np
import pandas as pd

from density_field_properties.preprocessing.mass_correction import (
    abundance_match_mass,
    apply_mass_correction,
    correct_lr_mass,
)
from density_field_properties.preprocessing.mass_correction.config import (
    default_mass_correction_settings,
)


def test_abundance_match_mass_volume_changes_result():
    """
    Different reference and target volumes should change the calibrated masses.
    """
    target = np.array([1.0e11, 5.0e11, 1.0e12])
    reference = np.linspace(1.0e11, 1.0e12, 20)
    reference_volume = 200.0**3
    target_volume = 100.0**3
    same_volume = abundance_match_mass(
        target,
        reference,
        reference_volume,
        reference_volume,
    )
    different_volumes = abundance_match_mass(
        target,
        reference,
        reference_volume,
        target_volume,
    )
    assert different_volumes.shape == target.shape
    assert not np.allclose(same_volume, different_volumes)


def test_apply_mass_correction_abundance_uses_pipeline_box_size():
    """
    Shared pipeline box size applies the same volume to HR and LR abundance matching.
    """
    hr = pd.DataFrame({"M200b": np.logspace(11.0, 12.5, 50)})
    lr = pd.DataFrame({"M200b": np.logspace(10.8, 12.2, 40)})
    no_box_lr, _ = apply_mass_correction(
        lr.copy(),
        hr,
        default_mass_correction_settings(enabled=True),
        box_size_mpc_h=0.0,
    )
    with_box_lr, _ = apply_mass_correction(
        lr.copy(),
        hr,
        default_mass_correction_settings(enabled=True),
        box_size_mpc_h=200.0,
    )
    assert np.allclose(no_box_lr["M200b_cal"], with_box_lr["M200b_cal"])
    assert np.all(np.isfinite(with_box_lr["M200b_cal"]))


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
    calibrated_lr, mass_column = correct_lr_mass(lr, hr, correct_mass=True)
    assert calibrated_lr is lr
    assert mass_column == "M200b_cal"
    assert "M200b_cal" in calibrated_lr.columns
    assert "M200b_cal" not in hr.columns
    pd.testing.assert_series_equal(hr["M200b"], hr_m200b)


def test_apply_mass_correction_dispatch_matches_legacy_abundance_matching():
    """
    Dispatcher with AM config should match ``correct_lr_mass`` output.
    """
    hr = pd.DataFrame({"M200b": np.logspace(11.0, 12.5, 50)})
    lr = pd.DataFrame({"M200b": np.logspace(10.8, 12.2, 40)})
    legacy_lr, legacy_column = correct_lr_mass(lr.copy(), hr, correct_mass=True)
    dispatched_lr, dispatched_column = apply_mass_correction(
        lr.copy(),
        hr,
        default_mass_correction_settings(enabled=True),
        box_size_mpc_h=0.0,
    )
    assert legacy_column == dispatched_column
    pd.testing.assert_series_equal(legacy_lr["M200b_cal"], dispatched_lr["M200b_cal"])


def test_mass_calibration_disabled_keeps_lr_mass_column():
    """
    When calibration is disabled, LR bin assignment uses raw M200b.
    """
    hr = pd.DataFrame({"M200b": [1.0e12, 2.0e12]})
    lr = pd.DataFrame({"M200b": [5.0e11, 6.0e11]})
    calibrated_lr, mass_column = correct_lr_mass(lr, hr, correct_mass=False)
    assert mass_column == "M200b"
    assert "M200b_cal" not in calibrated_lr.columns
