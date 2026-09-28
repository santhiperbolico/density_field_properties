"""Tests for matching_1to1 mass calibration."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.context import (
    PreprocessingContext,
    SimulationRunContext,
)
from density_field_properties.preprocessing.feature_table import build_feature_tables
from density_field_properties.preprocessing.mass_calibration import apply_mass_calibration
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MassCalibrationConfig,
    MatchingHyperparameters,
)


def _matching_config(**overrides) -> MassCalibrationConfig:
    """
    Build a ``matching_1to1`` mass calibration config for tests.

    Parameters
    ----------
    **overrides
        Optional overrides for ``MatchingHyperparameters`` fields.

    Returns
    -------
    MassCalibrationConfig
        Enabled matching calibration settings.
    """
    matching = MatchingHyperparameters(**overrides) if overrides else MatchingHyperparameters()
    return MassCalibrationConfig(
        enabled=True,
        method=MASS_CALIBRATION_METHOD_MATCHING_1TO1,
        matching=matching,
    )


def test_matching_1to1_writes_hr_mass_and_trace_columns():
    """
    Matched LR halos should receive HR ``M200b`` and provenance columns.
    """
    hr = pd.DataFrame(
        {
            "id": [101],
            "x": [10.0],
            "y": [10.0],
            "z": [10.0],
            "M200b": [1.25e12],
        }
    )
    lr = pd.DataFrame(
        {
            "x": [10.2],
            "y": [10.0],
            "z": [10.0],
            "M200b": [9.0e11],
        }
    )
    config = _matching_config(k_neighbours=3, r_match_mpc_h=5.0)
    calibrated_lr, mass_column = apply_mass_calibration(lr, hr, config, box_size_mpc_h=100.0)

    assert mass_column == "M200b_cal"
    assert calibrated_lr["M200b_cal"].iloc[0] == pytest.approx(1.25e12)
    assert calibrated_lr["M200b"].iloc[0] == pytest.approx(9.0e11)
    assert bool(calibrated_lr["mass_calib_matched"].iloc[0])
    assert calibrated_lr["mass_calib_method"].iloc[0] == MASS_CALIBRATION_METHOD_MATCHING_1TO1
    assert calibrated_lr["mass_calib_hr_id"].iloc[0] == pytest.approx(101.0)


def test_matching_1to1_unmatched_lr_gets_nan_calibrated_mass():
    """
    LR halos without an HR partner should keep ``M200b_cal`` as NaN.
    """
    hr = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "M200b": [1.0e12]})
    lr = pd.DataFrame(
        {
            "x": [0.1, 50.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "M200b": [9.5e11, 8.0e11],
        }
    )
    config = _matching_config(r_match_mpc_h=2.0)
    calibrated_lr, _ = apply_mass_calibration(lr.copy(), hr, config, box_size_mpc_h=100.0)

    assert np.isfinite(calibrated_lr["M200b_cal"].iloc[0])
    assert np.isnan(calibrated_lr["M200b_cal"].iloc[1])
    assert calibrated_lr["mass_calib_matched"].tolist() == [True, False]


def test_matching_1to1_raises_when_all_lr_masses_are_nan():
    """
    Calibration should fail when no LR halo receives a finite calibrated mass.
    """
    hr = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "M200b": [1.0e12]})
    lr = pd.DataFrame({"x": [40.0], "y": [0.0], "z": [0.0], "M200b": [9.0e11]})
    config = _matching_config(r_match_mpc_h=1.0)
    with pytest.raises(ValueError, match="All LR calibrated masses are NaN"):
        apply_mass_calibration(lr, hr, config, box_size_mpc_h=100.0)


def test_build_feature_tables_with_matching_1to1():
    """
    Feature table builder should attach env and apply matching calibration.
    """
    hr = pd.DataFrame(
        {
            "id": [1, 2],
            "x": [100.0, 200.0],
            "y": [100.0, 200.0],
            "z": [100.0, 200.0],
            "M200b": [1.0e12, 2.0e12],
            "cv": [0.4, 0.5],
            "Spin": [0.03, 0.04],
            "ca": [0.8, 0.7],
            "ba": [0.6, 0.5],
        }
    )
    lr = pd.DataFrame(
        {
            "x": [100.5, 250.0],
            "y": [100.0, 200.0],
            "z": [100.0, 200.0],
            "M200b": [9.5e11, 1.4e12],
        }
    )
    context = PreprocessingContext(
        sim=SimulationRunContext(boxsize_mpc_h=1000.0, env_radius_mpc_h=5.0),
        fastpm=SimulationRunContext(boxsize_mpc_h=1000.0, env_radius_mpc_h=5.0),
        mass_calibration=_matching_config(k_neighbours=5, r_match_mpc_h=10.0),
    )
    hr_table, lr_table, mass_column = build_feature_tables(hr, lr, ("env",), context)

    assert mass_column == "M200b_cal"
    assert "env" in lr_table.columns
    assert np.isfinite(lr_table["M200b_cal"].iloc[0])
    assert lr_table["M200b_cal"].iloc[0] == pytest.approx(1.0e12)
