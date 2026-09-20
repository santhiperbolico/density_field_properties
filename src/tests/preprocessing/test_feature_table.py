"""Tests for unified Haloscope feature table builder."""

import pandas as pd

from density_field_properties.preprocessing.context import (
    PreprocessingContext,
    SimulationRunContext,
)
from density_field_properties.preprocessing.feature_table import build_feature_tables


def _sim_catalog() -> pd.DataFrame:
    return pd.DataFrame(
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


def _fastpm_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [150.0, 250.0],
            "y": [150.0, 250.0],
            "z": [150.0, 250.0],
            "M200b": [8.0e11, 1.5e12],
        }
    )


def _env_context(calibrate_mass: bool) -> PreprocessingContext:
    return PreprocessingContext(
        sim=SimulationRunContext(boxsize_mpc_h=1000.0, env_radius_mpc_h=5.0),
        fastpm=SimulationRunContext(boxsize_mpc_h=1000.0, env_radius_mpc_h=5.0),
        calibrate_mass=calibrate_mass,
    )


def test_build_feature_tables_env_only():
    """
    Env-only INPUT features attach env and calibrate LR mass when requested.
    """
    hr, lr, mass_column = build_feature_tables(
        _sim_catalog(),
        _fastpm_catalog(),
        ("env",),
        _env_context(calibrate_mass=True),
    )
    assert "env" in hr.columns
    assert "env" in lr.columns
    assert mass_column == "M200b_cal"


def test_build_feature_tables_env_only_without_mass_calibration():
    """
    Env-only mode works without LR mass calibration.
    """
    hr, lr, mass_column = build_feature_tables(
        _sim_catalog(),
        _fastpm_catalog(),
        ("env",),
        _env_context(calibrate_mass=False),
    )
    assert mass_column == "M200b"
    assert "M200b_cal" not in lr.columns
