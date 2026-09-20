"""Tests for preprocessing table schema validation."""

import pandas as pd
import pytest

from density_field_properties.preprocessing.schemas import (
    SchemaValidationError,
    validate_hr_training_table,
    validate_lr_target_table,
)


def _hr_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "M200b": [1.0e12],
            "env": [0.5],
            "cv": [0.4],
            "Spin": [0.03],
            "ca": [0.8],
            "ba": [0.6],
        }
    )


def _lr_frame(calibrated: bool = False) -> pd.DataFrame:
    data = {
        "x": [1.0],
        "y": [2.0],
        "z": [3.0],
        "M200b": [1.0e12],
        "env": [0.5],
    }
    if calibrated:
        data["M200b_cal"] = [1.1e12]
    return pd.DataFrame(data)


def test_schemas_reject_missing_columns():
    """
    Schema validation fails when required INPUT columns are absent.
    """
    frame = _hr_frame().drop(columns=["env"])
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_hr_training_table(frame, ("env",))


def test_preprocess_env_only_without_tidal_artifacts():
    """
    Env-only mode validates HR/LR tables without tidal input columns.
    """
    validate_hr_training_table(_hr_frame(), ("env",))
    validate_lr_target_table(_lr_frame(), ("env",), calibrate_mass=False)


def test_preprocess_tidal_requires_input_columns():
    """
    Tidal mode requires t_over_u and tidal_anisotropy on both tables.
    """
    hr = _hr_frame()
    lr = _lr_frame(calibrated=True)
    hr["t_over_u"] = [0.9]
    hr["tidal_anisotropy"] = [0.4]
    lr["t_over_u"] = [0.8]
    lr["tidal_anisotropy"] = [0.3]
    validate_hr_training_table(hr, ("t_over_u", "tidal_anisotropy"))
    validate_lr_target_table(
        lr,
        ("t_over_u", "tidal_anisotropy"),
        calibrate_mass=True,
    )

    lr_missing = lr.drop(columns=["tidal_anisotropy"])
    with pytest.raises(SchemaValidationError):
        validate_lr_target_table(
            lr_missing,
            ("t_over_u", "tidal_anisotropy"),
            calibrate_mass=True,
        )
