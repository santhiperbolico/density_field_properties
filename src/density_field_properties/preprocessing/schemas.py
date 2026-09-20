"""Parquet table schema validation for Haloscope preprocessing."""

from typing import Sequence

import pandas as pd

HR_BASE_COLUMNS = ("id", "x", "y", "z", "M200b", "cv", "Spin")
LR_BASE_COLUMNS = ("x", "y", "z", "M200b")
TIDAL_INPUT_COLUMNS = ("t_over_u", "tidal_anisotropy")


class SchemaValidationError(ValueError):
    """Raised when a preprocessing table does not match the expected schema."""


def _missing_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    """
    Return required column names absent from a DataFrame.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to inspect.
    required_columns : Sequence[str]
        Column names that must be present.

    Returns
    -------
    list[str]
        Missing column names.
    """
    return [column for column in required_columns if column not in frame.columns]


def validate_hr_training_table(
    frame: pd.DataFrame,
    input_features: Sequence[str],
) -> None:
    """
    Validate the HR training table before Haloscope fit.

    Parameters
    ----------
    frame : pd.DataFrame
        High-resolution training catalog.
    input_features : Sequence[str]
        Haloscope input feature column names.

    Raises
    ------
    SchemaValidationError
        If required columns are missing or the table is empty.
    """
    required = list(HR_BASE_COLUMNS) + list(input_features)
    missing = _missing_columns(frame, required)
    if missing:
        raise SchemaValidationError(
            f"HR training table missing required columns: {', '.join(missing)}"
        )
    if len(frame) == 0:
        raise SchemaValidationError("HR training table is empty after preprocessing.")


def validate_lr_target_table(
    frame: pd.DataFrame,
    input_features: Sequence[str],
    calibrate_mass: bool = False,
    calibrated_mass_column: str = "M200b_cal",
) -> None:
    """
    Validate the LR target table before Haloscope predict.

    Parameters
    ----------
    frame : pd.DataFrame
        Low-resolution target catalog.
    input_features : Sequence[str]
        Haloscope input feature column names.
    calibrate_mass : bool, optional
        If True, require ``calibrated_mass_column``.
    calibrated_mass_column : str, optional
        Calibrated mass column name on LR.

    Raises
    ------
    SchemaValidationError
        If required columns are missing or the table is empty.
    """
    required = list(LR_BASE_COLUMNS) + list(input_features)
    if calibrate_mass:
        required.append(calibrated_mass_column)
    missing = _missing_columns(frame, required)
    if missing:
        raise SchemaValidationError(
            f"LR target table missing required columns: {', '.join(missing)}"
        )
    if len(frame) == 0:
        raise SchemaValidationError("LR target table is empty after preprocessing.")
