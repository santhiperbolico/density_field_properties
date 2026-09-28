"""Tests for matching_ml mass calibration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.mass_calibration import apply_mass_calibration
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_MATCHING_ML,
    MassCalibrationConfig,
    MatchingHyperparameters,
    MatchingMlConfig,
    RegressorConfig,
    TrainSubboxConfig,
)
from density_field_properties.preprocessing.mass_calibration_ml import (
    REGRESSOR_SKLEARN_RANDOM_FOREST,
    apply_matching_ml,
    build_ml_feature_matrix,
    create_regressor,
    load_matching_ml_model,
    mask_subbox,
)
from density_field_properties.preprocessing.schemas import SchemaValidationError


def _ml_config(
    train_subbox: TrainSubboxConfig,
    ml_features: tuple[str, ...] = ("log10_M200b", "env"),
) -> MassCalibrationConfig:
    """
    Build a minimal ``matching_ml`` configuration for unit tests.

    Parameters
    ----------
    train_subbox : TrainSubboxConfig
        Train region bounds.
    ml_features : tuple[str, ...], optional
        LR feature names for the regressor.

    Returns
    -------
    MassCalibrationConfig
        Enabled ML calibration settings.
    """
    return MassCalibrationConfig(
        enabled=True,
        method=MASS_CALIBRATION_METHOD_MATCHING_ML,
        matching=MatchingHyperparameters(
            k_neighbours=5,
            r_match_mpc_h=5.0,
            max_matching_iterations=5,
        ),
        matching_ml=MatchingMlConfig(
            train_subbox=train_subbox,
            ml_features=ml_features,
            regressor=RegressorConfig(
                regressor_type=REGRESSOR_SKLEARN_RANDOM_FOREST,
                params={"n_estimators": 5, "max_depth": 8, "random_state": 0},
            ),
        ),
    )


def _paired_catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build tiny HR/LR catalogs with one obvious spatial pair in the train box.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        HR and LR tables.
    """
    hr = pd.DataFrame(
        {
            "x": [10.0, 80.0],
            "y": [10.0, 10.0],
            "z": [10.0, 10.0],
            "M200b": [1.2e12, 2.0e12],
        }
    )
    lr = pd.DataFrame(
        {
            "x": [10.3, 50.0],
            "y": [10.0, 10.0],
            "z": [10.0, 10.0],
            "M200b": [1.0e12, 9.0e11],
            "env": [0.5, 0.2],
        }
    )
    return hr, lr


def test_mask_subbox_excludes_outside_halos():
    """
    Sub-box masking should keep only halos within axis-aligned bounds.
    """
    catalog = pd.DataFrame({"x": [1.0, 60.0], "y": [0.0, 0.0], "z": [0.0, 0.0]})
    subbox = TrainSubboxConfig(x_max_mpc_h=50.0, y_max_mpc_h=100.0, z_max_mpc_h=100.0)
    mask = mask_subbox(catalog, subbox)
    assert mask.tolist() == [True, False]


def test_create_regressor_sklearn_random_forest_has_two_outputs():
    """
    sklearn RF should support joint keep and log-mass targets.
    """
    regressor = create_regressor(RegressorConfig(regressor_type=REGRESSOR_SKLEARN_RANDOM_FOREST))
    features = np.array([[0.1, 0.2], [0.3, 0.4]])
    targets = np.array([[1.0, 12.0], [0.0, 0.0]])
    regressor.fit(features, targets)
    predictions = regressor.predict(features)
    assert predictions.shape == (2, 2)


def test_build_ml_feature_matrix_derives_log10_m200b():
    """
    ``log10_M200b`` should be derived from ``M200b`` when omitted as a column.
    """
    lr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    matrix = build_ml_feature_matrix(lr, ("log10_M200b", "env"), lr_mass_column="M200b")
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == pytest.approx(12.0)


def test_build_ml_feature_matrix_raises_on_missing_column():
    """
    Missing configured features should raise ``SchemaValidationError``.
    """
    lr = pd.DataFrame({"M200b": [1.0e12]})
    with pytest.raises(SchemaValidationError, match="delta"):
        build_ml_feature_matrix(lr, ("delta",), lr_mass_column="M200b")


def test_apply_matching_ml_writes_calibrated_mass_and_p_keep(tmp_path: Path):
    """
    ML calibration should produce finite ``M200b_cal`` and keep scores on LR.
    """
    hr, lr = _paired_catalogs()
    subbox = TrainSubboxConfig(x_max_mpc_h=100.0, y_max_mpc_h=100.0, z_max_mpc_h=100.0)
    config = _ml_config(train_subbox=subbox)
    calibrated_lr = apply_matching_ml(lr.copy(), hr, config, box_size_mpc_h=100.0)

    assert np.isfinite(calibrated_lr["M200b_cal"]).any()
    assert "mass_calib_p_keep" in calibrated_lr.columns
    assert calibrated_lr["mass_calib_method"].iloc[0] == MASS_CALIBRATION_METHOD_MATCHING_ML


def test_saved_model_reproduces_predictions(tmp_path: Path):
    """
    Reloading a joblib bundle should reproduce inference on the same LR table.
    """
    hr, lr = _paired_catalogs()
    subbox = TrainSubboxConfig(x_max_mpc_h=100.0, y_max_mpc_h=100.0, z_max_mpc_h=100.0)
    model_path = tmp_path / "mass_calib_ml.joblib"
    fit_config = _ml_config(train_subbox=subbox)
    fit_config = replace(
        fit_config,
        matching_ml=replace(
            fit_config.matching_ml,
            model_path=model_path,
            write_model=True,
        ),
    )
    first_pass = apply_matching_ml(lr.copy(), hr, fit_config, box_size_mpc_h=100.0)

    infer_config = replace(
        fit_config,
        matching_ml=replace(
            fit_config.matching_ml,
            write_model=False,
        ),
    )
    second_pass = apply_matching_ml(lr.copy(), hr, infer_config, box_size_mpc_h=100.0)

    pd.testing.assert_series_equal(first_pass["M200b_cal"], second_pass["M200b_cal"])
    regressor, features = load_matching_ml_model(model_path)
    assert features == ("log10_M200b", "env")
    assert regressor is not None


def test_apply_mass_calibration_dispatcher_matching_ml():
    """
    ``apply_mass_calibration`` should route to the ML backend.
    """
    hr, lr = _paired_catalogs()
    subbox = TrainSubboxConfig(x_max_mpc_h=100.0, y_max_mpc_h=100.0, z_max_mpc_h=100.0)
    config = _ml_config(train_subbox=subbox)
    calibrated_lr, mass_column = apply_mass_calibration(
        lr.copy(), hr, config, box_size_mpc_h=100.0
    )
    assert mass_column == "M200b_cal"
    assert np.isfinite(calibrated_lr["M200b_cal"]).any()
