"""ML mass calibration labels, regressor fit, and inference for ``matching_ml``."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from density_field_properties.preprocessing.halo_matching import (
    UNMATCHED_INDEX,
    match_hr_to_lr_one_to_one,
)
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_MATCHING_ML,
    MassCalibrationConfig,
    MatchingHyperparameters,
    RegressorConfig,
    TrainSubboxConfig,
)
from density_field_properties.preprocessing.schemas import SchemaValidationError

REGRESSOR_SKLEARN_RANDOM_FOREST = "sklearn_random_forest"
REGRESSOR_SKLEARN_EXTRA_TREES = "sklearn_extra_trees"
REGRESSOR_XGBOOST = "xgboost"

TARGET_KEEP_INDEX = 0
TARGET_LOG_MASS_INDEX = 1
JOBLIB_MODEL_VERSION = 1


def mask_subbox(
    catalog: pd.DataFrame,
    subbox: TrainSubboxConfig,
) -> np.ndarray:
    """
    Build a boolean mask for halos inside an axis-aligned sub-box.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table with ``x``, ``y``, ``z`` in Mpc/h.
    subbox : TrainSubboxConfig
        Inclusive lower and exclusive upper bounds per axis.

    Returns
    -------
    np.ndarray
        True for rows inside the sub-box.
    """
    positions = catalog[["x", "y", "z"]].to_numpy(dtype=float)
    x_coord = positions[:, 0]
    y_coord = positions[:, 1]
    z_coord = positions[:, 2]
    inside_x = (x_coord >= subbox.x_min_mpc_h) & (x_coord < subbox.x_max_mpc_h)
    inside_y = (y_coord >= subbox.y_min_mpc_h) & (y_coord < subbox.y_max_mpc_h)
    inside_z = (z_coord >= subbox.z_min_mpc_h) & (z_coord < subbox.z_max_mpc_h)
    return inside_x & inside_y & inside_z


def _resolve_ml_feature_array(
    lr_catalog: pd.DataFrame,
    feature_name: str,
    lr_mass_column: str,
) -> np.ndarray:
    """
    Resolve one ML feature column, including simple derived names.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        LR table with attached INPUT features.
    feature_name : str
        Configured feature name.
    lr_mass_column : str
        Raw LR mass column for derived log-mass features.

    Returns
    -------
    np.ndarray
        Feature values aligned with ``lr_catalog`` row order.

    Raises
    ------
    SchemaValidationError
        If the feature cannot be resolved from ``lr_catalog``.
    """
    if feature_name in lr_catalog.columns:
        return lr_catalog[feature_name].to_numpy(dtype=float)
    if feature_name == "log10_M200b":
        if lr_mass_column not in lr_catalog.columns:
            raise SchemaValidationError(
                f"Cannot derive log10_M200b: missing LR mass column '{lr_mass_column}'."
            )
        masses = lr_catalog[lr_mass_column].to_numpy(dtype=float)
        return np.log10(masses)
    if feature_name == "log10_sigma_v":
        if "sigma_v" not in lr_catalog.columns:
            raise SchemaValidationError(
                "Cannot derive log10_sigma_v: missing column 'sigma_v' on LR catalog."
            )
        sigma = lr_catalog["sigma_v"].to_numpy(dtype=float)
        return np.log10(sigma)
    raise SchemaValidationError(
        f"LR catalog missing ML feature column '{feature_name}' "
        f"(available: {', '.join(sorted(lr_catalog.columns))})."
    )


def build_ml_feature_matrix(
    lr_catalog: pd.DataFrame,
    ml_features: tuple[str, ...],
    lr_mass_column: str,
) -> np.ndarray:
    """
    Stack configured ML features into a design matrix.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        LR table after INPUT feature attachment.
    ml_features : tuple[str, ...]
        Feature names from run configuration.
    lr_mass_column : str
        Raw LR mass column used for derived features.

    Returns
    -------
    np.ndarray
        Feature matrix with shape ``(n_lr, n_features)``.

    Raises
    ------
    SchemaValidationError
        If any configured feature is missing.
    """
    if not ml_features:
        raise SchemaValidationError("matching_ml.ml_features must list at least one feature.")
    columns = [
        _resolve_ml_feature_array(lr_catalog, feature_name, lr_mass_column)
        for feature_name in ml_features
    ]
    return np.column_stack(columns)


def build_matching_ml_training_targets(
    hr_catalog: pd.DataFrame,
    lr_catalog: pd.DataFrame,
    box_size_mpc_h: float,
    matching: MatchingHyperparameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Forero-style keep and log-mass targets from train-sub-box matching.

    Parameters
    ----------
    hr_catalog : pd.DataFrame
        HR halos inside the train sub-box.
    lr_catalog : pd.DataFrame
        LR halos inside the train sub-box.
    box_size_mpc_h : float
        Periodic box side in Mpc/h.
    matching : MatchingHyperparameters
        Shared matching hyperparameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``y_keep`` and ``y_log_m_hr`` arrays aligned with ``lr_catalog``.

    Raises
    ------
    ValueError
        If no matched LR–HR pairs exist in the train sub-box.
    """
    match_result = match_hr_to_lr_one_to_one(
        hr_catalog,
        lr_catalog,
        box_size_mpc_h=box_size_mpc_h,
        params=matching,
    )
    matched_mask = match_result.lr_to_hr_index != UNMATCHED_INDEX
    if not np.any(matched_mask):
        raise ValueError("No matched LR–HR pairs in train sub-box for matching_ml.")

    hr_masses = hr_catalog[matching.hr_mass_column].to_numpy(dtype=float)
    y_keep = np.zeros(len(lr_catalog), dtype=float)
    y_log_m_hr = np.zeros(len(lr_catalog), dtype=float)
    y_keep[matched_mask] = 1.0
    partner_indices = match_result.lr_to_hr_index[matched_mask]
    y_log_m_hr[matched_mask] = np.log10(hr_masses[partner_indices])
    return y_keep, y_log_m_hr


def create_regressor(regressor_config: RegressorConfig) -> Any:
    """
    Instantiate a multi-target regressor from the run configuration.

    Parameters
    ----------
    regressor_config : RegressorConfig
        Regressor type key and constructor parameters.

    Returns
    -------
    Any
        Fitted-capable sklearn (or wrapped xgboost) regressor.

    Raises
    ------
    ValueError
        If the regressor type is unknown or an optional dependency is missing.
    """
    regressor_type = regressor_config.regressor_type
    params = dict(regressor_config.params)
    if regressor_type == REGRESSOR_SKLEARN_RANDOM_FOREST:
        return RandomForestRegressor(**params)
    if regressor_type == REGRESSOR_SKLEARN_EXTRA_TREES:
        return ExtraTreesRegressor(**params)
    if regressor_type == REGRESSOR_XGBOOST:
        try:
            from sklearn.multioutput import MultiOutputRegressor
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ValueError(
                "xgboost is not installed; use sklearn_random_forest or install xgboost."
            ) from exc
        return MultiOutputRegressor(XGBRegressor(**params))
    raise ValueError(f"Unknown matching_ml regressor type: {regressor_type}")


def _config_fingerprint(config: MassCalibrationConfig) -> str:
    """
    Hash mass calibration settings for joblib bundle metadata.

    Parameters
    ----------
    config : MassCalibrationConfig
        Full mass calibration configuration.

    Returns
    -------
    str
        Short hexadecimal digest.
    """
    payload = {
        "method": config.method,
        "matching": asdict(config.matching),
        "matching_ml": {
            "train_subbox": asdict(config.matching_ml.train_subbox),
            "ml_features": list(config.matching_ml.ml_features),
            "regressor": {
                "type": config.matching_ml.regressor.regressor_type,
                "params": config.matching_ml.regressor.params,
            },
            "keep_threshold": config.matching_ml.keep_threshold,
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_matching_ml_model(
    model_path: Path,
    regressor: Any,
    ml_features: tuple[str, ...],
    config: MassCalibrationConfig,
) -> None:
    """
    Persist a fitted regressor and feature metadata with joblib.

    Parameters
    ----------
    model_path : Path
        Destination ``.joblib`` path.
    regressor : Any
        Fitted multi-target regressor.
    ml_features : tuple[str, ...]
        Feature column order used at fit time.
    config : MassCalibrationConfig
        Mass calibration settings for provenance.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": JOBLIB_MODEL_VERSION,
        "method": MASS_CALIBRATION_METHOD_MATCHING_ML,
        "ml_features": tuple(ml_features),
        "config_fingerprint": _config_fingerprint(config),
        "regressor": regressor,
    }
    joblib.dump(bundle, model_path)


def load_matching_ml_model(model_path: Path) -> tuple[Any, tuple[str, ...]]:
    """
    Load a joblib mass-calibration model bundle.

    Parameters
    ----------
    model_path : Path
        Path to a bundle written by ``save_matching_ml_model``.

    Returns
    -------
    tuple[Any, tuple[str, ...]]
        Regressor and feature name tuple.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    ValueError
        If the bundle format is invalid.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"matching_ml model not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "regressor" not in bundle:
        raise ValueError(f"Invalid matching_ml model bundle: {path}")
    ml_features = tuple(bundle.get("ml_features", ()))
    return bundle["regressor"], ml_features


def _stack_targets(y_keep: np.ndarray, y_log_m_hr: np.ndarray) -> np.ndarray:
    """
    Stack multi-target training labels for sklearn regressors.

    Parameters
    ----------
    y_keep : np.ndarray
        Keep probability targets.
    y_log_m_hr : np.ndarray
        log10 HR mass targets.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_samples, 2)``.
    """
    return np.column_stack((y_keep, y_log_m_hr))


def _predict_keep_and_log_mass(
    regressor: Any,
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict keep score and log HR mass from a fitted regressor.

    Parameters
    ----------
    regressor : Any
        Fitted multi-target model.
    feature_matrix : np.ndarray
        LR feature matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``p_keep`` and ``log10_m_hr_pred`` per LR row.
    """
    predictions = regressor.predict(feature_matrix)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    p_keep = np.clip(predictions[:, TARGET_KEEP_INDEX], 0.0, 1.0)
    log10_m_hr = predictions[:, TARGET_LOG_MASS_INDEX]
    return p_keep, log10_m_hr


def _apply_keep_threshold(
    calibrated_mass: np.ndarray,
    p_keep: np.ndarray,
    keep_threshold: Optional[float],
) -> np.ndarray:
    """
    Optionally mask calibrated masses below a keep-probability threshold.

    Parameters
    ----------
    calibrated_mass : np.ndarray
        Predicted LR masses in Msun/h.
    p_keep : np.ndarray
        Model keep scores.
    keep_threshold : Optional[float]
        When set, masses with ``p_keep`` below this become NaN.

    Returns
    -------
    np.ndarray
        Calibrated masses after optional filtering.
    """
    if keep_threshold is None:
        return calibrated_mass
    filtered = calibrated_mass.copy()
    filtered[p_keep < keep_threshold] = np.nan
    return filtered


def apply_matching_ml(
    lr_catalog: pd.DataFrame,
    hr_catalog: pd.DataFrame,
    config: MassCalibrationConfig,
    box_size_mpc_h: float,
) -> pd.DataFrame:
    """
    Fit or load an ML regressor and write calibrated masses on the LR catalog.

    Parameters
    ----------
    lr_catalog : pd.DataFrame
        Low-resolution target catalog with INPUT features attached.
    hr_catalog : pd.DataFrame
        High-resolution reference catalog.
    config : MassCalibrationConfig
        Mass calibration settings including ``matching_ml``.
    box_size_mpc_h : float
        Periodic box side in Mpc/h.

    Returns
    -------
    pd.DataFrame
        ``lr_catalog`` with calibrated mass and ML trace columns.

    Raises
    ------
    ValueError
        If training fails or all calibrated masses are NaN.
    SchemaValidationError
        If configured ML features are missing on the LR table.
    """
    ml_settings = config.matching_ml
    lr_mass_column = config.matching.lr_mass_column
    train_mask = mask_subbox(lr_catalog, ml_settings.train_subbox)
    hr_train_mask = mask_subbox(hr_catalog, ml_settings.train_subbox)
    lr_train = lr_catalog.loc[train_mask]
    hr_train = hr_catalog.loc[hr_train_mask]
    if len(lr_train) == 0 or len(hr_train) == 0:
        raise ValueError("Train sub-box contains no LR or HR halos for matching_ml.")

    model_path = ml_settings.model_path
    regressor = None
    feature_names = ml_settings.ml_features
    if model_path is not None and Path(model_path).is_file() and not ml_settings.write_model:
        regressor, feature_names = load_matching_ml_model(Path(model_path))
    else:
        y_keep, y_log_m_hr = build_matching_ml_training_targets(
            hr_train,
            lr_train,
            box_size_mpc_h=box_size_mpc_h,
            matching=config.matching,
        )
        train_features = build_ml_feature_matrix(lr_train, ml_settings.ml_features, lr_mass_column)
        regressor = create_regressor(ml_settings.regressor)
        regressor.fit(train_features, _stack_targets(y_keep, y_log_m_hr))
        if model_path is not None and ml_settings.write_model:
            save_matching_ml_model(
                Path(model_path),
                regressor,
                ml_settings.ml_features,
                config,
            )

    infer_mask = np.ones(len(lr_catalog), dtype=bool)
    if ml_settings.infer_subbox is not None:
        infer_mask = mask_subbox(lr_catalog, ml_settings.infer_subbox)

    infer_features = build_ml_feature_matrix(lr_catalog, feature_names, lr_mass_column)
    p_keep, log10_m_hr = _predict_keep_and_log_mass(regressor, infer_features)
    calibrated = np.full(len(lr_catalog), np.nan, dtype=float)
    calibrated[infer_mask] = 10.0 ** log10_m_hr[infer_mask]
    calibrated = _apply_keep_threshold(calibrated, p_keep, ml_settings.keep_threshold)

    lr_catalog[config.calibrated_column] = calibrated
    lr_catalog["mass_calib_p_keep"] = p_keep
    lr_catalog["mass_calib_method"] = MASS_CALIBRATION_METHOD_MATCHING_ML

    if not np.any(np.isfinite(calibrated)):
        raise ValueError("All LR calibrated masses are NaN after matching_ml.")

    return lr_catalog
