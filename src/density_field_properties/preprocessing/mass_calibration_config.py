"""Mass calibration settings for Haloscope LR preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

MASS_CALIBRATION_METHOD_ABUNDANCE = "abundance_matching"
MASS_CALIBRATION_METHOD_MATCHING_1TO1 = "matching_1to1"
MASS_CALIBRATION_METHOD_MATCHING_ML = "matching_ml"

MASS_CALIBRATION_METHODS = (
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
)

DEFAULT_MATCHING_K_NEIGHBOURS = 10
DEFAULT_R_MATCH_MPC_H = 5.0
DEFAULT_MAX_MASS_DISTANCE = 0.95
DEFAULT_MAX_MATCHING_ITERATIONS = 5

DEFAULT_TRAIN_SUBBOX_X_MAX_MPC_H = 500.0
DEFAULT_TRAIN_SUBBOX_Y_MAX_MPC_H = 1000.0
DEFAULT_TRAIN_SUBBOX_Z_MAX_MPC_H = 1000.0

DEFAULT_ML_FEATURES = ("log10_M200b", "delta", "log10_sigma_v")
DEFAULT_REGRESSOR_TYPE = "sklearn_random_forest"
DEFAULT_REGRESSOR_PARAMS = {
    "n_estimators": 10,
    "max_depth": 64,
    "random_state": 42,
}


@dataclass(frozen=True)
class MatchingHyperparameters:
    """
    Hyperparameters for Forero-style HR–LR halo matching.

    Parameters
    ----------
    k_neighbours : int
        LR candidates per HR query in the kd-tree search.
    r_match_mpc_h : float
        Maximum comoving separation in Mpc/h.
    max_mass_distance : float
        Relative mass gap threshold |M_LR - M_HR| / M_HR.
    max_matching_iterations : int
        Conflict-resolution loop count.
    hr_mass_column : str
        HR mass column used in the distance metric.
    lr_mass_column : str
        LR mass column used in the distance metric.
    """

    k_neighbours: int = DEFAULT_MATCHING_K_NEIGHBOURS
    r_match_mpc_h: float = DEFAULT_R_MATCH_MPC_H
    max_mass_distance: float = DEFAULT_MAX_MASS_DISTANCE
    max_matching_iterations: int = DEFAULT_MAX_MATCHING_ITERATIONS
    hr_mass_column: str = "M200b"
    lr_mass_column: str = "M200b"


@dataclass(frozen=True)
class TrainSubboxConfig:
    """
    Axis-aligned train region for ``matching_ml`` label construction.

    Parameters
    ----------
    x_min_mpc_h : float
        Lower x bound in Mpc/h.
    x_max_mpc_h : float
        Upper x bound in Mpc/h.
    y_min_mpc_h : float
        Lower y bound in Mpc/h.
    y_max_mpc_h : float
        Upper y bound in Mpc/h.
    z_min_mpc_h : float
        Lower z bound in Mpc/h.
    z_max_mpc_h : float
        Upper z bound in Mpc/h.
    """

    x_min_mpc_h: float = 0.0
    x_max_mpc_h: float = DEFAULT_TRAIN_SUBBOX_X_MAX_MPC_H
    y_min_mpc_h: float = 0.0
    y_max_mpc_h: float = DEFAULT_TRAIN_SUBBOX_Y_MAX_MPC_H
    z_min_mpc_h: float = 0.0
    z_max_mpc_h: float = DEFAULT_TRAIN_SUBBOX_Z_MAX_MPC_H


@dataclass(frozen=True)
class RegressorConfig:
    """
    ML regressor backend selection for ``matching_ml``.

    Parameters
    ----------
    regressor_type : str
        Registry key (e.g. ``sklearn_random_forest``).
    params : dict[str, Any]
        Keyword arguments forwarded to the backend constructor.
    """

    regressor_type: str = DEFAULT_REGRESSOR_TYPE
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_REGRESSOR_PARAMS))


@dataclass(frozen=True)
class MatchingMlConfig:
    """
    Settings for ML mass calibration after sub-box matching labels.

    Parameters
    ----------
    train_subbox : TrainSubboxConfig
        Region where 1:1 matching builds training labels.
    ml_features : tuple[str, ...]
        LR feature column names for the regressor.
    regressor : RegressorConfig
        Backend type and constructor parameters.
    keep_threshold : Optional[float]
        When set, LR halos with ``p_keep`` below this get NaN calibrated mass.
    model_path : Optional[Path]
        Optional joblib artifact for load-or-fit behaviour.
    write_model : bool
        When True with ``model_path``, overwrite the artifact after fitting.
    infer_subbox : Optional[TrainSubboxConfig]
        Optional inference region; ``None`` means all loaded LR halos.
    """

    train_subbox: TrainSubboxConfig = field(default_factory=TrainSubboxConfig)
    ml_features: tuple[str, ...] = DEFAULT_ML_FEATURES
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    keep_threshold: Optional[float] = None
    model_path: Optional[Path] = None
    write_model: bool = False
    infer_subbox: Optional[TrainSubboxConfig] = None


@dataclass(frozen=True)
class MassCalibrationConfig:
    """
    LR mass calibration method and parameters for Haloscope preprocessing.

    Parameters
    ----------
    enabled : bool
        When False, LR bin assignment uses raw ``lr_mass_column``.
    method : str
        Calibration method key from ``MASS_CALIBRATION_METHODS``.
    calibrated_column : str
        Output column written on the LR table when enabled.
    matching : MatchingHyperparameters
        Shared matching hyperparameters for ``matching_1to1`` and ``matching_ml``.
    matching_ml : MatchingMlConfig
        ML-specific settings for ``matching_ml``.
    """

    enabled: bool = True
    method: str = MASS_CALIBRATION_METHOD_ABUNDANCE
    calibrated_column: str = "M200b_cal"
    matching: MatchingHyperparameters = field(default_factory=MatchingHyperparameters)
    matching_ml: MatchingMlConfig = field(default_factory=MatchingMlConfig)


def default_mass_calibration_config(enabled: Optional[bool] = None) -> MassCalibrationConfig:
    """
    Build default mass calibration settings (abundance matching).

    Parameters
    ----------
    enabled : Optional[bool]
        Whether LR mass calibration runs before Haloscope bin assignment. When
        ``None``, uses ``CALIBRATE_MASS`` from pipeline run defaults.

    Returns
    -------
    MassCalibrationConfig
        Fiducial configuration aligned with legacy ``CALIBRATE_MASS``.
    """
    from density_field_properties.pipelines.run_defaults import CALIBRATE_MASS

    resolved_enabled = CALIBRATE_MASS if enabled is None else enabled
    return MassCalibrationConfig(enabled=resolved_enabled)


def _parse_train_subbox(payload: Mapping[str, Any]) -> TrainSubboxConfig:
    """
    Parse train or inference sub-box bounds from JSON.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Sub-box object from run JSON.

    Returns
    -------
    TrainSubboxConfig
        Parsed axis-aligned bounds.
    """
    defaults = TrainSubboxConfig()
    return TrainSubboxConfig(
        x_min_mpc_h=float(payload.get("x_min_mpc_h", defaults.x_min_mpc_h)),
        x_max_mpc_h=float(payload.get("x_max_mpc_h", defaults.x_max_mpc_h)),
        y_min_mpc_h=float(payload.get("y_min_mpc_h", defaults.y_min_mpc_h)),
        y_max_mpc_h=float(payload.get("y_max_mpc_h", defaults.y_max_mpc_h)),
        z_min_mpc_h=float(payload.get("z_min_mpc_h", defaults.z_min_mpc_h)),
        z_max_mpc_h=float(payload.get("z_max_mpc_h", defaults.z_max_mpc_h)),
    )


def _parse_regressor(payload: Mapping[str, Any]) -> RegressorConfig:
    """
    Parse ML regressor settings from JSON.

    Parameters
    ----------
    payload : Mapping[str, Any]
        ``regressor`` object from run JSON.

    Returns
    -------
    RegressorConfig
        Parsed regressor type and params.
    """
    regressor_type = str(payload.get("type", DEFAULT_REGRESSOR_TYPE))
    raw_params = payload.get("params", DEFAULT_REGRESSOR_PARAMS)
    if raw_params is None:
        params = dict(DEFAULT_REGRESSOR_PARAMS)
    elif isinstance(raw_params, dict):
        params = dict(raw_params)
    else:
        raise ValueError("mass_calibration.matching_ml.regressor.params must be a JSON object.")
    return RegressorConfig(regressor_type=regressor_type, params=params)


def _parse_matching_ml(payload: Mapping[str, Any]) -> MatchingMlConfig:
    """
    Parse ``matching_ml`` nested settings from JSON.

    Parameters
    ----------
    payload : Mapping[str, Any]
        ``matching_ml`` object from run JSON.

    Returns
    -------
    MatchingMlConfig
        Parsed ML calibration settings.
    """
    train_subbox = _parse_train_subbox(payload.get("train_subbox", {}))
    infer_raw = payload.get("infer_subbox")
    infer_subbox = None if infer_raw is None else _parse_train_subbox(infer_raw)
    ml_features_raw = payload.get("ml_features")
    if ml_features_raw is None:
        ml_features = DEFAULT_ML_FEATURES
    else:
        ml_features = tuple(str(item) for item in ml_features_raw)
    regressor = _parse_regressor(payload.get("regressor", {}))
    model_path_raw = payload.get("model_path")
    model_path = None if model_path_raw in (None, "") else Path(str(model_path_raw))
    keep_threshold = payload.get("keep_threshold")
    if keep_threshold is not None:
        keep_threshold = float(keep_threshold)
    return MatchingMlConfig(
        train_subbox=train_subbox,
        ml_features=ml_features,
        regressor=regressor,
        keep_threshold=keep_threshold,
        model_path=model_path,
        write_model=bool(payload.get("write_model", False)),
        infer_subbox=infer_subbox,
    )


def _parse_matching(payload: Mapping[str, Any]) -> MatchingHyperparameters:
    """
    Parse shared matching hyperparameters from JSON.

    Parameters
    ----------
    payload : Mapping[str, Any]
        ``matching`` object from run JSON.

    Returns
    -------
    MatchingHyperparameters
        Parsed matching settings.
    """
    defaults = MatchingHyperparameters()
    return MatchingHyperparameters(
        k_neighbours=int(payload.get("k_neighbours", defaults.k_neighbours)),
        r_match_mpc_h=float(payload.get("r_match_mpc_h", defaults.r_match_mpc_h)),
        max_mass_distance=float(payload.get("max_mass_distance", defaults.max_mass_distance)),
        max_matching_iterations=int(
            payload.get("max_matching_iterations", defaults.max_matching_iterations)
        ),
        hr_mass_column=str(payload.get("hr_mass_column", defaults.hr_mass_column)),
        lr_mass_column=str(payload.get("lr_mass_column", defaults.lr_mass_column)),
    )


def parse_mass_calibration_config(
    payload: Mapping[str, Any],
    haloscope_payload: Mapping[str, Any],
    legacy_calibrate_mass: Any = None,
) -> MassCalibrationConfig:
    """
    Parse mass calibration settings from Haloscope run JSON.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Root JSON object.
    haloscope_payload : Mapping[str, Any]
        ``haloscope`` section of the run JSON.
    legacy_calibrate_mass : Any, optional
        Top-level ``calibrate_mass`` when present.

    Returns
    -------
    MassCalibrationConfig
        Parsed configuration with legacy fallbacks applied.

    Raises
    ------
    ValueError
        If ``method`` is not a supported calibration key.
    """
    mass_block = payload.get("mass_calibration")
    if mass_block is None:
        mass_block = {}
    if not isinstance(mass_block, dict):
        raise ValueError("mass_calibration must be a JSON object when provided.")

    enabled = mass_block.get("enabled")
    if enabled is None:
        haloscope_flag = haloscope_payload.get("calibrate_mass")
        if haloscope_flag is not None:
            enabled = bool(haloscope_flag)
        elif legacy_calibrate_mass is not None:
            enabled = bool(legacy_calibrate_mass)
        else:
            enabled = True
    else:
        enabled = bool(enabled)

    method_raw = mass_block.get("method")
    if method_raw is None:
        legacy_method = haloscope_payload.get("mass_calibration_method")
        method = (
            str(legacy_method) if legacy_method is not None else MASS_CALIBRATION_METHOD_ABUNDANCE
        )
    else:
        method = str(method_raw)

    if method not in MASS_CALIBRATION_METHODS:
        raise ValueError(
            f"Unknown mass calibration method '{method}'; "
            f"expected one of {', '.join(MASS_CALIBRATION_METHODS)}."
        )

    calibrated_column = str(mass_block.get("calibrated_column", "M200b_cal"))
    matching = _parse_matching(mass_block.get("matching", {}))
    matching_ml = _parse_matching_ml(mass_block.get("matching_ml", {}))

    return MassCalibrationConfig(
        enabled=enabled,
        method=method,
        calibrated_column=calibrated_column,
        matching=matching,
        matching_ml=matching_ml,
    )


def mass_calibration_from_legacy_bool(
    calibrate_mass: bool,
    calibrated_column: str = "M200b_cal",
) -> MassCalibrationConfig:
    """
    Build a minimal config from a legacy on/off flag.

    Parameters
    ----------
    calibrate_mass : bool
        Legacy calibration toggle.
    calibrated_column : str, optional
        Output column name when calibration is enabled.

    Returns
    -------
    MassCalibrationConfig
        Abundance-matching configuration.
    """
    return MassCalibrationConfig(
        enabled=calibrate_mass,
        method=MASS_CALIBRATION_METHOD_ABUNDANCE,
        calibrated_column=calibrated_column,
    )
