"""
Auxiliary functions for ``compare_mass_calibration_catalogs.py``.

Multi-catalog mass calibration benchmarks: config parsing, catalog I/O,
HMF metrics, and comparison figures.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from density_field_properties.pipelines.config import (
    HaloscopeEnrichmentConfig,
    load_haloscope_enrichment_config,
)
from density_field_properties.pipelines.run_defaults import ENV_RADIUS_MPC_H
from density_field_properties.preprocessing.catalog_loaders import (
    load_fastpm_fof_target_catalog,
    load_fastpm_rockstar_for_mass_calibration,
    load_unit_sim_training_catalog,
)
from density_field_properties.preprocessing.context import build_preprocessing_context
from density_field_properties.preprocessing.feature_table import _apply_attachers
from density_field_properties.preprocessing.filters import filter_mass_range
from density_field_properties.preprocessing.input_features.registry import resolve_attachers
from density_field_properties.preprocessing.mass_calibration import apply_mass_calibration
from density_field_properties.preprocessing.mass_calibration_config import (
    DEFAULT_TRAIN_SUBBOX_X_MAX_MPC_H,
    DEFAULT_TRAIN_SUBBOX_Y_MAX_MPC_H,
    DEFAULT_TRAIN_SUBBOX_Z_MAX_MPC_H,
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
    TrainSubboxConfig,
)
from density_field_properties.preprocessing.mass_calibration_ml import mask_subbox

FORERO_ML_FEATURE_NAMES = ("log10_M200b", "delta", "log10_sigma_v")

LOADER_ROCKSTAR_LIST = "rockstar_list"
LOADER_FASTPM_FOF = "fastpm_fof"
SUPPORTED_LR_LOADERS = (LOADER_ROCKSTAR_LIST, LOADER_FASTPM_FOF)

DEFAULT_BENCHMARK_METHODS = (
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
)

UNIT_HMF_LABEL = "UNIT_sim"
FASTPM_RAW_HMF_LABEL = "FastPM (uncalibrated)"
HR_MASS_COLUMN = "M200b"
LR_RAW_MASS_COLUMN = "M200b"

HMF_LOG10_BIN_MIN = 10.5
HMF_LOG10_BIN_MAX = 15.5
HMF_BIN_COUNT = 25
HMF_X_MAJOR_TICK = 0.5
HMF_X_MINOR_TICK = 0.25
HMF_Y_MAJOR_TICK = 1.0
HMF_Y_MINOR_TICK = 0.5

DEFAULT_BENCHMARK_HMF_MIN_MASS_MSUN_H = 2.0e11
DEFAULT_BENCHMARK_HMF_MAX_MASS_MSUN_H = 3.2e14


@dataclass(frozen=True)
class BenchmarkHmfConfig:
    """
    Mass range for benchmark HMF plots and RMSE metrics.

    Parameters
    ----------
    min_mass_msun_h : Optional[float]
        Lower mass cut in Msun/h (inclusive).
    max_mass_msun_h : Optional[float]
        Upper mass cut in Msun/h (inclusive).
    """

    min_mass_msun_h: Optional[float] = DEFAULT_BENCHMARK_HMF_MIN_MASS_MSUN_H
    max_mass_msun_h: Optional[float] = DEFAULT_BENCHMARK_HMF_MAX_MASS_MSUN_H


@dataclass(frozen=True)
class BenchmarkCatalogEntry:
    """
    One FastPM catalog entry for a mass-calibration benchmark run.

    Parameters
    ----------
    key : str
        Stable identifier used in output filenames.
    label : str
        Human-readable name for plots and summaries.
    path : Path
        Catalog path on disk.
    loader : str
        Loader key from ``SUPPORTED_LR_LOADERS``.
    """

    key: str
    label: str
    path: Path
    loader: str
    methods: Optional[tuple[str, ...]] = None
    train_subbox: Optional[TrainSubboxConfig] = None


def parse_benchmark_catalogs(
    payload: Mapping[str, Any],
    repo_root: Path,
) -> tuple[BenchmarkCatalogEntry, ...]:
    """
    Parse the ``benchmark_catalogs`` block from a benchmark JSON config.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Root JSON object.
    repo_root : Path
        Repository root for resolving relative catalog paths.

    Returns
    -------
    tuple[BenchmarkCatalogEntry, ...]
        Ordered catalog entries.

    Raises
    ------
    ValueError
        If the block is missing or an entry is invalid.
    """
    block = payload.get("benchmark_catalogs")
    if not isinstance(block, dict) or not block:
        raise ValueError("benchmark_catalogs must be a non-empty JSON object.")
    entries: list[BenchmarkCatalogEntry] = []
    for key, raw_entry in block.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"benchmark_catalogs.{key} must be a JSON object.")
        path_value = raw_entry.get("path")
        if path_value is None:
            raise ValueError(f"benchmark_catalogs.{key}.path is required.")
        loader = str(raw_entry.get("loader", LOADER_ROCKSTAR_LIST))
        if loader not in SUPPORTED_LR_LOADERS:
            raise ValueError(
                f"benchmark_catalogs.{key}.loader must be one of "
                f"{', '.join(SUPPORTED_LR_LOADERS)}."
            )
        label = str(raw_entry.get("label", key))
        path = Path(path_value)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        entry_methods = raw_entry.get("benchmark_methods")
        parsed_methods: Optional[tuple[str, ...]] = None
        if entry_methods is not None:
            if not isinstance(entry_methods, list) or not entry_methods:
                raise ValueError(
                    f"benchmark_catalogs.{key}.benchmark_methods must be a non-empty array."
                )
            parsed_methods = tuple(str(method) for method in entry_methods)
        train_subbox = _parse_optional_train_subbox(raw_entry.get("train_subbox"))
        entries.append(
            BenchmarkCatalogEntry(
                key=str(key),
                label=label,
                path=path,
                loader=loader,
                methods=parsed_methods,
                train_subbox=train_subbox,
            )
        )
    return tuple(entries)


def _parse_optional_train_subbox(raw_block: Any) -> Optional[TrainSubboxConfig]:
    """
    Parse an optional train sub-box override from benchmark catalog JSON.

    Parameters
    ----------
    raw_block : Any
        JSON value for ``train_subbox``.

    Returns
    -------
    Optional[TrainSubboxConfig]
        Parsed sub-box or ``None`` when absent.
    """
    if raw_block is None:
        return None
    if not isinstance(raw_block, dict):
        raise ValueError("train_subbox must be a JSON object when provided.")
    return TrainSubboxConfig(
        x_min_mpc_h=float(raw_block.get("x_min_mpc_h", 0.0)),
        x_max_mpc_h=float(raw_block.get("x_max_mpc_h", DEFAULT_TRAIN_SUBBOX_X_MAX_MPC_H)),
        y_min_mpc_h=float(raw_block.get("y_min_mpc_h", 0.0)),
        y_max_mpc_h=float(raw_block.get("y_max_mpc_h", DEFAULT_TRAIN_SUBBOX_Y_MAX_MPC_H)),
        z_min_mpc_h=float(raw_block.get("z_min_mpc_h", 0.0)),
        z_max_mpc_h=float(raw_block.get("z_max_mpc_h", DEFAULT_TRAIN_SUBBOX_Z_MAX_MPC_H)),
    )


def train_subbox_has_hr_and_lr_halos(
    hr_table: pd.DataFrame,
    lr_table: pd.DataFrame,
    train_subbox: TrainSubboxConfig,
) -> bool:
    """
    Return whether both HR and LR tables contain halos inside the train sub-box.

    Parameters
    ----------
    hr_table : pd.DataFrame
        UNIT reference catalog.
    lr_table : pd.DataFrame
        FastPM target catalog.
    train_subbox : TrainSubboxConfig
        Axis-aligned train region for ``matching_ml``.

    Returns
    -------
    bool
        True when both catalogs have at least one halo in the sub-box.
    """
    hr_mask = mask_subbox(hr_table, train_subbox)
    lr_mask = mask_subbox(lr_table, train_subbox)
    return bool(hr_mask.any() and lr_mask.any())


def resolve_catalog_train_subbox(
    base_config: HaloscopeEnrichmentConfig,
    entry: BenchmarkCatalogEntry,
) -> TrainSubboxConfig:
    """
    Resolve the train sub-box for one benchmark catalog.

    Parameters
    ----------
    base_config : HaloscopeEnrichmentConfig
        Run configuration with default ``matching_ml`` sub-box.
    entry : BenchmarkCatalogEntry
        Catalog entry with optional sub-box override.

    Returns
    -------
    TrainSubboxConfig
        Train region for label construction.
    """
    if entry.train_subbox is not None:
        return entry.train_subbox
    return base_config.mass_calibration.matching_ml.train_subbox


def resolve_catalog_methods(
    entry: BenchmarkCatalogEntry,
    default_methods: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Return calibration methods to run for one catalog.

    Parameters
    ----------
    entry : BenchmarkCatalogEntry
        Catalog entry with optional method override.
    default_methods : tuple[str, ...]
        Global methods from the benchmark JSON root.

    Returns
    -------
    tuple[str, ...]
        Method keys for this catalog.
    """
    if entry.methods is not None:
        return entry.methods
    return default_methods


def skip_matching_ml_reason(
    hr_table: pd.DataFrame,
    lr_table: pd.DataFrame,
    base_config: HaloscopeEnrichmentConfig,
    entry: BenchmarkCatalogEntry,
) -> Optional[str]:
    """
    Explain why ``matching_ml`` should be skipped for one catalog.

    Parameters
    ----------
    hr_table : pd.DataFrame
        UNIT reference catalog.
    lr_table : pd.DataFrame
        FastPM target catalog.
    base_config : HaloscopeEnrichmentConfig
        Run configuration.
    entry : BenchmarkCatalogEntry
        Benchmark catalog entry.

    Returns
    -------
    Optional[str]
        Skip reason, or ``None`` when ML calibration can run.
    """
    train_subbox = resolve_catalog_train_subbox(base_config, entry)
    if not train_subbox_has_hr_and_lr_halos(hr_table, lr_table, train_subbox):
        return (
            "matching_ml skipped: HR and LR do not both contain halos in the train "
            f"sub-box (catalog={entry.key})."
        )
    return None


def parse_benchmark_methods(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """
    Parse optional ``benchmark_methods`` from benchmark JSON.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Root JSON object.

    Returns
    -------
    tuple[str, ...]
        Mass calibration method keys to run.
    """
    methods = payload.get("benchmark_methods")
    if methods is None:
        return DEFAULT_BENCHMARK_METHODS
    if not isinstance(methods, list) or not methods:
        raise ValueError("benchmark_methods must be a non-empty JSON array when set.")
    return tuple(str(method) for method in methods)


def parse_benchmark_hmf_config(payload: Mapping[str, Any]) -> BenchmarkHmfConfig:
    """
    Parse optional ``benchmark_hmf`` mass limits for plots and HMF metrics.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Root JSON object.

    Returns
    -------
    BenchmarkHmfConfig
        HMF mass range; defaults to Ramakrishnan et al. (2025) Table D.1 limits.
    """
    block = payload.get("benchmark_hmf")
    if block is None:
        return BenchmarkHmfConfig()
    if not isinstance(block, dict):
        raise ValueError("benchmark_hmf must be a JSON object when provided.")
    min_mass = block.get("min_mass_msun_h", DEFAULT_BENCHMARK_HMF_MIN_MASS_MSUN_H)
    max_mass = block.get("max_mass_msun_h", DEFAULT_BENCHMARK_HMF_MAX_MASS_MSUN_H)
    return BenchmarkHmfConfig(
        min_mass_msun_h=float(min_mass) if min_mass is not None else None,
        max_mass_msun_h=float(max_mass) if max_mass is not None else None,
    )


def filter_masses_to_benchmark_range(
    masses: np.ndarray,
    hmf_config: BenchmarkHmfConfig,
) -> np.ndarray:
    """
    Keep masses inside the configured benchmark HMF range.

    Parameters
    ----------
    masses : np.ndarray
        Halo masses in Msun/h.
    hmf_config : BenchmarkHmfConfig
        Inclusive mass limits.

    Returns
    -------
    np.ndarray
        Filtered mass sample.
    """
    mask = np.isfinite(masses) & (masses > 0.0)
    if hmf_config.min_mass_msun_h is not None:
        mask &= masses >= hmf_config.min_mass_msun_h
    if hmf_config.max_mass_msun_h is not None:
        mask &= masses <= hmf_config.max_mass_msun_h
    return masses[mask]


def input_features_for_matching_ml(
    ml_features: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Map Forero ``ml_features`` names to Haloscope INPUT attachers.

    Parameters
    ----------
    ml_features : tuple[str, ...]
        Names from ``mass_calibration.matching_ml.ml_features``.

    Returns
    -------
    tuple[str, ...]
        INPUT feature names to attach before ML calibration.
    """
    attach_names: list[str] = []
    if "delta" in ml_features:
        attach_names.append("overdensity")
    return tuple(attach_names)


def attach_matching_ml_input_features(
    hr_table: pd.DataFrame,
    lr_table: pd.DataFrame,
    base_config: HaloscopeEnrichmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach Haloscope INPUT columns required by ``matching_ml`` features.

    Parameters
    ----------
    hr_table : pd.DataFrame
        UNIT reference table.
    lr_table : pd.DataFrame
        FastPM target table.
    base_config : HaloscopeEnrichmentConfig
        Run configuration with tidal paths and box size.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tables with extra columns for ML feature construction.
    """
    ml_features = base_config.mass_calibration.matching_ml.ml_features
    input_features = input_features_for_matching_ml(ml_features)
    if not input_features:
        return hr_table, lr_table
    box_size = base_config.box_size_mpc_h
    if box_size is None:
        raise ValueError("box_size_mpc_h is required to attach ML input features.")
    context = build_preprocessing_context(
        repo_root=Path(base_config.repo_root),
        mass_calibration=base_config.mass_calibration,
        sim_boxsize_mpc_h=box_size,
        fastpm_boxsize_mpc_h=box_size,
        env_radius_mpc_h=ENV_RADIUS_MPC_H,
        unit_descriptors_dir=base_config.unit_descriptors_dir,
        fastpm_descriptors_dir=base_config.fastpm_descriptors_dir,
        tidal_n_grid=base_config.tidal_n_grid,
        max_descriptor_batch_files=base_config.max_descriptor_batch_files,
        mass_filter=base_config.mass_filter,
    )
    attachers = resolve_attachers(input_features)
    enriched_hr = _apply_attachers(hr_table, attachers, context.sim)
    enriched_lr = _apply_attachers(lr_table, attachers, context.fastpm)
    return enriched_hr, enriched_lr


def load_mass_calibration_benchmark_config(
    config_path: Path,
) -> tuple[
    HaloscopeEnrichmentConfig,
    tuple[BenchmarkCatalogEntry, ...],
    tuple[str, ...],
    BenchmarkHmfConfig,
]:
    """
    Load Haloscope settings and benchmark catalog list from one JSON file.

    Parameters
    ----------
    config_path : Path
        Path to the benchmark configuration file.

    Returns
    -------
    tuple
        ``(base_config, catalogs, methods, hmf_config)`` parsed from JSON.
    """
    base_config = load_haloscope_enrichment_config(config_path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark config must be a JSON object: {config_path}")
    catalogs = parse_benchmark_catalogs(payload, base_config.repo_root)
    methods = parse_benchmark_methods(payload)
    hmf_config = parse_benchmark_hmf_config(payload)
    return base_config, catalogs, methods, hmf_config


def load_lr_benchmark_catalog(
    entry: BenchmarkCatalogEntry,
    max_halos: Optional[int],
) -> pd.DataFrame:
    """
    Load one LR catalog for benchmark runs.

    Parameters
    ----------
    entry : BenchmarkCatalogEntry
        Catalog metadata and path.
    max_halos : Optional[int]
        Optional row limit passed to the underlying reader.

    Returns
    -------
    pd.DataFrame
        LR table with positions and ``M200b``.

    Raises
    ------
    FileNotFoundError
        If ``entry.path`` is missing.
    ValueError
        If ``entry.loader`` is unsupported.
    """
    if not entry.path.is_file() and not entry.path.is_dir():
        raise FileNotFoundError(f"LR catalog not found for {entry.key}: {entry.path}")
    if entry.loader == LOADER_ROCKSTAR_LIST:
        return load_fastpm_rockstar_for_mass_calibration(entry.path, max_halos=max_halos)
    if entry.loader == LOADER_FASTPM_FOF:
        return load_fastpm_fof_target_catalog(entry.path, max_halos=max_halos)
    raise ValueError(f"Unsupported LR loader: {entry.loader}")


def apply_benchmark_mass_filter(
    hr_table: pd.DataFrame,
    lr_table: pd.DataFrame,
    config: HaloscopeEnrichmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply optional mass bounds from the run config to HR and LR tables.

    Parameters
    ----------
    hr_table : pd.DataFrame
        UNIT reference table.
    lr_table : pd.DataFrame
        FastPM target table.
    config : HaloscopeEnrichmentConfig
        Run configuration with ``mass_filter``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Filtered HR and LR tables.
    """
    mass_filter = config.mass_filter
    if not mass_filter.enabled:
        return hr_table, lr_table
    mass_column = mass_filter.mass_column
    hr_filtered = filter_mass_range(
        hr_table,
        min_mass_msun_h=mass_filter.min_mass_msun_h,
        max_mass_msun_h=mass_filter.max_mass_msun_h,
        mass_column=mass_column,
    )
    lr_filtered = filter_mass_range(
        lr_table,
        min_mass_msun_h=mass_filter.min_mass_msun_h,
        max_mass_msun_h=mass_filter.max_mass_msun_h,
        mass_column=mass_column,
    )
    return hr_filtered, lr_filtered


def run_mass_calibration_method(
    base_config: HaloscopeEnrichmentConfig,
    method: str,
    hr_table: pd.DataFrame,
    lr_table: pd.DataFrame,
    box_size_mpc_h: float,
) -> tuple[pd.DataFrame, str]:
    """
    Apply one mass-calibration method to loaded HR/LR tables.

    Parameters
    ----------
    base_config : HaloscopeEnrichmentConfig
        Run configuration (method is overridden).
    method : str
        Mass calibration method key.
    hr_table : pd.DataFrame
        High-resolution reference catalog.
    lr_table : pd.DataFrame
        Low-resolution target catalog copy.
    box_size_mpc_h : float
        Periodic box side in Mpc/h.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Calibrated LR table and mass column name for bin assignment.
    """
    method_config = replace(
        base_config,
        mass_calibration=replace(base_config.mass_calibration, method=method),
    )
    lr_copy = lr_table.copy()
    return apply_mass_calibration(
        lr_copy,
        hr_table,
        method_config.mass_calibration,
        box_size_mpc_h=box_size_mpc_h,
    )


def positive_finite_masses(series: pd.Series) -> np.ndarray:
    """
    Extract finite positive masses from a catalog column.

    Parameters
    ----------
    series : pd.Series
        Halo mass column.

    Returns
    -------
    np.ndarray
        Accepted masses in Msun/h.
    """
    values = series.to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def log_mass_bin_edges_from_masses(
    mass_samples: Sequence[np.ndarray],
    log10_min: float = HMF_LOG10_BIN_MIN,
    log10_max: float = HMF_LOG10_BIN_MAX,
    bin_count: int = HMF_BIN_COUNT,
) -> np.ndarray:
    """
    Build shared linear mass bin edges spanning several mass samples.

    Parameters
    ----------
    mass_samples : Sequence[np.ndarray]
        One or more mass arrays in Msun/h.
    log10_min : float, optional
        Lower log10 edge when samples are empty.
    log10_max : float, optional
        Upper log10 edge when samples are empty.
    bin_count : int, optional
        Number of log-spaced bins.

    Returns
    -------
    np.ndarray
        Linear bin edges in Msun/h.
    """
    positive_samples = [sample[sample > 0.0] for sample in mass_samples if sample.size]
    if not positive_samples:
        return np.logspace(log10_min, log10_max, bin_count + 1)
    log_min = max(log10_min, float(np.log10(min(sample.min() for sample in positive_samples))))
    log_max = min(log10_max, float(np.log10(max(sample.max() for sample in positive_samples))))
    if log_min >= log_max:
        log_max = log_min + 0.5
    return np.logspace(log_min, log_max, bin_count + 1)


def binned_log_hmf(
    masses: np.ndarray,
    bin_edges: np.ndarray,
    volume_mpc_h3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log10 bin centers and log10 number density for one mass sample.

    Parameters
    ----------
    masses : np.ndarray
        Halo masses in Msun/h.
    bin_edges : np.ndarray
        Linear mass bin edges in Msun/h.
    volume_mpc_h3 : float
        Sample volume in (Mpc/h)^3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``log10`` bin centers and ``log10(n/V)`` for all bins (including zeros as -inf).
    """
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    counts, _ = np.histogram(masses, bins=bin_edges)
    density = counts.astype(np.float64) / volume_mpc_h3
    with np.errstate(divide="ignore"):
        log_density = np.log10(density)
    return np.log10(bin_centers), log_density


def rmse_log_density_vs_reference(
    reference_log_density: np.ndarray,
    target_log_density: np.ndarray,
) -> float:
    """
    RMSE of log10(n/V) over bins where both curves are finite.

    Parameters
    ----------
    reference_log_density : np.ndarray
        Reference ``log10(n/V)`` per bin.
    target_log_density : np.ndarray
        Target ``log10(n/V)`` per bin.

    Returns
    -------
    float
        Root mean square error in dex; ``nan`` when no overlapping bins exist.
    """
    mask = np.isfinite(reference_log_density) & np.isfinite(target_log_density)
    if not np.any(mask):
        return float("nan")
    delta = target_log_density[mask] - reference_log_density[mask]
    return float(np.sqrt(np.mean(delta * delta)))


def summarize_lr_calibration(
    lr_table: pd.DataFrame,
    mass_column: str,
    method: str,
    catalog_key: str,
) -> dict[str, Any]:
    """
    Build summary statistics for one calibrated LR table.

    Parameters
    ----------
    lr_table : pd.DataFrame
        LR table after calibration.
    mass_column : str
        Mass column used for Haloscope bins.
    method : str
        Mass calibration method key.
    catalog_key : str
        Benchmark catalog identifier.

    Returns
    -------
    dict
        JSON-serializable summary record.
    """
    calibrated = lr_table[mass_column].to_numpy(dtype=float)
    raw = lr_table[LR_RAW_MASS_COLUMN].to_numpy(dtype=float)
    finite_mask = np.isfinite(calibrated)
    n_lr = len(lr_table)
    n_finite = int(finite_mask.sum())
    summary: dict[str, Any] = {
        "catalog_key": catalog_key,
        "method": method,
        "n_lr": n_lr,
        "n_finite_calibrated_mass": n_finite,
        "fraction_finite_calibrated_mass": float(n_finite / n_lr) if n_lr else 0.0,
        "mass_column": mass_column,
    }
    if n_finite > 0:
        log_cal = np.log10(calibrated[finite_mask])
        log_raw = np.log10(raw[finite_mask])
        summary["log10_mass_cal_mean"] = float(log_cal.mean())
        summary["log10_mass_cal_std"] = float(log_cal.std())
        summary["mean_log10_mass_offset_cal_minus_raw"] = float((log_cal - log_raw).mean())
    if "mass_calib_matched" in lr_table.columns:
        summary["n_matched_flag_true"] = int(lr_table["mass_calib_matched"].sum())
    if "mass_calib_p_keep" in lr_table.columns:
        p_keep = lr_table["mass_calib_p_keep"].to_numpy(dtype=float)
        summary["p_keep_mean"] = float(np.nanmean(p_keep))
        summary["p_keep_median"] = float(np.nanmedian(p_keep))
    return summary


def save_hmf_comparison_plot(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str,
    hmf_config: Optional[BenchmarkHmfConfig] = None,
) -> None:
    """
    Save a log-log HMF comparison figure.

    Parameters
    ----------
    curves : Mapping[str, tuple[np.ndarray, np.ndarray]]
        Maps legend label to ``(log10 M, log10 n/V)`` finite points per series.
    output_path : Path
        Destination PNG path.
    title : str
        Figure title.
    hmf_config : Optional[BenchmarkHmfConfig], optional
        When set, axis limits follow the configured mass range.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    for label, (log_mass, log_density) in curves.items():
        finite = np.isfinite(log_mass) & np.isfinite(log_density)
        axis.plot(log_mass[finite], log_density[finite], label=label, linewidth=1.5)
    axis.set_xlabel(r"$\log_{10}(M / h^{-1} M_\odot)$")
    axis.set_ylabel(r"$\log_{10}(n / h^3 \mathrm{Mpc}^{-3})$")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    axis.xaxis.set_major_locator(MultipleLocator(HMF_X_MAJOR_TICK))
    axis.xaxis.set_minor_locator(MultipleLocator(HMF_X_MINOR_TICK))
    axis.yaxis.set_major_locator(MultipleLocator(HMF_Y_MAJOR_TICK))
    axis.yaxis.set_minor_locator(MultipleLocator(HMF_Y_MINOR_TICK))
    axis.grid(True, which="major", alpha=0.3)
    if hmf_config is not None:
        x_min = (
            float(np.log10(hmf_config.min_mass_msun_h))
            if hmf_config.min_mass_msun_h is not None
            else None
        )
        x_max = (
            float(np.log10(hmf_config.max_mass_msun_h))
            if hmf_config.max_mass_msun_h is not None
            else None
        )
        axis.set_xlim(left=x_min, right=x_max)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_hmf_comparison_curves(
    hr_table: pd.DataFrame,
    lr_table: pd.DataFrame,
    mass_column: str,
    method: str,
    box_size_mpc_h: float,
    hmf_config: Optional[BenchmarkHmfConfig] = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """
    Build HMF curves and RMSE metrics for UNIT, raw FastPM, and calibrated FastPM.

    Parameters
    ----------
    hr_table : pd.DataFrame
        UNIT halo table.
    lr_table : pd.DataFrame
        Calibrated FastPM table.
    mass_column : str
        Calibrated mass column on ``lr_table``.
    method : str
        Calibration method key (plot legend suffix).
    box_size_mpc_h : float
        Box side length in Mpc/h.
    hmf_config : Optional[BenchmarkHmfConfig], optional
        Mass range filter for HMF binning and metrics.

    Returns
    -------
    tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]
        Plot curves and RMSE values vs UNIT for raw and calibrated FastPM.
    """
    volume = box_size_mpc_h**3
    hmf_range = hmf_config if hmf_config is not None else BenchmarkHmfConfig()
    unit_masses = filter_masses_to_benchmark_range(
        positive_finite_masses(hr_table[HR_MASS_COLUMN]),
        hmf_range,
    )
    raw_masses = filter_masses_to_benchmark_range(
        positive_finite_masses(lr_table[LR_RAW_MASS_COLUMN]),
        hmf_range,
    )
    calibrated_masses = filter_masses_to_benchmark_range(
        positive_finite_masses(lr_table[mass_column]),
        hmf_range,
    )
    log_min = (
        float(np.log10(hmf_range.min_mass_msun_h))
        if hmf_range.min_mass_msun_h is not None
        else HMF_LOG10_BIN_MIN
    )
    log_max = (
        float(np.log10(hmf_range.max_mass_msun_h))
        if hmf_range.max_mass_msun_h is not None
        else HMF_LOG10_BIN_MAX
    )
    bin_edges = log_mass_bin_edges_from_masses(
        [unit_masses, raw_masses, calibrated_masses],
        log10_min=log_min,
        log10_max=log_max,
    )

    unit_log_m, unit_log_n = binned_log_hmf(unit_masses, bin_edges, volume)
    raw_log_m, raw_log_n = binned_log_hmf(raw_masses, bin_edges, volume)
    cal_log_m, cal_log_n = binned_log_hmf(calibrated_masses, bin_edges, volume)

    curves = {
        UNIT_HMF_LABEL: (unit_log_m, unit_log_n),
        FASTPM_RAW_HMF_LABEL: (raw_log_m, raw_log_n),
        f"FastPM ({method})": (cal_log_m, cal_log_n),
    }
    metrics = {
        "hmf_rmse_log_nv_raw_vs_unit": rmse_log_density_vs_reference(unit_log_n, raw_log_n),
        f"hmf_rmse_log_nv_{method}_vs_unit": rmse_log_density_vs_reference(unit_log_n, cal_log_n),
    }
    return curves, metrics


def load_benchmark_hr_lr_tables(
    base_config: HaloscopeEnrichmentConfig,
    entry: BenchmarkCatalogEntry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and filter UNIT and one FastPM catalog for benchmarking.

    Parameters
    ----------
    base_config : HaloscopeEnrichmentConfig
        Run configuration with paths and filters.
    entry : BenchmarkCatalogEntry
        FastPM catalog to load.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Filtered HR and LR tables.
    """
    hr_table = load_unit_sim_training_catalog(
        Path(base_config.sim_hlist_path),
        max_halos=base_config.max_sim_halos,
    )
    lr_table = load_lr_benchmark_catalog(entry, max_halos=base_config.max_fastpm_halos)
    return apply_benchmark_mass_filter(hr_table, lr_table, base_config)
