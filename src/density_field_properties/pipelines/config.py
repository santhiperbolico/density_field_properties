"""Run configuration for the Haloscope enrichment pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from density_field_properties.haloscope.sim_to_fastpm.config import (
    ENRICHED_PARQUET_NAME,
    ENRICHED_TIDAL_PARQUET_NAME,
    INPUT_FEATURES,
    OUTPUT_DIR,
    OUTPUT_DIR_TIDAL,
    OUTPUT_DIR_TIDAL_SMOKE,
    PRODUCTION_MIN_BIN_SIZE,
    SMOKE_MIN_BIN_SIZE,
    TIDAL_DENSITY_N_GRID,
    TIDAL_INPUT_FEATURES,
    default_fastpm_list_path,
    default_sim_hlist_path,
)

DEFAULT_ASSEMBLY_BIAS_N_GRID = 128
ASSEMBLY_BIAS_TIDAL_PDF_NAME = "assembly_bias_tidal_input.pdf"

DEFAULT_ENV_SMOKE_CONFIG = Path("config/haloscope_run_env_smoke.json")
DEFAULT_ENV_PRODUCTION_CONFIG = Path("config/haloscope_run_env_production.json")
DEFAULT_TIDAL_SMOKE_CONFIG = Path("config/haloscope_run_tidal_smoke.json")
DEFAULT_TIDAL_PRODUCTION_CONFIG = Path("config/haloscope_run_tidal_production.json")


@dataclass
class HaloscopeEnrichmentConfig:
    """
    Configuration for a Haloscope SIM-to-FastPM enrichment run.

    Parameters
    ----------
    sim_hlist_path : Path
        Path to the SIM (UNIT) consistent-trees catalog.
    fastpm_list_path : Path
        Path to the FastPM Rockstar ``out_*.list`` catalog.
    repo_root : Path, optional
        Repository root for resolving relative tidal descriptor directories.
    output_dir : Path, optional
        Directory for Parquet and validation outputs.
    max_sim_halos : Optional[int], optional
        Cap rows read from the SIM hlist; ``None`` reads the full catalog.
    max_fastpm_halos : Optional[int], optional
        Cap halos read from the FastPM catalog.
    max_descriptor_batch_files : Optional[int], optional
        Smoke cap on tidal descriptor batch files per simulation.
    min_bin_size : int, optional
        Minimum halos per mass bin for fit and validation.
    run_holdout_validation : bool, optional
        Whether to run SIM train/test validation before enrichment.
    input_features : tuple[str, ...], optional
        Haloscope INPUT columns to attach.
    unit_descriptors_dir : Optional[Path], optional
        UNIT tidal descriptor directory.
    fastpm_descriptors_dir : Optional[Path], optional
        FastPM tidal descriptor directory.
    tidal_n_grid : int, optional
        Grid resolution used when tidal descriptors were computed.
    enriched_parquet_name : str, optional
        Output Parquet filename inside ``output_dir``.
    run_assembly_bias_plot : bool, optional
        If True, write the tidal assembly-bias PDF after enrichment.
    assembly_bias_n_grid : int, optional
        Grid resolution for the Paranjape assembly-bias diagnostic.
    collect_tables : bool, optional
        If True, return HR/LR tables in ``HaloscopeEnrichmentRun``.
    """

    sim_hlist_path: Path
    fastpm_list_path: Path
    repo_root: Path = Path.cwd()
    output_dir: Path = OUTPUT_DIR
    max_sim_halos: Optional[int] = None
    max_fastpm_halos: Optional[int] = None
    max_descriptor_batch_files: Optional[int] = None
    min_bin_size: int = PRODUCTION_MIN_BIN_SIZE
    run_holdout_validation: bool = True
    input_features: tuple[str, ...] = INPUT_FEATURES
    unit_descriptors_dir: Optional[Path] = None
    fastpm_descriptors_dir: Optional[Path] = None
    tidal_n_grid: int = TIDAL_DENSITY_N_GRID
    enriched_parquet_name: str = ENRICHED_PARQUET_NAME
    run_assembly_bias_plot: bool = False
    assembly_bias_n_grid: int = DEFAULT_ASSEMBLY_BIAS_N_GRID
    collect_tables: bool = False


def _optional_path(value: Any) -> Optional[Path]:
    """
    Convert a JSON value to an optional filesystem path.

    Parameters
    ----------
    value : Any
        Raw JSON value.

    Returns
    -------
    Optional[Path]
        Parsed path, or ``None`` when the value is null or empty.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _required_path(value: Any, default: Path) -> Path:
    """
    Resolve a required path from JSON, falling back to a default.

    Parameters
    ----------
    value : Any
        Raw JSON value.
    default : Path
        Default path when ``value`` is null or empty.

    Returns
    -------
    Path
        Resolved path.
    """
    resolved = _optional_path(value)
    return default if resolved is None else resolved


def _optional_int(value: Any) -> Optional[int]:
    """
    Convert a JSON value to an optional integer cap.

    Parameters
    ----------
    value : Any
        Raw JSON value.

    Returns
    -------
    Optional[int]
        Parsed integer, or ``None`` when unlimited.
    """
    if value is None:
        return None
    return int(value)


def _feature_tuple(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    """
    Parse Haloscope INPUT feature names from JSON.

    Parameters
    ----------
    value : Any
        Raw JSON value.
    default : Sequence[str]
        Default feature list when ``value`` is null.

    Returns
    -------
    tuple[str, ...]
        Feature names in run order.
    """
    if value is None:
        return tuple(default)
    return tuple(str(item) for item in value)


def load_haloscope_enrichment_config(config_path: Path) -> HaloscopeEnrichmentConfig:
    """
    Load a Haloscope enrichment run configuration from JSON.

    Parameters
    ----------
    config_path : Path
        Path to the JSON configuration file.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Parsed run configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the JSON root is not an object.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Haloscope run config not found: {path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Haloscope run config must be a JSON object: {path}")

    paths = payload.get("paths", {})
    sample = payload.get("sample", {})
    haloscope = payload.get("haloscope", {})
    tidal = payload.get("tidal", {})
    validation = payload.get("validation", {})
    assembly_bias = validation.get("assembly_bias", {})

    return HaloscopeEnrichmentConfig(
        sim_hlist_path=_required_path(
            paths.get("sim_hlist"),
            default_sim_hlist_path(),
        ),
        fastpm_list_path=_required_path(
            paths.get("fastpm_list"),
            default_fastpm_list_path(),
        ),
        repo_root=_required_path(paths.get("repo_root"), Path.cwd()),
        output_dir=_required_path(paths.get("output_dir"), OUTPUT_DIR),
        max_sim_halos=_optional_int(sample.get("max_sim_halos")),
        max_fastpm_halos=_optional_int(sample.get("max_fastpm_halos")),
        max_descriptor_batch_files=_optional_int(sample.get("max_descriptor_batch_files")),
        min_bin_size=int(haloscope.get("min_bin_size", PRODUCTION_MIN_BIN_SIZE)),
        run_holdout_validation=bool(haloscope.get("run_holdout_validation", True)),
        input_features=_feature_tuple(haloscope.get("input_features"), INPUT_FEATURES),
        unit_descriptors_dir=_optional_path(tidal.get("unit_descriptors_dir")),
        fastpm_descriptors_dir=_optional_path(tidal.get("fastpm_descriptors_dir")),
        tidal_n_grid=int(tidal.get("n_grid", TIDAL_DENSITY_N_GRID)),
        enriched_parquet_name=str(haloscope.get("enriched_parquet_name", ENRICHED_PARQUET_NAME)),
        run_assembly_bias_plot=bool(assembly_bias.get("enabled", False)),
        assembly_bias_n_grid=int(assembly_bias.get("n_grid", DEFAULT_ASSEMBLY_BIAS_N_GRID)),
        collect_tables=bool(payload.get("collect_tables", False)),
    )


def tidal_smoke_config(repo_root: Optional[Path] = None) -> HaloscopeEnrichmentConfig:
    """
    Build the default tidal smoke configuration without reading JSON.

    Parameters
    ----------
    repo_root : Optional[Path], optional
        Repository root override.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Tidal smoke run settings.
    """
    root = Path.cwd() if repo_root is None else Path(repo_root)
    return HaloscopeEnrichmentConfig(
        sim_hlist_path=default_sim_hlist_path(),
        fastpm_list_path=default_fastpm_list_path(),
        repo_root=root,
        output_dir=OUTPUT_DIR_TIDAL_SMOKE,
        max_sim_halos=8000,
        max_fastpm_halos=8000,
        max_descriptor_batch_files=1,
        min_bin_size=SMOKE_MIN_BIN_SIZE,
        input_features=TIDAL_INPUT_FEATURES,
        enriched_parquet_name=ENRICHED_TIDAL_PARQUET_NAME,
    )


def tidal_production_config(repo_root: Optional[Path] = None) -> HaloscopeEnrichmentConfig:
    """
    Build the default tidal production configuration without reading JSON.

    Parameters
    ----------
    repo_root : Optional[Path], optional
        Repository root override.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Tidal production run settings.
    """
    root = Path.cwd() if repo_root is None else Path(repo_root)
    return HaloscopeEnrichmentConfig(
        sim_hlist_path=default_sim_hlist_path(),
        fastpm_list_path=default_fastpm_list_path(),
        repo_root=root,
        output_dir=OUTPUT_DIR_TIDAL,
        min_bin_size=PRODUCTION_MIN_BIN_SIZE,
        input_features=TIDAL_INPUT_FEATURES,
        enriched_parquet_name=ENRICHED_TIDAL_PARQUET_NAME,
    )
