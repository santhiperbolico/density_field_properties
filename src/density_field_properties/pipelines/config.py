"""Run configuration for the Haloscope enrichment pipeline."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from density_field_properties.pipelines.catalog_columns import (
    HLIST_CATALOG_LAYOUT,
    ROCKSTAR_LIST_CATALOG_LAYOUT,
)
from density_field_properties.pipelines.run_defaults import (
    DM_MASS_PARTICLE_MSUN_H,
    ENRICHED_PARQUET_NAME,
    ENRICHED_TIDAL_PARQUET_NAME,
    FASTPM_BOXSIZE_MPC_H,
    INPUT_FEATURES,
    OUTPUT_DIR,
    OUTPUT_DIR_TIDAL,
    OUTPUT_DIR_TIDAL_SMOKE,
    PRODUCTION_MIN_BIN_SIZE,
    SIM_BOXSIZE_MPC_H,
    SMOKE_MIN_BIN_SIZE,
    TIDAL_DENSITY_N_GRID,
    TIDAL_INPUT_FEATURES,
    default_fastpm_dm_particles_path,
    default_fastpm_list_path,
    default_sim_dm_particles_path,
    default_sim_hlist_path,
)
from density_field_properties.pipelines.tidal_anisotropy_run import (
    DEFAULT_DESCRIPTOR_BATCH_SIZE,
)

PIPELINE_TARGET_UNIT = "unit"
PIPELINE_TARGET_FASTPM = "fastpm"
_PIPELINE_TARGET_ALIASES = {
    "hr": PIPELINE_TARGET_UNIT,
    "lr": PIPELINE_TARGET_FASTPM,
}

DEFAULT_ASSEMBLY_BIAS_N_GRID = 128
ASSEMBLY_BIAS_TIDAL_PDF_NAME = "assembly_bias_tidal_input.pdf"
HOLDOUT_CORNER_PLOT_FILENAME_TEMPLATE = "holdout_sim_validation_bin{bin_index}.pdf"

DEFAULT_ENV_SMOKE_CONFIG = Path("config/haloscope_run_env_smoke.json")
DEFAULT_ENV_PRODUCTION_CONFIG = Path("config/haloscope_run_env_production.json")
DEFAULT_TIDAL_SMOKE_CONFIG = Path("config/haloscope_run_tidal_smoke.json")
DEFAULT_TIDAL_PRODUCTION_CONFIG = Path("config/haloscope_run_tidal_production.json")


def _default_tidal_unit_target() -> "TidalAnisotropyTargetConfig":
    """
    Build default UNIT tidal-anisotropy target settings.

    Returns
    -------
    TidalAnisotropyTargetConfig
        Default HR target configuration.
    """
    return TidalAnisotropyTargetConfig()


def _default_tidal_fastpm_target() -> "TidalAnisotropyTargetConfig":
    """
    Build default FastPM tidal-anisotropy target settings.

    Returns
    -------
    TidalAnisotropyTargetConfig
        Default LR target configuration with Rockstar list column layout.
    """
    return TidalAnisotropyTargetConfig(catalog_layout=ROCKSTAR_LIST_CATALOG_LAYOUT)


def _default_tidal_anisotropy_stage() -> "TidalAnisotropyStageConfig":
    """
    Build default tidal-anisotropy stage settings.

    Returns
    -------
    TidalAnisotropyStageConfig
        Disabled stage with both simulation targets configured.
    """
    return TidalAnisotropyStageConfig()


def _default_preprocess_stage() -> "PreprocessStageConfig":
    """
    Build default preprocess stage settings.

    Returns
    -------
    PreprocessStageConfig
        Disabled preprocess stage configuration.
    """
    return PreprocessStageConfig()


def _default_haloscope_stage() -> "HaloscopeStageConfig":
    """
    Build default Haloscope stage settings.

    Returns
    -------
    HaloscopeStageConfig
        Enabled Haloscope enrichment stage configuration.
    """
    return HaloscopeStageConfig()


def _default_pipeline_stages() -> "HaloscopePipelineStages":
    """
    Build default multi-stage pipeline toggles.

    Returns
    -------
    HaloscopePipelineStages
        Legacy-compatible stage defaults (Haloscope only).
    """
    return HaloscopePipelineStages()


@dataclass
class TidalAnisotropyTargetConfig:
    """
    Per-simulation settings for the tidal-anisotropy pipeline stage.

    Parameters
    ----------
    dm_particles_file : Optional[Path]
        DM particle source for CIC density construction.
    work_dir : Optional[Path]
        Simulation-specific working directory under the run output tree.
    catalog_layout : str
        Rockstar reader column layout (``hlist`` or ``rockstar_list``).
    read_tensor_from_disk : bool
        If True, skip tidal-tensor recomputation and reuse ``tidal_tensor/``.
    """

    dm_particles_file: Optional[Path] = None
    work_dir: Optional[Path] = None
    catalog_layout: str = HLIST_CATALOG_LAYOUT
    read_tensor_from_disk: bool = False


@dataclass
class TidalAnisotropyStageConfig:
    """
    Configuration for the tidal-anisotropy computation stage.

    Parameters
    ----------
    enabled : bool
        Whether to run CIC + tidal tensor + descriptor generation.
    targets : tuple[str, ...]
        Simulation targets to process sequentially (``unit``, ``fastpm``).
    mass_particle : float
        DM particle mass in Msun/h for CIC deposition.
    cic_batch_size : Optional[int]
        Particle batch size for CIC; ``None`` reads all particles at once.
    descriptor_batch_size : Optional[int]
        Halo batch size for descriptor generation.
    unit : TidalAnisotropyTargetConfig
        UNIT / HR target overrides.
    fastpm : TidalAnisotropyTargetConfig
        FastPM / LR target overrides.
    """

    enabled: bool = False
    targets: tuple[str, ...] = (PIPELINE_TARGET_UNIT, PIPELINE_TARGET_FASTPM)
    mass_particle: float = DM_MASS_PARTICLE_MSUN_H
    cic_batch_size: Optional[int] = None
    descriptor_batch_size: Optional[int] = DEFAULT_DESCRIPTOR_BATCH_SIZE
    unit: TidalAnisotropyTargetConfig = field(default_factory=_default_tidal_unit_target)
    fastpm: TidalAnisotropyTargetConfig = field(default_factory=_default_tidal_fastpm_target)


@dataclass
class PreprocessStageConfig:
    """
    Configuration for writing preprocessed HR/LR feature tables.

    Parameters
    ----------
    enabled : bool
        Whether to persist HR/LR tables with attached INPUT features.
    hr_parquet_name : str
        HR table filename inside ``output_dir``.
    lr_parquet_name : str
        LR table filename inside ``output_dir``.
    """

    enabled: bool = False
    hr_parquet_name: str = "hr_preprocessed.parquet"
    lr_parquet_name: str = "lr_preprocessed.parquet"


@dataclass
class HaloscopeStageConfig:
    """
    Configuration for the Haloscope train-and-predict stage.

    Parameters
    ----------
    enabled : bool
        Whether to fit models on HR data and enrich the LR catalog.
    """

    enabled: bool = True


@dataclass
class HaloscopePipelineStages:
    """
    Optional multi-stage Haloscope pipeline toggles.

    Parameters
    ----------
    tidal_anisotropy : TidalAnisotropyStageConfig
        CIC + tidal tensor + descriptor generation settings.
    preprocess : PreprocessStageConfig
        Preprocessed HR/LR table export settings.
    haloscope : HaloscopeStageConfig
        Haloscope enrichment stage settings.
    """

    tidal_anisotropy: TidalAnisotropyStageConfig = field(
        default_factory=_default_tidal_anisotropy_stage
    )
    preprocess: PreprocessStageConfig = field(default_factory=_default_preprocess_stage)
    haloscope: HaloscopeStageConfig = field(default_factory=_default_haloscope_stage)


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
    run_holdout_corner_plots : bool, optional
        If True, write per-bin SIM hold-out corner plots during enrichment.
    collect_tables : bool, optional
        If True, return HR/LR tables in ``HaloscopeEnrichmentRun``.
    box_size_mpc_h : Optional[float], optional
        Periodic box side length in Mpc/h for both simulations. When ``None``,
        ``SIM_BOXSIZE_MPC_H`` and ``FASTPM_BOXSIZE_MPC_H`` from run defaults apply.
    pipeline_stages : HaloscopePipelineStages, optional
        Optional multi-stage pipeline toggles. When left at defaults, only the
        Haloscope enrichment stage runs (legacy behaviour).
    """

    sim_hlist_path: Path
    fastpm_list_path: Path
    repo_root: Path = Path.cwd()
    box_size_mpc_h: Optional[float] = None
    output_dir: Path = OUTPUT_DIR
    pipeline_stages: HaloscopePipelineStages = field(default_factory=_default_pipeline_stages)
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
    run_holdout_corner_plots: bool = False
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


def _optional_float(value: Any) -> Optional[float]:
    """
    Convert a JSON value to an optional float.

    Parameters
    ----------
    value : Any
        Raw JSON value.

    Returns
    -------
    Optional[float]
        Parsed float, or ``None`` when unset.
    """
    if value is None:
        return None
    return float(value)


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


def resolve_sim_boxsize_mpc_h(config: HaloscopeEnrichmentConfig) -> float:
    """
    Resolve the SIM periodic box size for preprocessing.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    float
        Box side length in Mpc/h.
    """
    if config.box_size_mpc_h is not None:
        return float(config.box_size_mpc_h)
    return float(SIM_BOXSIZE_MPC_H)


def _normalize_pipeline_target(target: str) -> str:
    """
    Normalize a pipeline target name to ``unit`` or ``fastpm``.

    Parameters
    ----------
    target : str
        Target label from JSON (``unit``, ``fastpm``, ``hr``, or ``lr``).

    Returns
    -------
    str
        Canonical target name.

    Raises
    ------
    ValueError
        If the target label is not supported.
    """
    normalized = str(target).strip().lower()
    canonical = _PIPELINE_TARGET_ALIASES.get(normalized, normalized)
    if canonical not in {PIPELINE_TARGET_UNIT, PIPELINE_TARGET_FASTPM}:
        raise ValueError(
            f"Unsupported pipeline target '{target}'; expected unit, fastpm, hr, or lr."
        )
    return canonical


def _target_tuple(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    """
    Parse pipeline target names from JSON.

    Parameters
    ----------
    value : Any
        Raw JSON value.
    default : Sequence[str]
        Default targets when ``value`` is null.

    Returns
    -------
    tuple[str, ...]
        Canonical target names in run order.
    """
    if value is None:
        return tuple(_normalize_pipeline_target(item) for item in default)
    return tuple(_normalize_pipeline_target(str(item)) for item in value)


def _parse_tidal_target_config(
    payload: dict[str, Any],
    default_layout: str,
    default_dm_particles: Optional[Path],
    default_work_dir: Path,
) -> TidalAnisotropyTargetConfig:
    """
    Parse per-simulation tidal-anisotropy settings from JSON.

    Parameters
    ----------
    payload : dict[str, Any]
        Target-specific JSON object.
    default_layout : str
        Default Rockstar catalog column layout.
    default_dm_particles : Optional[Path]
        Default DM particle source when omitted.
    default_work_dir : Path
        Default working directory when omitted.

    Returns
    -------
    TidalAnisotropyTargetConfig
        Parsed target settings.
    """
    dm_particles = _optional_path(payload.get("dm_particles_file"))
    if dm_particles is None:
        dm_particles = default_dm_particles
    work_dir = _optional_path(payload.get("work_dir"))
    if work_dir is None:
        work_dir = default_work_dir
    layout = str(payload.get("catalog_layout", default_layout))
    return TidalAnisotropyTargetConfig(
        dm_particles_file=dm_particles,
        work_dir=work_dir,
        catalog_layout=layout,
        read_tensor_from_disk=bool(payload.get("read_tensor_from_disk", False)),
    )


def _parse_pipeline_stages(
    payload: dict[str, Any],
    output_dir: Path,
) -> HaloscopePipelineStages:
    """
    Parse optional multi-stage pipeline settings from JSON.

    Parameters
    ----------
    payload : dict[str, Any]
        Root JSON object.
    output_dir : Path
        Run output directory used for default work dirs.

    Returns
    -------
    HaloscopePipelineStages
        Parsed stage toggles.
    """
    pipeline = payload.get("pipeline")
    if pipeline is None:
        return HaloscopePipelineStages()

    tidal_payload = pipeline.get("tidal_anisotropy", {})
    preprocess_payload = pipeline.get("preprocess", {})
    haloscope_payload = pipeline.get("haloscope", {})

    unit_defaults = TidalAnisotropyTargetConfig(
        dm_particles_file=default_sim_dm_particles_path(),
        work_dir=output_dir / PIPELINE_TARGET_UNIT,
        catalog_layout=HLIST_CATALOG_LAYOUT,
    )
    fastpm_defaults = TidalAnisotropyTargetConfig(
        dm_particles_file=default_fastpm_dm_particles_path(),
        work_dir=output_dir / PIPELINE_TARGET_FASTPM,
        catalog_layout=ROCKSTAR_LIST_CATALOG_LAYOUT,
    )

    unit_target = _parse_tidal_target_config(
        tidal_payload.get("unit", {}),
        default_layout=HLIST_CATALOG_LAYOUT,
        default_dm_particles=unit_defaults.dm_particles_file,
        default_work_dir=unit_defaults.work_dir or output_dir / PIPELINE_TARGET_UNIT,
    )
    fastpm_target = _parse_tidal_target_config(
        tidal_payload.get("fastpm", {}),
        default_layout=ROCKSTAR_LIST_CATALOG_LAYOUT,
        default_dm_particles=fastpm_defaults.dm_particles_file,
        default_work_dir=fastpm_defaults.work_dir or output_dir / PIPELINE_TARGET_FASTPM,
    )

    return HaloscopePipelineStages(
        tidal_anisotropy=TidalAnisotropyStageConfig(
            enabled=bool(tidal_payload.get("enabled", False)),
            targets=_target_tuple(
                tidal_payload.get("targets"),
                (PIPELINE_TARGET_UNIT, PIPELINE_TARGET_FASTPM),
            ),
            mass_particle=float(tidal_payload.get("mass_particle", DM_MASS_PARTICLE_MSUN_H)),
            cic_batch_size=_optional_int(tidal_payload.get("cic_batch_size")),
            descriptor_batch_size=_optional_int(
                tidal_payload.get("descriptor_batch_size", DEFAULT_DESCRIPTOR_BATCH_SIZE)
            ),
            unit=unit_target,
            fastpm=fastpm_target,
        ),
        preprocess=PreprocessStageConfig(
            enabled=bool(preprocess_payload.get("enabled", False)),
            hr_parquet_name=str(
                preprocess_payload.get("hr_parquet_name", "hr_preprocessed.parquet")
            ),
            lr_parquet_name=str(
                preprocess_payload.get("lr_parquet_name", "lr_preprocessed.parquet")
            ),
        ),
        haloscope=HaloscopeStageConfig(
            enabled=bool(haloscope_payload.get("enabled", True)),
        ),
    )


def resolve_fastpm_boxsize_mpc_h(config: HaloscopeEnrichmentConfig) -> float:
    """
    Resolve the FastPM periodic box size for preprocessing.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    float
        Box side length in Mpc/h.
    """
    if config.box_size_mpc_h is not None:
        return float(config.box_size_mpc_h)
    return float(FASTPM_BOXSIZE_MPC_H)


def resolve_sim_dm_particles_path(config: HaloscopeEnrichmentConfig) -> Optional[Path]:
    """
    Resolve the UNIT DM particle path for assembly-bias matter overdensity.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    Optional[Path]
        DM particle file or snapshot block, or ``None`` when not configured.
    """
    unit_dm_path = config.pipeline_stages.tidal_anisotropy.unit.dm_particles_file
    if unit_dm_path is not None:
        return unit_dm_path
    return default_sim_dm_particles_path()


def resolve_fastpm_dm_particles_path(config: HaloscopeEnrichmentConfig) -> Optional[Path]:
    """
    Resolve the FastPM DM particle path for assembly-bias matter overdensity.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    Optional[Path]
        DM particle file or snapshot block, or ``None`` when not configured.
    """
    fastpm_dm_path = config.pipeline_stages.tidal_anisotropy.fastpm.dm_particles_file
    if fastpm_dm_path is not None:
        return fastpm_dm_path
    return default_fastpm_dm_particles_path()


def resolve_dm_mass_particle_msun_h(config: HaloscopeEnrichmentConfig) -> float:
    """
    Resolve the DM particle mass used for assembly-bias CIC deposition.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    float
        DM particle mass in Msun/h.
    """
    return float(config.pipeline_stages.tidal_anisotropy.mass_particle)


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
    box_size_mpc_h = _optional_float(sample.get("box_size_mpc_h", payload.get("box_size_mpc_h")))
    haloscope = payload.get("haloscope", {})
    tidal = payload.get("tidal", {})
    validation = payload.get("validation", {})
    assembly_bias = validation.get("assembly_bias", {})
    holdout_corner_plots = validation.get("holdout_corner_plots", {})
    output_dir = _required_path(paths.get("output_dir"), OUTPUT_DIR)
    pipeline_stages = _parse_pipeline_stages(payload, output_dir)

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
        output_dir=output_dir,
        box_size_mpc_h=box_size_mpc_h,
        pipeline_stages=pipeline_stages,
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
        run_holdout_corner_plots=bool(holdout_corner_plots.get("enabled", False)),
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
