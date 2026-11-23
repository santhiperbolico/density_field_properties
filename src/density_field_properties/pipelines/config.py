"""Run configuration for the Haloscope enrichment pipeline (JSON template shape)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from density_field_properties.haloscope.mass_bin_config import (
    HaloscopeMassBinConfig,
    default_haloscope_mass_bin_config,
)
from density_field_properties.pipelines.run_defaults import (
    DM_MASS_PARTICLE_MSUN_H,
    INPUT_FEATURES,
    OUTPUT_DIR,
)
from density_field_properties.pipelines.tidal_anisotropy_run import DEFAULT_DESCRIPTOR_BATCH_SIZE
from density_field_properties.preprocessing.mass_correction.config import (
    MASS_CORRECTION_METHODS,
    MassCorrectionSettings,
    _parse_matching,
    _parse_matching_ml,
)
from density_field_properties.preprocessing.mass_filter_config import HaloMassFilterConfig

PIPELINE_TARGET_UNIT = "unit"
PIPELINE_TARGET_FASTPM = "fastpm"

DEFAULT_PIPELINE_CONFIG = Path("config/pipeline_haloscope_template.json")

_ENVIRONMENT_FEATURES_KEYS = ("environment_features", "enviroment_features")

HOLDOUT_CORNER_PLOT_FILENAME_TEMPLATE = "holdout_sim_validation_bin{bin_index}.pdf"
ASSEMBLY_BIAS_TIDAL_PDF_NAME = "assembly_bias_tidal_input.pdf"
DEFAULT_ASSEMBLY_BIAS_N_GRID = 128


def _parse_path(value: Any, default: Optional[Path] = None) -> Optional[Path]:
    """
    Parse a path from JSON, returning ``default`` when the value is null or empty.

    Parameters
    ----------
    value : Any
        Raw JSON value.
    default : Optional[Path], optional
        Fallback when ``value`` is null, empty, or whitespace.

    Returns
    -------
    Optional[Path]
        Resolved path or ``default``.
    """
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return Path(text)


def _pipeline_environment_features(pipeline: dict) -> dict:
    """
    Return the environment-features subsection from a pipeline object.

    Parameters
    ----------
    pipeline : dict
        Raw ``pipeline`` object from the run JSON.

    Returns
    -------
    dict
        Environment-features settings (empty dict when absent).
    """
    for key in _ENVIRONMENT_FEATURES_KEYS:
        section = pipeline.get(key)
        if isinstance(section, dict):
            return section
    return {}


@dataclass
class GeneralConfig:
    """
    Top-level paths and run output location (``general`` in JSON).
    """

    sim_hlist: Path
    fastpm_list: Path
    repo_root: Path = Path.cwd()
    output_dir: Path = OUTPUT_DIR
    input_features: list[str] = field(default_factory=lambda: list(INPUT_FEATURES))
    unit_descriptors_dir: Optional[Path] = None
    fastpm_descriptors_dir: Optional[Path] = None


@dataclass
class TidalAnisotropyTargetConfig:
    """
    Per-simulation settings for the tidal-anisotropy pipeline stage.
    """

    name: str
    dm_particles_file: Optional[Path] = None
    work_dir: Optional[Path] = None
    catalog_layout: str = "hlist"
    read_tensor_from_disk: bool = False
    descriptors_dir: Optional[Path] = None
    mass_particle: Optional[float] = DM_MASS_PARTICLE_MSUN_H
    max_halos: Optional[int] = None


@dataclass
class EnvironmentFeaturesConfig:
    """
    Configuration for the environment / tidal-anisotropy features stage.
    """

    enabled: bool
    n_grid: int
    cic_batch_size: Optional[int]
    descriptor_batch_size: Optional[int]
    targets: list[TidalAnisotropyTargetConfig]


@dataclass
class SampleConfig:
    """
    Halo sampling limits for preprocessing.
    """

    box_size_mpc_h: float
    min_mass_msun_h: float
    get_min_mass_lr_from_hmf: bool
    max_mass_msun_h: Optional[float]
    max_sim_halos: Optional[int]
    max_fastpm_halos: Optional[int]


@dataclass
class MassCorrectionConfig:
    """
    LR mass correction settings for preprocessing.
    """

    method: str
    correction_column: str
    params_method: dict


@dataclass
class PreprocessingConfig:
    """
    Preprocess stage: sample filters, mass correction, cached Parquet names.
    """

    enabled: bool
    hr_parquet_name: str
    lr_parquet_name: str
    sample: SampleConfig
    mass_correction: MassCorrectionConfig


@dataclass
class HaloscopeConfig:
    """
    Haloscope enrichment stage settings.
    """

    enabled: bool
    input_features: list[str]
    run_holdout_validation: bool
    min_bin_size: int
    enriched_parquet_name: str


@dataclass
class ValidationConfig:
    """
    Post-enrichment validation (e.g. assembly-bias grid).
    """

    enabled: bool
    n_grid: int


@dataclass
class HaloscopePipelineFlowConfig:
    """
    Nested ``pipeline`` section of the run JSON.
    """

    environment_features: EnvironmentFeaturesConfig
    preprocessing: PreprocessingConfig
    haloscope: HaloscopeConfig
    validation: ValidationConfig


@dataclass
class HaloscopeEnrichmentConfig:
    """
    Full run configuration: ``run_name``, ``general``, and ``pipeline``.
    """

    run_name: str
    general: GeneralConfig
    pipeline: HaloscopePipelineFlowConfig

    @property
    def sim_hlist_path(self) -> Path:
        """
        HR halo catalog path (legacy alias for enrichment modules).

        Returns
        -------
        Path
            Path from ``general.sim_hlist``.
        """
        return self.general.sim_hlist

    @property
    def fastpm_list_path(self) -> Path:
        """
        LR halo catalog path (legacy alias for enrichment modules).

        Returns
        -------
        Path
            Path from ``general.fastpm_list``.
        """
        return self.general.fastpm_list

    @property
    def repo_root(self) -> Path:
        """
        Repository root directory.

        Returns
        -------
        Path
            Path from ``general.repo_root``.
        """
        return self.general.repo_root

    @property
    def output_dir(self) -> Path:
        """
        Run output directory.

        Returns
        -------
        Path
            Path from ``general.output_dir``.
        """
        return self.general.output_dir

    @property
    def input_features(self) -> tuple[str, ...]:
        """
        Haloscope INPUT feature names for this run.

        Returns
        -------
        tuple[str, ...]
            Feature names from ``general.input_features``.
        """
        return tuple(self.general.input_features)

    @property
    def unit_descriptors_dir(self) -> Optional[Path]:
        """
        UNIT tidal descriptor directory override.

        Returns
        -------
        Optional[Path]
            Path from ``general.unit_descriptors_dir``.
        """
        return self.general.unit_descriptors_dir

    @unit_descriptors_dir.setter
    def unit_descriptors_dir(self, value: Optional[Path]) -> None:
        self.general.unit_descriptors_dir = value

    @property
    def fastpm_descriptors_dir(self) -> Optional[Path]:
        """
        FastPM tidal descriptor directory override.

        Returns
        -------
        Optional[Path]
            Path from ``general.fastpm_descriptors_dir``.
        """
        return self.general.fastpm_descriptors_dir

    @fastpm_descriptors_dir.setter
    def fastpm_descriptors_dir(self, value: Optional[Path]) -> None:
        self.general.fastpm_descriptors_dir = value

    @property
    def max_sim_halos(self) -> Optional[int]:
        """
        Optional cap on loaded HR halos.

        Returns
        -------
        Optional[int]
            ``preprocessing.sample.max_sim_halos``.
        """
        return self.pipeline.preprocessing.sample.max_sim_halos

    @property
    def max_fastpm_halos(self) -> Optional[int]:
        """
        Optional cap on loaded LR halos.

        Returns
        -------
        Optional[int]
            ``preprocessing.sample.max_fastpm_halos``.
        """
        return self.pipeline.preprocessing.sample.max_fastpm_halos

    @property
    def tidal_n_grid(self) -> int:
        """
        CIC grid resolution for tidal descriptors.

        Returns
        -------
        int
            ``environment_features.n_grid``.
        """
        return self.pipeline.environment_features.n_grid

    @property
    def mass_correction(self) -> MassCorrectionSettings:
        """
        LR mass correction settings derived from preprocessing JSON.

        Returns
        -------
        MassCorrectionSettings
            Runtime settings for ``preprocessing.mass_correction`` apply step.
        """
        return mass_correction_settings_from_preprocessing(self.pipeline.preprocessing)

    @property
    def mass_filter(self) -> HaloMassFilterConfig:
        """
        Optional mass range filter from preprocessing sample settings.

        Returns
        -------
        HaloMassFilterConfig
            Filter with sample mass bounds.
        """
        sample = self.pipeline.preprocessing.sample
        return HaloMassFilterConfig(
            min_mass_msun_h=sample.min_mass_msun_h,
            max_mass_msun_h=sample.max_mass_msun_h,
        )

    @property
    def enriched_parquet_name(self) -> str:
        """
        Filename for enriched LR Parquet output.

        Returns
        -------
        str
            ``haloscope.enriched_parquet_name``.
        """
        return self.pipeline.haloscope.enriched_parquet_name

    @property
    def min_bin_size(self) -> int:
        """
        Minimum halos per mass bin for Haloscope fitting.

        Returns
        -------
        int
            ``haloscope.min_bin_size``.
        """
        return self.pipeline.haloscope.min_bin_size

    @property
    def run_holdout_validation(self) -> bool:
        """
        Whether to run hold-out validation during enrichment.

        Returns
        -------
        bool
            ``haloscope.run_holdout_validation``.
        """
        return self.pipeline.haloscope.run_holdout_validation

    @property
    def run_holdout_corner_plots(self) -> bool:
        """
        Whether to write hold-out corner plots.

        Returns
        -------
        bool
            Same as ``run_holdout_validation`` until a dedicated flag exists.
        """
        return self.pipeline.haloscope.run_holdout_validation

    @property
    def run_assembly_bias_plot(self) -> bool:
        """
        Whether to write assembly-bias validation plots.

        Returns
        -------
        bool
            ``validation.enabled``.
        """
        return self.pipeline.validation.enabled

    @property
    def assembly_bias_n_grid(self) -> int:
        """
        Grid resolution for assembly-bias panels.

        Returns
        -------
        int
            ``validation.n_grid``.
        """
        return self.pipeline.validation.n_grid

    @property
    def collect_tables(self) -> bool:
        """
        Whether enrichment should return in-memory tables.

        Returns
        -------
        bool
            Always False for the simplified pipeline config.
        """
        return False

    @property
    def mass_bin(self) -> HaloscopeMassBinConfig:
        """
        Optional custom mass-bin edges for Haloscope.

        Returns
        -------
        HaloscopeMassBinConfig
            Default multi-bin configuration.
        """
        return default_haloscope_mass_bin_config()

    @property
    def max_descriptor_batch_files(self) -> Optional[int]:
        """
        Smoke cap on tidal descriptor batch files.

        Returns
        -------
        Optional[int]
            Not configurable in the template; always ``None``.
        """
        return None


def mass_correction_settings_from_preprocessing(
    preprocessing: PreprocessingConfig,
) -> MassCorrectionSettings:
    """
    Build runtime mass-correction settings from the preprocessing JSON block.

    Parameters
    ----------
    preprocessing : PreprocessingConfig
        Preprocessing stage settings.

    Returns
    -------
    MassCorrectionSettings
        Settings consumed by ``apply_mass_correction``.

    Raises
    ------
    ValueError
        If the correction method is not supported.
    """
    mass_block = preprocessing.mass_correction
    method = str(mass_block.method)
    if method not in MASS_CORRECTION_METHODS:
        raise ValueError(
            f"Unknown mass correction method '{method}'; "
            f"expected one of {', '.join(MASS_CORRECTION_METHODS)}."
        )
    params = dict(mass_block.params_method or {})
    matching_payload = params.pop("matching", {})
    if not isinstance(matching_payload, dict):
        matching_payload = {}
    matching = _parse_matching(matching_payload)
    matching_ml = _parse_matching_ml(params)
    sample = preprocessing.sample
    return MassCorrectionSettings(
        enabled=True,
        method=method,
        corrected_column=mass_block.correction_column,
        matching=matching,
        matching_ml=matching_ml,
        auto_lr_correction_min_mass_from_hmf=sample.get_min_mass_lr_from_hmf,
    )


def _parse_tidal_target(raw: dict) -> TidalAnisotropyTargetConfig:
    """
    Parse one tidal-anisotropy target from JSON.

    Parameters
    ----------
    raw : dict
        Target object from ``targets`` array.

    Returns
    -------
    TidalAnisotropyTargetConfig
        Parsed target settings.
    """
    mass_particle = raw.get("mass_particle")
    return TidalAnisotropyTargetConfig(
        name=str(raw["name"]),
        dm_particles_file=_parse_path(raw.get("dm_particles_file")),
        work_dir=_parse_path(raw.get("work_dir")),
        catalog_layout=str(raw.get("catalog_layout", "hlist")),
        read_tensor_from_disk=bool(raw.get("read_tensor_from_disk", False)),
        descriptors_dir=_parse_path(raw.get("descriptors_dir")),
        mass_particle=(
            float(mass_particle) if mass_particle is not None else DM_MASS_PARTICLE_MSUN_H
        ),
        max_halos=raw.get("max_halos"),
    )


def _parse_environment_features(raw: dict) -> EnvironmentFeaturesConfig:
    """
    Parse the environment-features pipeline subsection.

    Parameters
    ----------
    raw : dict
        Raw subsection dict.

    Returns
    -------
    EnvironmentFeaturesConfig
        Parsed stage config.
    """
    targets_raw = raw.get("targets") or []
    targets = [_parse_tidal_target(item) for item in targets_raw]
    cic_batch_size = raw.get("cic_batch_size")
    descriptor_batch_size = raw.get("descriptor_batch_size", DEFAULT_DESCRIPTOR_BATCH_SIZE)
    return EnvironmentFeaturesConfig(
        enabled=bool(raw.get("enabled", False)),
        n_grid=int(raw.get("n_grid", 512)),
        cic_batch_size=int(cic_batch_size) if cic_batch_size is not None else None,
        descriptor_batch_size=(
            int(descriptor_batch_size)
            if descriptor_batch_size is not None
            else DEFAULT_DESCRIPTOR_BATCH_SIZE
        ),
        targets=targets,
    )


def _parse_sample(raw: dict) -> SampleConfig:
    """
    Parse the preprocessing ``sample`` object.

    Parameters
    ----------
    raw : dict
        Raw sample dict.

    Returns
    -------
    SampleConfig
        Parsed sample limits.
    """
    return SampleConfig(
        box_size_mpc_h=float(raw["box_size_mpc_h"]),
        min_mass_msun_h=float(raw["min_mass_msun_h"]),
        get_min_mass_lr_from_hmf=bool(raw.get("get_min_mass_lr_from_hmf", False)),
        max_mass_msun_h=raw.get("max_mass_msun_h"),
        max_sim_halos=raw.get("max_sim_halos"),
        max_fastpm_halos=raw.get("max_fastpm_halos"),
    )


def _parse_preprocessing(raw: dict) -> PreprocessingConfig:
    """
    Parse the preprocessing pipeline subsection.

    Parameters
    ----------
    raw : dict
        Raw subsection dict.

    Returns
    -------
    PreprocessingConfig
        Parsed stage config.
    """
    mass_raw = raw.get("mass_correction") or {}
    method = str(mass_raw.get("method", "abundance_matching"))
    if method not in MASS_CORRECTION_METHODS:
        raise ValueError(
            f"Unknown mass correction method '{method}'; "
            f"expected one of {', '.join(MASS_CORRECTION_METHODS)}."
        )
    return PreprocessingConfig(
        enabled=bool(raw.get("enabled", True)),
        hr_parquet_name=str(raw["hr_parquet_name"]),
        lr_parquet_name=str(raw["lr_parquet_name"]),
        sample=_parse_sample(raw.get("sample") or {}),
        mass_correction=MassCorrectionConfig(
            method=method,
            correction_column=str(mass_raw.get("correction_column", "M200b_cal")),
            params_method=dict(mass_raw.get("params_method") or {}),
        ),
    )


def _parse_haloscope(raw: dict) -> HaloscopeConfig:
    """
    Parse the haloscope pipeline subsection.

    Parameters
    ----------
    raw : dict
        Raw subsection dict.

    Returns
    -------
    HaloscopeConfig
        Parsed stage config.
    """
    return HaloscopeConfig(
        enabled=bool(raw.get("enabled", True)),
        input_features=list(raw.get("input_features") or []),
        run_holdout_validation=bool(raw.get("run_holdout_validation", True)),
        min_bin_size=int(raw.get("min_bin_size", 9)),
        enriched_parquet_name=str(raw.get("enriched_parquet_name", "haloscope_enriched.parquet")),
    )


def _parse_validation(raw: dict) -> ValidationConfig:
    """
    Parse the validation pipeline subsection.

    Parameters
    ----------
    raw : dict
        Raw subsection dict.

    Returns
    -------
    ValidationConfig
        Parsed validation config.
    """
    return ValidationConfig(
        enabled=bool(raw.get("enabled", False)),
        n_grid=int(raw.get("n_grid", 256)),
    )


def _parse_general(raw: dict) -> GeneralConfig:
    """
    Parse the top-level ``general`` section.

    Parameters
    ----------
    raw : dict
        Raw ``general`` dict.

    Returns
    -------
    GeneralConfig
        Parsed paths and output directory.
    """
    return GeneralConfig(
        sim_hlist=_parse_path(raw.get("sim_hlist"), Path("sim_hlist.list")),
        fastpm_list=_parse_path(raw.get("fastpm_list"), Path("fastpm.list")),
        repo_root=_parse_path(raw.get("repo_root"), Path.cwd()),
        output_dir=_parse_path(raw.get("output_dir"), OUTPUT_DIR),
        input_features=list(raw.get("input_features") or INPUT_FEATURES),
        unit_descriptors_dir=_parse_path(raw.get("unit_descriptors_dir")),
        fastpm_descriptors_dir=_parse_path(raw.get("fastpm_descriptors_dir")),
    )


def resolve_sim_dm_particles_path(config: HaloscopeEnrichmentConfig) -> Optional[Path]:
    """
    DM particle file for the UNIT target when configured.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    Optional[Path]
        UNIT ``dm_particles_file`` when present.
    """
    target = tidal_target_by_name(config, PIPELINE_TARGET_UNIT)
    if target is None:
        return None
    return target.dm_particles_file


def resolve_fastpm_dm_particles_path(config: HaloscopeEnrichmentConfig) -> Optional[Path]:
    """
    DM particle file for the FastPM target when configured.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    Optional[Path]
        FastPM ``dm_particles_file`` when present.
    """
    target = tidal_target_by_name(config, PIPELINE_TARGET_FASTPM)
    if target is None:
        return None
    return target.dm_particles_file


def resolve_dm_mass_particle_msun_h(config: HaloscopeEnrichmentConfig) -> float:
    """
    DM particle mass for environment-feature CIC deposition.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    float
        Target ``mass_particle`` or the module default.
    """
    target = tidal_target_by_name(config, PIPELINE_TARGET_UNIT)
    if target is not None and target.mass_particle is not None:
        return float(target.mass_particle)
    return float(DM_MASS_PARTICLE_MSUN_H)


def resolve_sim_boxsize_mpc_h(config: HaloscopeEnrichmentConfig) -> float:
    """
    HR simulation box side in Mpc/h from preprocessing sample settings.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    float
        ``preprocessing.sample.box_size_mpc_h``.
    """
    return float(config.pipeline.preprocessing.sample.box_size_mpc_h)


def resolve_fastpm_boxsize_mpc_h(config: HaloscopeEnrichmentConfig) -> float:
    """
    LR simulation box side in Mpc/h from preprocessing sample settings.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    float
        ``preprocessing.sample.box_size_mpc_h``.
    """
    return float(config.pipeline.preprocessing.sample.box_size_mpc_h)


def resolve_assembly_bias_hr_parquet_path(config: HaloscopeEnrichmentConfig) -> Path:
    """
    HR Parquet path under ``general.output_dir``.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    Path
        HR table Parquet path.
    """
    prep = config.pipeline.preprocessing
    return config.general.output_dir / prep.hr_parquet_name


def resolve_assembly_bias_enriched_parquet_path(config: HaloscopeEnrichmentConfig) -> Path:
    """
    Enriched LR Parquet path under ``general.output_dir``.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    Path
        Enriched LR table Parquet path.
    """
    haloscope = config.pipeline.haloscope
    return config.general.output_dir / haloscope.enriched_parquet_name


def resolve_assembly_bias_mass_column_fastpm(config: HaloscopeEnrichmentConfig) -> str:
    """
    LR mass column used after mass correction.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.

    Returns
    -------
    str
        Mass column name on the LR table.
    """
    return config.pipeline.preprocessing.mass_correction.correction_column


def tidal_target_by_name(
    config: HaloscopeEnrichmentConfig,
    name: str,
) -> Optional[TidalAnisotropyTargetConfig]:
    """
    Look up a tidal target by ``name`` (``unit`` or ``fastpm``).

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.
    name : str
        Target name.

    Returns
    -------
    Optional[TidalAnisotropyTargetConfig]
        Matching target, or ``None``.
    """
    for target in config.pipeline.environment_features.targets:
        if target.name == name:
            return target
    return None


def resolve_descriptor_dir_for_target(
    config: HaloscopeEnrichmentConfig,
    target_name: str,
) -> Optional[Path]:
    """
    Resolve tidal descriptor directory for one simulation target.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Loaded run configuration.
    target_name : str
        ``unit`` or ``fastpm``.

    Returns
    -------
    Optional[Path]
        Descriptor directory from general overrides or the target block.
    """
    if target_name == PIPELINE_TARGET_UNIT:
        if config.general.unit_descriptors_dir is not None:
            return config.general.unit_descriptors_dir
    elif target_name == PIPELINE_TARGET_FASTPM:
        if config.general.fastpm_descriptors_dir is not None:
            return config.general.fastpm_descriptors_dir
    target = tidal_target_by_name(config, target_name)
    if target is None:
        return None
    return target.descriptors_dir


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

    general = payload.get("general") or {}
    pipeline = payload.get("pipeline") or {}

    return HaloscopeEnrichmentConfig(
        run_name=str(payload.get("run_name") or path.stem),
        general=_parse_general(general),
        pipeline=HaloscopePipelineFlowConfig(
            environment_features=_parse_environment_features(
                _pipeline_environment_features(pipeline),
            ),
            preprocessing=_parse_preprocessing(pipeline.get("preprocessing") or {}),
            haloscope=_parse_haloscope(pipeline.get("haloscope") or {}),
            validation=_parse_validation(pipeline.get("validation") or {}),
        ),
    )
