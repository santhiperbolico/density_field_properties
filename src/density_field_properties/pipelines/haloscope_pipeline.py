"""Multi-stage Haloscope pipeline orchestration."""

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    HaloscopeEnrichmentConfig,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    HaloscopeEnrichmentRun,
    run_haloscope_enrichment_from_tables,
)
from density_field_properties.pipelines.preprocessing_run import (
    build_haloscope_feature_tables,
    save_preprocessed_tables,
)
from density_field_properties.pipelines.tidal_anisotropy_run import (
    run_tidal_anisotropy_pipeline_for_target,
)


@dataclass
class HaloscopePipelineState:
    """
    Outputs accumulated while running pipeline stages.

    Parameters
    ----------
    preprocess_paths : Optional[tuple[Path, Path]]
        HR/LR Parquet paths when preprocessing wrote files.
    enrichment_result : Optional[Union[Path, HaloscopeEnrichmentRun]]
        Enriched catalog path or run object from the Haloscope stage.
    """

    preprocess_paths: Optional[tuple[Path, Path]] = None
    enrichment_result: Optional[Union[Path, HaloscopeEnrichmentRun]] = None


class PipelineElement(ABC):
    """
    A single Haloscope pipeline stage.
    """

    def __init__(self, config: HaloscopeEnrichmentConfig, state: HaloscopePipelineState):
        self.config = config
        self.state = state

    @abstractmethod
    def run(self) -> None:
        """
        Run the pipeline element.
        """
        raise NotImplementedError("Subclasses must implement this method.")


def _mass_column_fastpm(config: HaloscopeEnrichmentConfig) -> str:
    """
    LR mass column used for Haloscope binning after preprocessing.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.

    Returns
    -------
    str
        Column name from mass correction settings.
    """
    return config.pipeline.preprocessing.mass_correction.correction_column


def _resolve_feature_tables(
    config: HaloscopeEnrichmentConfig,
    state: HaloscopePipelineState,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load or build HR/LR tables for Haloscope and validation stages.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.
    state : HaloscopePipelineState
        Pipeline state with optional preprocess Parquet paths.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, str]
        HR table, LR table, and FastPM mass column name.
    """
    mass_col = _mass_column_fastpm(config)
    if state.preprocess_paths is not None:
        hr_path, lr_path = state.preprocess_paths
        return pd.read_parquet(hr_path), pd.read_parquet(lr_path), mass_col

    prep_enabled = config.pipeline.preprocessing.enabled
    haloscope_enabled = config.pipeline.haloscope.enabled
    validation_enabled = config.pipeline.validation.enabled
    if not prep_enabled and (haloscope_enabled or validation_enabled):
        out_dir = Path(config.output_dir)
        prep = config.pipeline.preprocessing
        hr_path = out_dir / prep.hr_parquet_name
        lr_path = out_dir / prep.lr_parquet_name
        if hr_path.is_file() and lr_path.is_file():
            state.preprocess_paths = (hr_path, lr_path)
            return pd.read_parquet(hr_path), pd.read_parquet(lr_path), mass_col
        raise FileNotFoundError(
            "Preprocessing is disabled but HR/LR Parquet files are missing under "
            f"{out_dir}: {prep.hr_parquet_name}, {prep.lr_parquet_name}"
        )

    return build_haloscope_feature_tables(config)


def _resolve_enriched_lr_table(
    config: HaloscopeEnrichmentConfig,
    state: HaloscopePipelineState,
    halos_fastpm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the enriched LR table from pipeline state, disk, or fall back to LR preprocess.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration.
    state : HaloscopePipelineState
        Pipeline outputs accumulated in earlier stages.
    halos_fastpm : pd.DataFrame
        Preprocessed LR table when no enriched catalog is available.

    Returns
    -------
    pd.DataFrame
        LR table used for assembly-bias validation.
    """
    enrichment = state.enrichment_result
    if enrichment is not None:
        path = (
            enrichment.output_path
            if isinstance(enrichment, HaloscopeEnrichmentRun)
            else enrichment
        )
        return pd.read_parquet(path)

    enriched_path = Path(config.output_dir) / config.enriched_parquet_name
    if enriched_path.is_file():
        return pd.read_parquet(enriched_path)

    return halos_fastpm


def _prime_cached_outputs(
    config: HaloscopeEnrichmentConfig, state: HaloscopePipelineState
) -> None:
    """
    Point pipeline state at Parquet artifacts already present in ``output_dir``.

    Used when earlier stages are disabled but Haloscope or validation still run.
    """
    out_dir = Path(config.output_dir)
    prep = config.pipeline.preprocessing
    if not prep.enabled and state.preprocess_paths is None:
        hr_path = out_dir / prep.hr_parquet_name
        lr_path = out_dir / prep.lr_parquet_name
        if hr_path.is_file() and lr_path.is_file():
            state.preprocess_paths = (hr_path, lr_path)

    if not config.pipeline.haloscope.enabled and state.enrichment_result is None:
        enriched_path = out_dir / config.enriched_parquet_name
        if enriched_path.is_file():
            state.enrichment_result = enriched_path


class EnvironmentFeaturesElement(PipelineElement):
    """
    Environment / tidal-anisotropy feature generation stage.
    """

    def run(self) -> None:
        """
        Run the environment features element when enabled.
        """
        if not self.config.pipeline.environment_features.enabled:
            return

        stage = self.config.pipeline.environment_features
        box_size = self.config.pipeline.preprocessing.sample.box_size_mpc_h

        for target_config in stage.targets:
            target_name = target_config.name
            if target_config.dm_particles_file is None:
                raise ValueError(
                    f"dm_particles_file is required for tidal target '{target_name}'."
                )
            if target_config.work_dir is None:
                raise ValueError(f"work_dir is required for tidal target '{target_name}'.")

            if target_name == PIPELINE_TARGET_UNIT:
                halo_catalog_path = Path(self.config.general.sim_hlist)
            elif target_name == PIPELINE_TARGET_FASTPM:
                halo_catalog_path = Path(self.config.general.fastpm_list)
            else:
                raise ValueError(f"Unsupported tidal target name: {target_name}")

            max_halos = target_config.max_halos
            if max_halos is None:
                sample = self.config.pipeline.preprocessing.sample
                max_halos = (
                    sample.max_sim_halos
                    if target_name == PIPELINE_TARGET_UNIT
                    else sample.max_fastpm_halos
                )

            run_tidal_anisotropy_pipeline_for_target(
                target_name=target_name,
                dm_particles_file=Path(target_config.dm_particles_file),
                halo_catalog_path=halo_catalog_path,
                work_dir=Path(target_config.work_dir),
                box_size=float(box_size),
                n_grid=stage.n_grid,
                mass_particle=target_config.mass_particle,
                catalog_layout=target_config.catalog_layout,
                cic_batch_size=stage.cic_batch_size,
                descriptor_batch_size=stage.descriptor_batch_size,
                max_halos=max_halos,
                read_tensor_from_disk=target_config.read_tensor_from_disk,
            )
            gc.collect()


class PreprocessingElement(PipelineElement):
    """
    HR/LR catalog loading, feature attachment, and mass correction stage.
    """

    def run(self) -> None:
        """
        Build feature tables and write preprocessed Parquet files when enabled.
        """
        if not self.config.pipeline.preprocessing.enabled:
            return
        halos_sim, halos_fastpm, _mass_col = build_haloscope_feature_tables(self.config)
        self.state.preprocess_paths = save_preprocessed_tables(
            self.config,
            halos_sim,
            halos_fastpm,
        )
        gc.collect()


class HaloscopeElement(PipelineElement):
    """
    Haloscope enrichment and optional hold-out corner validation plots.
    """

    def run(self) -> None:
        """
        Fit Haloscope on HR and write enriched LR Parquet when enabled.
        """
        if not self.config.pipeline.haloscope.enabled:
            return

        halos_sim, halos_fastpm, mass_col_fastpm = _resolve_feature_tables(self.config, self.state)
        skip_assembly = self.config.pipeline.validation.enabled
        self.state.enrichment_result = run_haloscope_enrichment_from_tables(
            self.config,
            halos_sim,
            halos_fastpm,
            mass_col_fastpm,
            run_enrichment=True,
            write_assembly_bias=not skip_assembly,
        )
        gc.collect()


class ValidationElement(PipelineElement):
    """
    Assembly-bias validation panel after Haloscope enrichment.
    """

    def run(self) -> None:
        """
        Write assembly-bias diagnostics when ``validation.enabled`` is set.
        """
        if not self.config.pipeline.validation.enabled:
            return

        halos_sim, halos_fastpm, mass_col_fastpm = _resolve_feature_tables(self.config, self.state)
        enriched_lr = _resolve_enriched_lr_table(self.config, self.state, halos_fastpm)

        self.state.enrichment_result = run_haloscope_enrichment_from_tables(
            self.config,
            halos_sim,
            enriched_lr,
            mass_col_fastpm,
            run_enrichment=False,
        )
        gc.collect()


class HaloscopePipeline:
    """
    Orchestrator for environment features, preprocessing, Haloscope, and validation.
    """

    def __init__(self, config: HaloscopeEnrichmentConfig):
        self.config = config
        self.state = HaloscopePipelineState()

    def run(self) -> HaloscopePipelineState:
        """
        Run configured pipeline stages in order.

        Returns
        -------
        HaloscopePipelineState
            Paths and enrichment outputs from executed stages.
        """
        _prime_cached_outputs(self.config, self.state)
        EnvironmentFeaturesElement(self.config, self.state).run()
        PreprocessingElement(self.config, self.state).run()
        HaloscopeElement(self.config, self.state).run()
        ValidationElement(self.config, self.state).run()
        return self.state


def run_haloscope_pipeline(config: HaloscopeEnrichmentConfig) -> HaloscopePipelineState:
    """
    Execute the configured Haloscope pipeline.

    Parameters
    ----------
    config : HaloscopeEnrichmentConfig
        Run configuration loaded from JSON or built programmatically.

    Returns
    -------
    HaloscopePipelineState
        Accumulated stage outputs.
    """
    return HaloscopePipeline(config).run()
