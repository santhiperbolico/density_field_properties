"""Tests for Haloscope pipeline stages."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    EnvironmentFeaturesConfig,
    GeneralConfig,
    HaloscopeConfig,
    HaloscopeEnrichmentConfig,
    HaloscopePipelineFlowConfig,
    MassCorrectionConfig,
    PreprocessingConfig,
    SampleConfig,
    TidalAnisotropyTargetConfig,
    ValidationConfig,
    load_haloscope_enrichment_config,
)
from density_field_properties.pipelines.haloscope_enrichment import HaloscopeEnrichmentRun
from density_field_properties.pipelines.haloscope_pipeline import (
    EnvironmentFeaturesElement,
    HaloscopeElement,
    HaloscopePipeline,
    HaloscopePipelineState,
    PreprocessingElement,
    ValidationElement,
    run_haloscope_pipeline,
)

PIPELINE_MODULE = "density_field_properties.pipelines.haloscope_pipeline"
PATCH_TIDAL_TARGET = f"{PIPELINE_MODULE}.run_tidal_anisotropy_pipeline_for_target"
PATCH_BUILD_TABLES = f"{PIPELINE_MODULE}.build_haloscope_feature_tables"
PATCH_ENRICHMENT = f"{PIPELINE_MODULE}.run_haloscope_enrichment_from_tables"


def _minimal_pipeline_config(tmp_path: Path) -> HaloscopeEnrichmentConfig:
    """
    Build a minimal in-memory pipeline config with placeholder catalogs.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for catalog paths.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Config with optional stages toggled off by default.
    """
    sim_path = tmp_path / "sim.list"
    fastpm_path = tmp_path / "fastpm.list"
    sim_path.write_text("# header\n", encoding="utf-8")
    fastpm_path.write_text("# header\n", encoding="utf-8")
    sample = SampleConfig(
        box_size_mpc_h=200.0,
        min_mass_msun_h=1.0e11,
        get_min_mass_lr_from_hmf=False,
        max_mass_msun_h=None,
        max_sim_halos=None,
        max_fastpm_halos=None,
    )
    return HaloscopeEnrichmentConfig(
        run_name="test",
        general=GeneralConfig(
            sim_hlist=sim_path,
            fastpm_list=fastpm_path,
            repo_root=tmp_path,
            output_dir=tmp_path / "out",
            input_features=["env"],
        ),
        pipeline=HaloscopePipelineFlowConfig(
            environment_features=EnvironmentFeaturesConfig(
                enabled=False,
                n_grid=256,
                cic_batch_size=None,
                descriptor_batch_size=1000,
                targets=[],
            ),
            preprocessing=PreprocessingConfig(
                enabled=True,
                hr_parquet_name="hr.parquet",
                lr_parquet_name="lr.parquet",
                sample=sample,
                mass_correction=MassCorrectionConfig(
                    method="abundance_matching",
                    correction_column="M200b_cal",
                    params_method={},
                ),
            ),
            haloscope=HaloscopeConfig(
                enabled=False,
                input_features=[],
                run_holdout_validation=False,
                min_bin_size=5,
                enriched_parquet_name="enriched.parquet",
            ),
            validation=ValidationConfig(enabled=False, n_grid=128),
        ),
    )


def test_environment_features_runs_each_target(tmp_path):
    """
    Enabled environment features should invoke tidal pipeline once per target.
    """
    config = _minimal_pipeline_config(tmp_path)
    config.pipeline.environment_features.enabled = True
    config.pipeline.environment_features.targets = [
        TidalAnisotropyTargetConfig(
            name=PIPELINE_TARGET_UNIT,
            dm_particles_file=tmp_path / "unit_dm.dat",
            work_dir=tmp_path / "unit_work",
        ),
        TidalAnisotropyTargetConfig(
            name=PIPELINE_TARGET_FASTPM,
            dm_particles_file=tmp_path / "fastpm_dm",
            work_dir=tmp_path / "fastpm_work",
            catalog_layout="rockstar_list",
        ),
    ]
    calls = []

    def fake_tidal(**kwargs):
        calls.append(kwargs["target_name"])

    state = HaloscopePipelineState()
    with patch(PATCH_TIDAL_TARGET, side_effect=fake_tidal):
        EnvironmentFeaturesElement(config, state).run()

    assert calls == [PIPELINE_TARGET_UNIT, PIPELINE_TARGET_FASTPM]


def test_preprocessing_writes_parquet_tables(tmp_path):
    """
    Preprocessing stage should persist HR/LR Parquet files.
    """
    config = _minimal_pipeline_config(tmp_path)
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2], "M200b_cal": [9.0e11]})
    state = HaloscopePipelineState()

    with patch(PATCH_BUILD_TABLES, return_value=(hr, lr, "M200b_cal")):
        PreprocessingElement(config, state).run()

    assert state.preprocess_paths is not None
    hr_path, lr_path = state.preprocess_paths
    assert hr_path.is_file()
    assert lr_path.is_file()


def test_haloscope_stage_calls_enrichment(tmp_path):
    """
    Haloscope stage should fit the model when enabled.
    """
    config = _minimal_pipeline_config(tmp_path)
    config.pipeline.haloscope.enabled = True
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2], "M200b_cal": [9.0e11]})
    state = HaloscopePipelineState()
    state.preprocess_paths = (tmp_path / "hr.parquet", tmp_path / "lr.parquet")
    hr.to_parquet(state.preprocess_paths[0], index=False)
    lr.to_parquet(state.preprocess_paths[1], index=False)

    with patch(PATCH_ENRICHMENT, return_value=tmp_path / "enriched.parquet") as enrich_mock:
        HaloscopeElement(config, state).run()

    enrich_mock.assert_called_once()
    call_kwargs = enrich_mock.call_args.kwargs
    assert call_kwargs["run_enrichment"] is True
    assert call_kwargs["write_assembly_bias"] is True


def test_validation_stage_runs_assembly_only(tmp_path):
    """
    Validation stage should call enrichment with ``run_enrichment=False``.
    """
    config = _minimal_pipeline_config(tmp_path)
    config.pipeline.haloscope.enabled = True
    config.pipeline.validation.enabled = True
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2], "M200b_cal": [9.0e11]})
    enriched_path = config.output_dir / "enriched.parquet"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    lr.to_parquet(enriched_path, index=False)

    state = HaloscopePipelineState()
    state.preprocess_paths = (tmp_path / "hr.parquet", tmp_path / "lr.parquet")
    hr.to_parquet(state.preprocess_paths[0], index=False)
    lr.to_parquet(state.preprocess_paths[1], index=False)
    state.enrichment_result = enriched_path

    with patch(
        PATCH_ENRICHMENT,
        return_value=HaloscopeEnrichmentRun(output_path=enriched_path),
    ) as enrich_mock:
        ValidationElement(config, state).run()

    enrich_mock.assert_called_once()
    assert enrich_mock.call_args.kwargs["run_enrichment"] is False


def test_run_haloscope_pipeline_orchestrates_stages(tmp_path):
    """
    Full pipeline run should populate ``HaloscopePipelineState``.
    """
    config = _minimal_pipeline_config(tmp_path)
    config.pipeline.haloscope.enabled = True
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2], "M200b_cal": [9.0e11]})

    with (
        patch(PATCH_BUILD_TABLES, return_value=(hr, lr, "M200b_cal")),
        patch(PATCH_ENRICHMENT, return_value=tmp_path / "enriched.parquet"),
    ):
        state = run_haloscope_pipeline(config)

    assert state.preprocess_paths is not None
    assert state.enrichment_result is not None


def test_haloscope_defers_assembly_when_validation_enabled(tmp_path):
    """
    With validation enabled, Haloscope should skip assembly plots in enrichment.
    """
    config = _minimal_pipeline_config(tmp_path)
    config.pipeline.haloscope.enabled = True
    config.pipeline.validation.enabled = True
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2], "M200b_cal": [9.0e11]})
    state = HaloscopePipelineState()
    state.preprocess_paths = (tmp_path / "hr.parquet", tmp_path / "lr.parquet")
    hr.to_parquet(state.preprocess_paths[0], index=False)
    lr.to_parquet(state.preprocess_paths[1], index=False)

    with patch(PATCH_ENRICHMENT, return_value=tmp_path / "enriched.parquet") as enrich_mock:
        HaloscopeElement(config, state).run()

    assert enrich_mock.call_args.kwargs["write_assembly_bias"] is False


def test_validation_only_skips_heavy_stages(tmp_path):
    """
    With only validation enabled, the pipeline should not rebuild preprocess or Haloscope.
    """
    config = _minimal_pipeline_config(tmp_path)
    config.pipeline.environment_features.enabled = False
    config.pipeline.preprocessing.enabled = False
    config.pipeline.haloscope.enabled = False
    config.pipeline.validation.enabled = True
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2], "M200b_cal": [9.0e11]})
    enriched_path = out_dir / config.pipeline.haloscope.enriched_parquet_name
    hr.to_parquet(out_dir / "hr.parquet", index=False)
    lr.to_parquet(out_dir / "lr.parquet", index=False)
    lr.to_parquet(enriched_path, index=False)

    with (
        patch(PATCH_BUILD_TABLES) as build_mock,
        patch(PATCH_ENRICHMENT, return_value=HaloscopeEnrichmentRun(output_path=enriched_path)),
        patch(PATCH_TIDAL_TARGET) as tidal_mock,
    ):
        HaloscopePipeline(config).run()

    build_mock.assert_not_called()
    tidal_mock.assert_not_called()


def test_load_template_json_pipeline_shape():
    """
    Repository template JSON should parse nested pipeline sections.
    """
    template = Path("config/pipeline_haloscope_template.json")
    if not template.is_file():
        pytest.skip("pipeline template not in checkout")
    config = load_haloscope_enrichment_config(template)
    assert config.pipeline.preprocessing.mass_correction.method == "matching_ml"
    assert config.pipeline.haloscope.enabled is True
    assert config.pipeline.validation.enabled is True
    assert config.pipeline.validation.n_grid == 256
