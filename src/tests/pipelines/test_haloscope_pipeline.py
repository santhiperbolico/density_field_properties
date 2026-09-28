"""Tests for multi-stage Haloscope pipeline orchestration."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from density_field_properties.pipelines.config import (
    HaloscopeEnrichmentConfig,
    HaloscopePipelineStages,
    HaloscopeStageConfig,
    PreprocessStageConfig,
    TidalAnisotropyStageConfig,
    TidalAnisotropyTargetConfig,
)
from density_field_properties.pipelines.haloscope_pipeline import (
    _resolve_descriptor_dirs_from_pipeline,
    run_haloscope_pipeline,
)


def _minimal_config(tmp_path: Path, stages: HaloscopePipelineStages) -> HaloscopeEnrichmentConfig:
    """
    Build a minimal enrichment config for pipeline orchestration tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for catalog and output paths.
    stages : HaloscopePipelineStages
        Pipeline stage toggles under test.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Config pointing at placeholder catalog files.
    """
    sim_path = tmp_path / "sim.list"
    fastpm_path = tmp_path / "fastpm.list"
    sim_path.write_text("# header\n", encoding="utf-8")
    fastpm_path.write_text("# header\n", encoding="utf-8")
    return HaloscopeEnrichmentConfig(
        sim_hlist_path=sim_path,
        fastpm_list_path=fastpm_path,
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        box_size_mpc_h=200.0,
        pipeline_stages=stages,
        input_features=("env",),
        run_assembly_bias_plot=False,
    )


def test_run_haloscope_pipeline_executes_tidal_targets_sequentially(tmp_path):
    """
    Enabled tidal-anisotropy stage should process each configured target once.
    """
    stages = HaloscopePipelineStages(
        tidal_anisotropy=TidalAnisotropyStageConfig(
            enabled=True,
            targets=("unit", "fastpm"),
            unit=TidalAnisotropyTargetConfig(
                dm_particles_file=tmp_path / "unit_dm.dat",
                work_dir=tmp_path / "unit_work",
            ),
            fastpm=TidalAnisotropyTargetConfig(
                dm_particles_file=tmp_path / "fastpm_snap",
                work_dir=tmp_path / "fastpm_work",
                catalog_layout="rockstar_list",
            ),
        ),
        preprocess=PreprocessStageConfig(enabled=False),
        haloscope=HaloscopeStageConfig(enabled=False),
    )
    config = _minimal_config(tmp_path, stages)
    processed_targets = []

    def fake_tidal_stage(active_config):
        processed_targets.extend(active_config.pipeline_stages.tidal_anisotropy.targets)

    with patch(
        "density_field_properties.pipelines.haloscope_pipeline.run_tidal_anisotropy_stage",
        side_effect=fake_tidal_stage,
    ):
        result = run_haloscope_pipeline(config)

    assert processed_targets == ["unit", "fastpm"]
    assert result is None


def test_resolve_descriptor_dirs_from_pipeline_work_dirs(tmp_path):
    """
    Tidal descriptor dirs should resolve from pipeline work dirs when unset.
    """
    unit_work = tmp_path / "unit_work"
    fastpm_work = tmp_path / "fastpm_work"
    stages = HaloscopePipelineStages(
        tidal_anisotropy=TidalAnisotropyStageConfig(
            enabled=False,
            unit=TidalAnisotropyTargetConfig(work_dir=unit_work),
            fastpm=TidalAnisotropyTargetConfig(work_dir=fastpm_work),
        ),
    )
    config = _minimal_config(tmp_path, stages)
    config.input_features = ("t_over_u", "tidal_anisotropy")

    _resolve_descriptor_dirs_from_pipeline(config)

    assert config.unit_descriptors_dir == unit_work / "tidal_anisotropy"
    assert config.fastpm_descriptors_dir == fastpm_work / "tidal_anisotropy"


def test_run_haloscope_pipeline_preprocess_only(tmp_path):
    """
    Preprocess stage should persist HR/LR tables without running Haloscope.
    """
    stages = HaloscopePipelineStages(
        preprocess=PreprocessStageConfig(enabled=True),
        haloscope=HaloscopeStageConfig(enabled=False),
    )
    config = _minimal_config(tmp_path, stages)
    hr = pd.DataFrame({"M200b": [1.0e12], "env": [0.1]})
    lr = pd.DataFrame({"M200b": [8.0e11], "env": [0.2]})

    with patch(
        "density_field_properties.pipelines.haloscope_pipeline.build_haloscope_feature_tables",
        return_value=(hr, lr, "M200b_cal"),
    ):
        result = run_haloscope_pipeline(config)

    assert result is not None
    hr_path, lr_path = result
    assert hr_path.is_file()
    assert lr_path.is_file()
    assert pd.read_parquet(hr_path).shape[0] == 1
