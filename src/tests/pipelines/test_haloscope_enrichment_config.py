"""Unit tests for Haloscope enrichment run configuration."""

import json
from pathlib import Path

import pytest

from density_field_properties.pipelines.config import (
    DEFAULT_ENV_SMOKE_CONFIG,
    HaloscopeEnrichmentConfig,
    HaloscopePipelineStages,
    TidalAnisotropyStageConfig,
    TidalAnisotropyTargetConfig,
    load_haloscope_enrichment_config,
    resolve_dm_mass_particle_msun_h,
    resolve_fastpm_boxsize_mpc_h,
    resolve_fastpm_dm_particles_path,
    resolve_sim_boxsize_mpc_h,
    resolve_sim_dm_particles_path,
)
from density_field_properties.pipelines.run_defaults import (
    FASTPM_BOXSIZE_MPC_H,
    SIM_BOXSIZE_MPC_H,
)


def test_load_env_smoke_config_from_repo_defaults(tmp_path):
    """
    Smoke JSON should resolve null catalog paths to config.py defaults.
    """
    config_path = Path(DEFAULT_ENV_SMOKE_CONFIG)
    if not config_path.is_file():
        pytest.skip("Default smoke config not available in this checkout")

    config = load_haloscope_enrichment_config(config_path)

    assert config.max_sim_halos == 8000
    assert config.max_fastpm_halos == 8000
    assert config.input_features == ("env",)
    assert config.min_bin_size == 5
    assert config.run_assembly_bias_plot is False
    assert config.sim_hlist_path.is_absolute()
    assert config.fastpm_list_path.is_absolute()
    assert config.box_size_mpc_h is None
    assert resolve_sim_boxsize_mpc_h(config) == SIM_BOXSIZE_MPC_H
    assert resolve_fastpm_boxsize_mpc_h(config) == FASTPM_BOXSIZE_MPC_H


def test_load_config_from_custom_json(tmp_path):
    """
    Custom JSON values should override defaults while keeping typed fields.
    """
    config_path = tmp_path / "custom_run.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "sim_hlist": str(tmp_path / "sim.list"),
                    "fastpm_list": str(tmp_path / "fastpm.list"),
                    "repo_root": str(tmp_path),
                    "output_dir": str(tmp_path / "out"),
                },
                "sample": {
                    "box_size_mpc_h": 200,
                    "max_sim_halos": 123,
                    "max_fastpm_halos": 456,
                    "max_descriptor_batch_files": 2,
                },
                "haloscope": {
                    "input_features": ["t_over_u", "tidal_anisotropy"],
                    "min_bin_size": 7,
                    "run_holdout_validation": False,
                    "enriched_parquet_name": "enriched.parquet",
                },
                "tidal": {
                    "n_grid": 256,
                    "unit_descriptors_dir": "unit_desc",
                    "fastpm_descriptors_dir": "fastpm_desc",
                },
                "validation": {
                    "assembly_bias": {
                        "enabled": True,
                        "n_grid": 64,
                    },
                    "holdout_corner_plots": {
                        "enabled": True,
                    },
                },
                "collect_tables": True,
            }
        ),
        encoding="utf-8",
    )

    config = load_haloscope_enrichment_config(config_path)

    assert isinstance(config, HaloscopeEnrichmentConfig)
    assert config.sim_hlist_path == tmp_path / "sim.list"
    assert config.fastpm_list_path == tmp_path / "fastpm.list"
    assert config.repo_root == tmp_path
    assert config.output_dir == tmp_path / "out"
    assert config.box_size_mpc_h == 200
    assert resolve_sim_boxsize_mpc_h(config) == 200
    assert resolve_fastpm_boxsize_mpc_h(config) == 200
    assert config.max_sim_halos == 123
    assert config.max_fastpm_halos == 456
    assert config.max_descriptor_batch_files == 2
    assert config.input_features == ("t_over_u", "tidal_anisotropy")
    assert config.min_bin_size == 7
    assert config.run_holdout_validation is False
    assert config.tidal_n_grid == 256
    assert config.unit_descriptors_dir == Path("unit_desc")
    assert config.fastpm_descriptors_dir == Path("fastpm_desc")
    assert config.enriched_parquet_name == "enriched.parquet"
    assert config.run_assembly_bias_plot is True
    assert config.assembly_bias_n_grid == 64
    assert config.run_holdout_corner_plots is True
    assert config.collect_tables is True


def test_resolve_assembly_bias_paths_from_pipeline_config(tmp_path):
    """
    Assembly-bias helpers should read DM paths from the tidal pipeline stage.
    """
    unit_dm = tmp_path / "unit_dm.dat"
    fastpm_dm = tmp_path / "fastpm_dm"
    config = HaloscopeEnrichmentConfig(
        sim_hlist_path=tmp_path / "sim.list",
        fastpm_list_path=tmp_path / "fastpm.list",
        box_size_mpc_h=200.0,
        pipeline_stages=HaloscopePipelineStages(),
    )
    config.pipeline_stages.tidal_anisotropy = TidalAnisotropyStageConfig(
        enabled=True,
        mass_particle=9.9e8,
        unit=TidalAnisotropyTargetConfig(dm_particles_file=unit_dm),
        fastpm=TidalAnisotropyTargetConfig(dm_particles_file=fastpm_dm),
    )

    assert resolve_sim_boxsize_mpc_h(config) == 200.0
    assert resolve_fastpm_boxsize_mpc_h(config) == 200.0
    assert resolve_sim_dm_particles_path(config) == unit_dm
    assert resolve_fastpm_dm_particles_path(config) == fastpm_dm
    assert resolve_dm_mass_particle_msun_h(config) == 9.9e8
