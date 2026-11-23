"""Tests for Haloscope pipeline JSON configuration."""

import json

import pytest

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    load_haloscope_enrichment_config,
    mass_correction_settings_from_preprocessing,
)
from density_field_properties.preprocessing.mass_correction.config import (
    DEFAULT_MATCHING_K_NEIGHBOURS,
    MASS_CORRECTION_METHOD_MATCHING_1TO1,
)


def test_load_pipeline_environment_targets(tmp_path):
    """
    Environment feature targets and preprocessing mass correction should parse from JSON.
    """
    config_path = tmp_path / "pipeline_run.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "unit_test",
                "general": {
                    "sim_hlist": str(tmp_path / "sim.list"),
                    "fastpm_list": str(tmp_path / "fastpm.list"),
                    "repo_root": str(tmp_path),
                    "output_dir": str(tmp_path / "out"),
                    "input_features": ["tidal_anisotropy"],
                },
                "pipeline": {
                    "environment_features": {
                        "enabled": True,
                        "n_grid": 128,
                        "targets": [
                            {
                                "name": "unit",
                                "dm_particles_file": "unit_dm.dat",
                                "work_dir": "out/unit",
                                "catalog_layout": "hlist",
                            },
                            {
                                "name": "fastpm",
                                "dm_particles_file": "fastpm_snap/1",
                                "work_dir": "out/fastpm",
                                "catalog_layout": "rockstar_list",
                                "read_tensor_from_disk": True,
                            },
                        ],
                    },
                    "preprocessing": {
                        "enabled": True,
                        "hr_parquet_name": "hr.parquet",
                        "lr_parquet_name": "lr.parquet",
                        "sample": {
                            "box_size_mpc_h": 200,
                            "min_mass_msun_h": 1.0e11,
                            "get_min_mass_lr_from_hmf": True,
                        },
                        "mass_correction": {
                            "method": MASS_CORRECTION_METHOD_MATCHING_1TO1,
                            "correction_column": "M200b_cal",
                            "params_method": {
                                "matching": {"k_neighbours": 12, "r_match_mpc_h": 4.5},
                            },
                        },
                    },
                    "haloscope": {"enabled": False},
                    "validation": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_haloscope_enrichment_config(config_path)
    env = config.pipeline.environment_features

    assert env.enabled is True
    assert env.targets[0].name == PIPELINE_TARGET_UNIT
    assert env.targets[1].name == PIPELINE_TARGET_FASTPM
    assert env.targets[1].read_tensor_from_disk is True
    mass_correction = mass_correction_settings_from_preprocessing(config.pipeline.preprocessing)
    assert mass_correction.method == MASS_CORRECTION_METHOD_MATCHING_1TO1
    assert mass_correction.matching.k_neighbours == 12
    assert mass_correction.auto_lr_correction_min_mass_from_hmf is True


def test_unknown_mass_correction_method_raises(tmp_path):
    """
    Unsupported mass correction methods should fail when loading JSON.
    """
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps(
            {
                "general": {
                    "sim_hlist": "sim.list",
                    "fastpm_list": "fastpm.list",
                },
                "pipeline": {
                    "preprocessing": {
                        "hr_parquet_name": "hr.parquet",
                        "lr_parquet_name": "lr.parquet",
                        "sample": {
                            "box_size_mpc_h": 200,
                            "min_mass_msun_h": 1.0e11,
                            "get_min_mass_lr_from_hmf": False,
                        },
                        "mass_correction": {"method": "invalid_method"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown mass correction method"):
        load_haloscope_enrichment_config(config_path)


def test_matching_defaults_from_empty_params(tmp_path):
    """
    Empty matching params should keep module-level defaults.
    """
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "general": {"sim_hlist": "a", "fastpm_list": "b"},
                "pipeline": {
                    "preprocessing": {
                        "hr_parquet_name": "hr.parquet",
                        "lr_parquet_name": "lr.parquet",
                        "sample": {
                            "box_size_mpc_h": 200,
                            "min_mass_msun_h": 1.0e11,
                            "get_min_mass_lr_from_hmf": False,
                        },
                        "mass_correction": {
                            "method": "abundance_matching",
                            "params_method": {"matching": {}},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_haloscope_enrichment_config(config_path)
    mass_correction = mass_correction_settings_from_preprocessing(config.pipeline.preprocessing)
    assert mass_correction.matching.k_neighbours == DEFAULT_MATCHING_K_NEIGHBOURS
