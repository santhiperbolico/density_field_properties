"""Tests for multi-stage Haloscope pipeline configuration."""

import json
from pathlib import Path

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    load_haloscope_enrichment_config,
)


def test_load_pipeline_stage_config(tmp_path):
    """
    Pipeline stage toggles and target aliases should parse from JSON.
    """
    config_path = tmp_path / "pipeline_run.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "sim_hlist": str(tmp_path / "sim.list"),
                    "fastpm_list": str(tmp_path / "fastpm.list"),
                    "repo_root": str(tmp_path),
                    "output_dir": str(tmp_path / "out"),
                },
                "sample": {"box_size_mpc_h": 200},
                "haloscope": {"input_features": ["t_over_u", "tidal_anisotropy"]},
                "pipeline": {
                    "tidal_anisotropy": {
                        "enabled": True,
                        "targets": ["hr", "lr"],
                        "unit": {
                            "dm_particles_file": "unit_dm.dat",
                            "work_dir": "out/unit",
                            "catalog_layout": "hlist",
                        },
                        "fastpm": {
                            "dm_particles_file": "fastpm_snap/1",
                            "work_dir": "out/fastpm",
                            "catalog_layout": "rockstar_list",
                            "read_tensor_from_disk": True,
                        },
                    },
                    "preprocess": {
                        "enabled": True,
                        "hr_parquet_name": "hr.parquet",
                        "lr_parquet_name": "lr.parquet",
                    },
                    "haloscope": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_haloscope_enrichment_config(config_path)
    stages = config.pipeline_stages

    assert stages.tidal_anisotropy.enabled is True
    assert stages.tidal_anisotropy.targets == (PIPELINE_TARGET_UNIT, PIPELINE_TARGET_FASTPM)
    assert stages.tidal_anisotropy.unit.dm_particles_file == Path("unit_dm.dat")
    assert stages.tidal_anisotropy.unit.work_dir == Path("out/unit")
    assert stages.tidal_anisotropy.fastpm.read_tensor_from_disk is True
    assert stages.preprocess.enabled is True
    assert stages.preprocess.hr_parquet_name == "hr.parquet"
    assert stages.haloscope.enabled is False
