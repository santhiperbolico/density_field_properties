"""Tests for Haloscope mass_calibration JSON parsing."""

import json
from pathlib import Path

import pytest

from density_field_properties.pipelines.config import load_haloscope_enrichment_config
from density_field_properties.pipelines.run_defaults import CALIBRATE_MASS
from density_field_properties.preprocessing.mass_calibration_config import (
    DEFAULT_MATCHING_K_NEIGHBOURS,
    DEFAULT_R_MATCH_MPC_H,
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
)


def _minimal_config_payload(tmp_path: Path, extra: dict) -> Path:
    """
    Write a minimal Haloscope JSON config under ``tmp_path``.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    extra : dict
        Keys merged into the root JSON object.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    base = {
        "paths": {
            "sim_hlist": str(tmp_path / "sim.list"),
            "fastpm_list": str(tmp_path / "fastpm.list"),
            "repo_root": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        },
    }
    base.update(extra)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(base), encoding="utf-8")
    return config_path


def test_mass_calibration_defaults_match_legacy_calibrate_mass(tmp_path):
    """
    Absent ``mass_calibration`` should enable AM like legacy ``CALIBRATE_MASS``.
    """
    config_path = _minimal_config_payload(tmp_path, {})
    config = load_haloscope_enrichment_config(config_path)

    assert config.mass_calibration.enabled is CALIBRATE_MASS
    assert config.mass_calibration.method == MASS_CALIBRATION_METHOD_ABUNDANCE
    assert config.mass_calibration.calibrated_column == "M200b_cal"


def test_mass_calibration_nested_block_parses_matching_defaults(tmp_path):
    """
    Nested ``mass_calibration`` should parse method and Forero-like matching defaults.
    """
    config_path = _minimal_config_payload(
        tmp_path,
        {
            "mass_calibration": {
                "enabled": True,
                "method": MASS_CALIBRATION_METHOD_MATCHING_1TO1,
                "matching": {
                    "k_neighbours": 12,
                    "r_match_mpc_h": 4.5,
                },
            },
        },
    )
    config = load_haloscope_enrichment_config(config_path)

    assert config.mass_calibration.method == MASS_CALIBRATION_METHOD_MATCHING_1TO1
    assert config.mass_calibration.matching.k_neighbours == 12
    assert config.mass_calibration.matching.r_match_mpc_h == 4.5
    assert config.mass_calibration.matching.max_mass_distance == pytest.approx(0.95)


def test_haloscope_legacy_calibrate_mass_flag(tmp_path):
    """
    ``haloscope.calibrate_mass`` should set ``mass_calibration.enabled``.
    """
    config_path = _minimal_config_payload(
        tmp_path,
        {"haloscope": {"calibrate_mass": False}},
    )
    config = load_haloscope_enrichment_config(config_path)

    assert config.mass_calibration.enabled is False


def test_unknown_mass_calibration_method_raises(tmp_path):
    """
    Unsupported method keys should fail at config load time.
    """
    config_path = _minimal_config_payload(
        tmp_path,
        {"mass_calibration": {"method": "invalid_method"}},
    )
    with pytest.raises(ValueError, match="Unknown mass calibration method"):
        load_haloscope_enrichment_config(config_path)


def test_matching_defaults_when_block_empty(tmp_path):
    """
    Empty ``matching`` object should keep module-level defaults.
    """
    config_path = _minimal_config_payload(
        tmp_path,
        {"mass_calibration": {"matching": {}}},
    )
    config = load_haloscope_enrichment_config(config_path)

    assert config.mass_calibration.matching.k_neighbours == DEFAULT_MATCHING_K_NEIGHBOURS
    assert config.mass_calibration.matching.r_match_mpc_h == DEFAULT_R_MATCH_MPC_H
