"""Tests for tidal Haloscope feature table builders."""

import sys

import numpy as np
import pytest

if sys.version_info < (3, 11):
    pytest.skip("tidal feature table tests require Python 3.11+", allow_module_level=True)
import pandas as pd
import pytest

from density_field_properties.preprocessing.tidal_feature_table import (
    build_tidal_feature_tables,
)


def _sim_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "x": [100.0, 200.0],
            "y": [100.0, 200.0],
            "z": [100.0, 200.0],
            "M200b": [1.0e12, 2.0e12],
            "cv": [0.4, 0.5],
            "Spin": [0.03, 0.04],
            "ca": [0.8, 0.7],
            "ba": [0.6, 0.5],
        }
    )


def _fastpm_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [150.0, 250.0],
            "y": [150.0, 250.0],
            "z": [150.0, 250.0],
            "M200b": [8.0e11, 1.5e12],
        }
    )


def test_build_tidal_feature_tables_requires_descriptor_batches(tmp_path):
    """
    Tidal builder fails fast when descriptor directories are missing.
    """
    hr = _sim_catalog()
    hr["t_over_u"] = [0.9, 1.0]
    lr = _fastpm_catalog()
    lr["t_over_u"] = [0.8, 0.7]
    with pytest.raises(FileNotFoundError):
        build_tidal_feature_tables(
            hr,
            lr,
            tmp_path / "unit",
            tmp_path / "fastpm",
            calibrate_mass=False,
            sim_boxsize_mpc_h=1000.0,
            fastpm_boxsize_mpc_h=1000.0,
            n_grid=512,
            input_features=("t_over_u", "tidal_anisotropy"),
        )


def test_build_tidal_feature_tables_joins_descriptors(tmp_path):
    """
    Tidal builder merges descriptor batches and validates schemas.
    """
    unit_dir = tmp_path / "unit"
    fastpm_dir = tmp_path / "fastpm"
    unit_dir.mkdir()
    fastpm_dir.mkdir()
    boxsize = 1000.0
    n_grid = 512
    cell_size = boxsize / n_grid
    unit_dir.joinpath("0_halo_environment_descriptors.txt").write_text(
        "# header\n1 10 20 30 1.0e12 1.5 0.42 -0.2\n"
    )
    fastpm_dir.joinpath("0_halo_environment_descriptors.txt").write_text(
        "# header\n1 15 25 35 1.0e12 1.5 0.51 -0.1\n"
    )
    hr = _sim_catalog()
    hr["t_over_u"] = [0.9, 1.0]
    lr = _fastpm_catalog()
    lr["t_over_u"] = [0.8, 0.7]
    hr.iloc[0, hr.columns.get_loc("x")] = 10 * cell_size + 0.1
    hr.iloc[0, hr.columns.get_loc("y")] = 20 * cell_size + 0.1
    hr.iloc[0, hr.columns.get_loc("z")] = 30 * cell_size + 0.1
    lr.iloc[0, lr.columns.get_loc("x")] = 15 * cell_size + 0.1
    lr.iloc[0, lr.columns.get_loc("y")] = 25 * cell_size + 0.1
    lr.iloc[0, lr.columns.get_loc("z")] = 35 * cell_size + 0.1

    hr_out, lr_out, mass_column = build_tidal_feature_tables(
        hr,
        lr,
        unit_dir,
        fastpm_dir,
        calibrate_mass=False,
        sim_boxsize_mpc_h=boxsize,
        fastpm_boxsize_mpc_h=boxsize,
        n_grid=n_grid,
        input_features=("t_over_u", "tidal_anisotropy"),
        max_descriptor_batch_files=1,
    )
    assert mass_column == "M200b"
    assert np.isfinite(hr_out["tidal_anisotropy"]).all()
    assert np.isfinite(lr_out["tidal_anisotropy"]).all()
