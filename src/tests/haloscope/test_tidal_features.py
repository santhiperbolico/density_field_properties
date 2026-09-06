"""Unit tests for tidal Haloscope feature attachment."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.haloscope.sim_to_fastpm.tidal_features import (
    attach_tidal_anisotropy,
    filter_finite_input_features,
    load_halo_environment_descriptor_table,
)


def test_load_halo_environment_descriptor_table_reads_first_batch(tmp_path):
    """
    Load a single descriptor batch file with the expected column names.
    """
    descriptor_file = tmp_path / "0_halo_environment_descriptors.txt"
    descriptor_file.write_text(
        "# Halo ID, X, Y, Z, M200b, R_G, Tidal Anisotropy, Overdensity\n"
        "1 10 20 30 1.0e12 1.5 0.8 -0.2\n"
        "2 11 21 31 2.0e12 1.6 0.9 -0.1\n"
    )
    frame = load_halo_environment_descriptor_table(tmp_path, max_batch_files=1)
    assert list(frame.columns) == [
        "descriptor_halo_id",
        "cell_x",
        "cell_y",
        "cell_z",
        "descriptor_m200b",
        "halo_rg",
        "tidal_anisotropy",
        "overdensity",
    ]
    assert len(frame) == 2


def test_attach_tidal_anisotropy_merges_by_grid_cell(tmp_path):
    """
    Attach tidal anisotropy using periodic grid-cell coordinates.
    """
    boxsize = 1000.0
    n_grid = 512
    cell_size = boxsize / n_grid
    descriptor_file = tmp_path / "0_halo_environment_descriptors.txt"
    descriptor_file.write_text("# header\n1 10 20 30 1.0e12 1.5 0.42 -0.2\n")
    catalog = pd.DataFrame(
        {
            "x": [10 * cell_size + 0.1],
            "y": [20 * cell_size + 0.1],
            "z": [30 * cell_size + 0.1],
            "M200b": [1.0e12],
        }
    )
    enriched = attach_tidal_anisotropy(
        catalog,
        tmp_path,
        boxsize,
        n_grid,
        max_batch_files=1,
    )
    assert enriched["tidal_anisotropy"].iloc[0] == pytest.approx(0.42)


def test_filter_finite_input_features_drops_nan_rows():
    """
    Remove rows with missing Haloscope input features.
    """
    catalog = pd.DataFrame(
        {
            "t_over_u": [0.8, np.nan, 1.1],
            "tidal_anisotropy": [0.2, 0.3, np.nan],
        }
    )
    filtered = filter_finite_input_features(
        catalog,
        ("t_over_u", "tidal_anisotropy"),
    )
    assert len(filtered) == 1
    assert filtered["t_over_u"].iloc[0] == pytest.approx(0.8)
