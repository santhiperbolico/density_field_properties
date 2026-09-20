"""Tests for tidal anisotropy feature attachment."""

import sys

import pandas as pd
import pytest

if sys.version_info < (3, 11):
    pytest.skip("tidal anisotropy tests require Python 3.11+", allow_module_level=True)

from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    TidalAnisotropyFeature,
    _load_halo_environment_descriptor_table,
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
    frame = _load_halo_environment_descriptor_table(tmp_path, max_batch_files=1)
    assert "tidal_anisotropy" in frame.columns
    assert len(frame) == 2


def test_tidal_anisotropy_feature_merges_by_grid_cell(tmp_path):
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
    run_context = SimulationRunContext(
        boxsize_mpc_h=boxsize,
        tidal_descriptors_dir=tmp_path,
        tidal_n_grid=n_grid,
        max_descriptor_batch_files=1,
    )
    enriched = TidalAnisotropyFeature().attach(catalog, run_context)
    assert enriched["tidal_anisotropy"].iloc[0] == pytest.approx(0.42)
