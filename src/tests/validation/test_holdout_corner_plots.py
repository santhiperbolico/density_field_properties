"""Tests for SIM hold-out corner-plot validation helpers."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from density_field_properties.haloscope import default_mass_bin_edges
from density_field_properties.pipelines.config import HOLDOUT_CORNER_PLOT_FILENAME_TEMPLATE
from density_field_properties.validation.holdout_corner_plots import (
    format_holdout_corner_plot_title,
    holdout_corner_plot_path,
    write_sim_holdout_corner_plots,
)


def _synthetic_catalog(n_halos: int, seed: int) -> pd.DataFrame:
    """
    Build a minimal SIM catalog for hold-out validation tests.

    Parameters
    ----------
    n_halos : int
        Number of synthetic halos.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Catalog with INPUT and OUTPUT feature columns.
    """
    rng = np.random.default_rng(seed)
    masses = 10 ** rng.uniform(11.0, 13.5, size=n_halos)
    return pd.DataFrame(
        {
            "M200b": masses,
            "env": rng.normal(size=n_halos),
            "cv": rng.uniform(1.0, 10.0, size=n_halos),
            "Spin": rng.uniform(0.01, 0.08, size=n_halos),
            "ca": rng.uniform(0.5, 1.0, size=n_halos),
            "ba": rng.uniform(0.5, 1.0, size=n_halos),
        }
    )


@pytest.mark.parametrize(
    "input_features,expected_feature_text",
    [
        (("env",), "env"),
        (("t_over_u", "tidal_anisotropy"), "T/|U| + tidal anisotropy"),
    ],
)
def test_format_holdout_corner_plot_title(input_features, expected_feature_text):
    """
    Hold-out corner titles should describe the Haloscope INPUT features.
    """
    title = format_holdout_corner_plot_title(input_features, bin_index=2)

    assert title == f"SIM hold-out validation — mass bin 2 (input: {expected_feature_text})"


def test_holdout_corner_plot_path_uses_configured_template(tmp_path):
    """
    Output paths should follow the configured filename template.
    """
    expected_name = HOLDOUT_CORNER_PLOT_FILENAME_TEMPLATE.format(bin_index=3)
    assert holdout_corner_plot_path(tmp_path, 3) == tmp_path / expected_name


@patch("density_field_properties.validation.holdout_corner_plots.corner_plot_sim_validation")
def test_write_sim_holdout_corner_plots_writes_one_pdf_per_populated_bin(
    mock_corner_plot,
    tmp_path,
):
    """
    Enabled corner plots should save one PDF for each populated hold-out bin.
    """
    halos_sim = _synthetic_catalog(200, seed=1)
    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))

    results = write_sim_holdout_corner_plots(
        halos_sim,
        bin_edges,
        tmp_path,
        input_features=("env",),
        output_features=("cv", "Spin", "ca", "ba"),
        min_bin_size=5,
        write_plots=True,
        random_seed=0,
    )

    populated_bins = [bin_index for bin_index, y_true, _ in results if y_true is not None]
    assert populated_bins
    assert mock_corner_plot.call_count == len(populated_bins)
    written_paths = {Path(call.args[3]) for call in mock_corner_plot.call_args_list}
    for bin_index in populated_bins:
        assert holdout_corner_plot_path(tmp_path, bin_index) in written_paths


@patch("density_field_properties.validation.holdout_corner_plots.release_validation_memory")
@patch("density_field_properties.validation.holdout_corner_plots.corner_plot_sim_validation")
def test_write_sim_holdout_corner_plots_releases_memory_after_each_bin(
    mock_corner_plot,
    mock_release_memory,
    tmp_path,
):
    """
    Corner-plot generation should release memory after every populated bin.
    """
    halos_sim = _synthetic_catalog(200, seed=3)
    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))

    write_sim_holdout_corner_plots(
        halos_sim,
        bin_edges,
        tmp_path,
        input_features=("env",),
        output_features=("cv", "Spin", "ca", "ba"),
        min_bin_size=5,
        write_plots=True,
        random_seed=0,
    )

    populated_bins = mock_corner_plot.call_count
    assert populated_bins > 0
    assert mock_release_memory.call_count == populated_bins + 1


@patch("density_field_properties.validation.holdout_corner_plots.corner_plot_sim_validation")
def test_write_sim_holdout_corner_plots_skips_pdf_when_disabled(
    mock_corner_plot,
    tmp_path,
):
    """
    Hold-out validation should run without writing PDFs when plots are disabled.
    """
    halos_sim = _synthetic_catalog(120, seed=2)
    bin_edges = default_mass_bin_edges(np.log10(halos_sim["M200b"].max()))

    write_sim_holdout_corner_plots(
        halos_sim,
        bin_edges,
        tmp_path,
        input_features=("env",),
        output_features=("cv", "Spin", "ca", "ba"),
        min_bin_size=5,
        write_plots=False,
        random_seed=0,
    )

    mock_corner_plot.assert_not_called()
    assert list(tmp_path.glob("*.pdf")) == []
