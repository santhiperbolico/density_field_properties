"""Tests for preprocessing sample filters."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.filters import (
    filter_finite_input_features,
    filter_host_training_halos,
    filter_min_m200b,
    filter_positive_mass,
)


def test_filter_host_training_halos_keeps_hosts_with_finite_cv():
    """
    Host halos with positive mass and finite concentration are retained.
    """
    frame = pd.DataFrame(
        {
            "pid": [-1, 10, -1, -1],
            "M200b": [1.0e12, 1.0e12, 0.0, 1.0e12],
            "cv": [0.5, 0.4, 0.3, np.nan],
        }
    )
    filtered = filter_host_training_halos(frame)
    assert len(filtered) == 1
    assert filtered["pid"].iloc[0] == -1


@pytest.mark.parametrize(
    "masses,expected_count",
    [
        ([1.0e12, 0.0, -1.0], 1),
        ([2.0e11, 3.0e11], 2),
    ],
)
def test_filter_positive_mass(masses, expected_count):
    """
    Only halos with strictly positive mass remain.
    """
    frame = pd.DataFrame({"M200b": masses})
    filtered = filter_positive_mass(frame)
    assert len(filtered) == expected_count


def test_filter_min_m200b_applies_threshold():
    """
    Halos below the minimum mass threshold are removed.
    """
    min_mass = 2.4e10
    frame = pd.DataFrame({"M200b": [min_mass - 1.0, min_mass, min_mass + 1.0]})
    filtered = filter_min_m200b(frame, min_mass)
    assert len(filtered) == 2


def test_filter_finite_input_features_drops_nan_rows():
    """
    Remove rows with missing Haloscope input features.
    """
    catalog = pd.DataFrame(
        {
            "env": [0.8, np.nan, 1.1],
            "t_over_u": [0.2, 0.3, np.nan],
        }
    )
    filtered = filter_finite_input_features(catalog, ("env", "t_over_u"))
    assert len(filtered) == 1
    assert filtered["env"].iloc[0] == pytest.approx(0.8)
