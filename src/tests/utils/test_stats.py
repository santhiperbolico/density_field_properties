"""Unit tests for utils.stats helpers."""

import numpy as np
import pytest

from density_field_properties.utils.stats import (
    bin_midpoints,
    central_68_scatter,
    confidence_intervals,
    safe_ratio,
    standard_error_mean,
)


def test_bin_midpoints_returns_centers():
    """Bin midpoints must lie between adjacent edges."""
    edges = np.array([1.0, 2.0, 4.0])
    centers = bin_midpoints(edges)
    assert centers.tolist() == [1.5, 3.0]


def test_central_68_scatter_empty_returns_nan():
    """Empty input must yield NaN scatter."""
    assert np.isnan(central_68_scatter(np.array([])))


@pytest.mark.parametrize(
    "values,expected",
    [
        (np.array([]), np.nan),
        (np.array([2.0]), 0.0),
        (np.array([0.0, 2.0]), 1.0),
    ],
)
def test_standard_error_mean(values, expected):
    """Standard error must follow std/sqrt(n) with ddof=1."""
    result = standard_error_mean(values)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert np.isclose(result, expected)


def test_safe_ratio_handles_zero_denominator():
    """Zero denominators must produce NaN ratios."""
    result = safe_ratio(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert result[0] == 1.0
    assert np.isnan(result[1])


def test_confidence_intervals_returns_monotonic_levels_for_spread_pdf():
    """Spread PDFs must yield usable ascending contour levels."""
    grid = np.zeros((5, 5))
    grid[2, 2] = 0.5
    grid[2, 3] = 0.3
    grid[3, 2] = 0.2
    pdf = grid / grid.sum()
    levels = confidence_intervals(pdf)
    assert levels.shape == (4,)
    assert np.all(np.diff(levels[:-1]) >= 0.0)
    assert levels[-1] == 1.0


def test_confidence_intervals_handles_single_cell_pdf():
    """Sparse hold-out bins can collapse to one histogram cell."""
    pdf = np.zeros((15, 15))
    pdf[7, 7] = 1.0
    levels = confidence_intervals(pdf)
    assert levels.shape == (4,)
    assert np.all(np.isfinite(levels))
    assert levels[-1] == 1.0
