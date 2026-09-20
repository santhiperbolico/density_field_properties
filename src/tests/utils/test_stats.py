"""Unit tests for utils.stats helpers."""

import numpy as np

from density_field_properties.utils.stats import (
    bin_midpoints,
    central_68_scatter,
    safe_ratio,
)


def test_bin_midpoints_returns_centers():
    """Bin midpoints must lie between adjacent edges."""
    edges = np.array([1.0, 2.0, 4.0])
    centers = bin_midpoints(edges)
    assert centers.tolist() == [1.5, 3.0]


def test_central_68_scatter_empty_returns_nan():
    """Empty input must yield NaN scatter."""
    assert np.isnan(central_68_scatter(np.array([])))


def test_safe_ratio_handles_zero_denominator():
    """Zero denominators must produce NaN ratios."""
    result = safe_ratio(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert result[0] == 1.0
    assert np.isnan(result[1])
