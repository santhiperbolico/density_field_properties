"""Pure numeric helpers shared by validation and notebooks."""

from typing import Tuple

import numpy as np
from scipy import interpolate, stats

_TARGET_CONFIDENCE_FRACTIONS = np.array([0.95, 0.68, 0.40])
_CONFIDENCE_THRESHOLD_SAMPLES = 20
_INTEGRAL_RANGE_EPS = 1e-12


def central_68_scatter(values: np.ndarray) -> float:
    """
    Half-width of the central 68.3% interval (1-sigma for a Gaussian).

    Parameters
    ----------
    values : np.ndarray
        Sample in one mass bin.

    Returns
    -------
    float
        ``(P84.15 - P15.85) / 2``, or ``nan`` for empty input.
    """
    if values.size == 0:
        return float("nan")
    upper = np.percentile(values, 84.15)
    lower = np.percentile(values, 15.85)
    return 0.5 * (upper - lower)


def standard_error_mean(values: np.ndarray) -> float:
    """
    Standard error of the mean for a one-dimensional sample.

    Parameters
    ----------
    values : np.ndarray
        Sample values in one mass bin.

    Returns
    -------
    float
        ``std(values, ddof=1) / sqrt(n)``, or ``nan`` for empty input.
    """
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def bin_midpoints(edges: np.ndarray) -> np.ndarray:
    """
    Midpoints of a monotonic bin-edge vector.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges.

    Returns
    -------
    np.ndarray
        Bin centers.
    """
    return (edges[1:] + edges[:-1]) / 2


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Element-wise ratio with NaNs where the denominator is non-finite or zero.

    Parameters
    ----------
    numerator : np.ndarray
        Values in the numerator.
    denominator : np.ndarray
        Values in the denominator.

    Returns
    -------
    np.ndarray
        ``numerator / denominator`` with invalid entries set to NaN.
    """
    safe_denominator = np.where(np.abs(denominator) > 0.0, denominator, np.nan)
    return numerator / safe_denominator


def interpolated_ratio(
    model_mass: np.ndarray,
    model_bias: np.ndarray,
    reference_mass: np.ndarray,
    reference_bias: np.ndarray,
) -> np.ndarray:
    """
    Ratio of model to reference bias on the reference mass grid.

    Parameters
    ----------
    model_mass : np.ndarray
        Mass support for the model curve.
    model_bias : np.ndarray
        Model bias on ``model_mass``.
    reference_mass : np.ndarray
        Mass grid for the output ratio.
    reference_bias : np.ndarray
        Reference bias on ``reference_mass``.

    Returns
    -------
    np.ndarray
        Interpolated ``model / reference`` on ``reference_mass``.
    """
    if len(model_mass) < 2:
        return np.full_like(reference_bias, np.nan, dtype=float)
    interpolator = interpolate.interp1d(
        model_mass,
        model_bias,
        bounds_error=False,
        fill_value=np.nan,
    )
    return safe_ratio(interpolator(reference_mass), reference_bias)


def _dedupe_increasing_integral_levels(
    integral: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse duplicate enclosed-probability levels.

    Parameters
    ----------
    integral : np.ndarray
        Enclosed probability for each threshold probe.
    thresholds : np.ndarray
        PDF thresholds aligned with ``integral``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Strictly increasing integral support and matching thresholds.
    """
    order = np.argsort(integral)
    sorted_integral = integral[order]
    sorted_thresholds = thresholds[order]
    unique_integral = []
    unique_thresholds = []
    for level, threshold in zip(sorted_integral, sorted_thresholds):
        if unique_integral and level == unique_integral[-1]:
            unique_thresholds[-1] = min(unique_thresholds[-1], threshold)
            continue
        unique_integral.append(level)
        unique_thresholds.append(threshold)
    return np.asarray(unique_integral), np.asarray(unique_thresholds)


def _degenerate_confidence_levels(pdf_max: float) -> np.ndarray:
    """
    Fallback contour levels for a single-cell or flat PDF.

    Parameters
    ----------
    pdf_max : float
        Maximum value of the normalized PDF grid.

    Returns
    -------
    np.ndarray
        Ascending contour levels plus the trailing sentinel.
    """
    scaled = np.sort(_TARGET_CONFIDENCE_FRACTIONS * pdf_max)
    return np.append(scaled, 1.0)


def confidence_intervals(pdf_2d: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Contour levels enclosing 95%, 68%, and 40% of a normalized 2D PDF.

    Parameters
    ----------
    pdf_2d : np.ndarray
        Normalized 2D probability grid.
    dx : float, optional
        Cell width in x.
    dy : float, optional
        Cell height in y.

    Returns
    -------
    np.ndarray
        Contour levels plus a trailing sentinel level.
    """
    pdf_max = float(np.max(pdf_2d))
    if pdf_max <= 0.0 or not np.isfinite(pdf_max):
        return np.append(np.full(_TARGET_CONFIDENCE_FRACTIONS.size, np.nan), 1.0)

    thresholds = np.linspace(0.0, pdf_max, _CONFIDENCE_THRESHOLD_SAMPLES)
    integral = ((pdf_2d >= thresholds[:, None, None]) * pdf_2d).sum(axis=(1, 2)) * dx * dy
    unique_integral, unique_thresholds = _dedupe_increasing_integral_levels(
        integral,
        thresholds,
    )
    if (
        unique_integral.size < 2
        or unique_integral.max() - unique_integral.min() < _INTEGRAL_RANGE_EPS
    ):
        return _degenerate_confidence_levels(pdf_max)

    level_interpolator = interpolate.interp1d(
        unique_integral,
        unique_thresholds,
        bounds_error=False,
        fill_value=(unique_thresholds[0], unique_thresholds[-1]),
    )
    clipped_fractions = np.clip(
        _TARGET_CONFIDENCE_FRACTIONS,
        unique_integral.min(),
        unique_integral.max(),
    )
    return np.append(level_interpolator(clipped_fractions), 1.0)


def median_property_vs_mass(
    mass: np.ndarray,
    values: np.ndarray,
    mass_range: Tuple[float, float],
    n_bins: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binned median property versus mass.

    Parameters
    ----------
    mass : np.ndarray
        Halo masses (linear Msun/h).
    values : np.ndarray
        Property values.
    mass_range : tuple[float, float]
        ``(m_min, m_max)`` for binning in log10 mass.
    n_bins : int, optional
        Number of mass bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mass bin centers and median property per bin.
    """
    log_mass = np.log10(mass)
    mass_centers = (
        10 ** stats.binned_statistic(log_mass, log_mass, "mean", bins=n_bins, range=mass_range)[0]
    )
    medians = (
        10
        ** stats.binned_statistic(
            log_mass, np.log10(values), "median", bins=n_bins, range=mass_range
        )[0]
    )
    return mass_centers, medians
