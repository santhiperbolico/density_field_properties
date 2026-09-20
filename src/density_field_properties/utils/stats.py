"""Pure numeric helpers shared by validation and notebooks."""

from typing import Tuple

import numpy as np
from scipy import interpolate, stats


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
    sample_count = 20
    thresholds = np.linspace(0, pdf_2d.max(), sample_count)
    integral = ((pdf_2d >= thresholds[:, None, None]) * pdf_2d).sum(axis=(1, 2)) * dx * dy
    level_interpolator = interpolate.interp1d(integral, thresholds)
    return np.append(level_interpolator(np.array([0.95, 0.68, 0.40])), 1)


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
