"""Spherical binning of auto and cross power spectra for overdensity fields."""

from typing import Optional

import numpy as np
from numpy.fft import rfftn

from density_field_properties.density_field.fourrier_transformations import kgrid


def _delta_fourier_modes(delta: np.ndarray) -> np.ndarray:
    """
    Compute Fourier modes of a real-space overdensity field.

    Parameters
    ----------
    delta : np.ndarray
        Real-space overdensity on a cubic grid.

    Returns
    -------
    np.ndarray
        Complex Fourier modes from ``rfftn``, normalized by ``n_grid**3``.
    """
    n_grid = delta.shape[0]
    if delta.shape != (n_grid, n_grid, n_grid):
        raise ValueError("delta must be a cubic 3D array")
    return rfftn(delta) / n_grid**3


def _spherical_bin_edges(k_mag: np.ndarray, n_k_bins: int, k_max: Optional[float]) -> np.ndarray:
    """
    Build linear bin edges in ``|k|`` for non-zero modes.

    Parameters
    ----------
    k_mag : np.ndarray
        Wavenumber magnitude field.
    n_k_bins : int
        Number of spherical shells.
    k_max : Optional[float]
        Upper edge of the last bin; ``None`` uses the maximum ``|k|`` in ``k_mag``.

    Returns
    -------
    np.ndarray
        Bin edges with length ``n_k_bins + 1``.
    """
    positive_k = k_mag[k_mag > 0.0]
    if positive_k.size == 0:
        raise ValueError("No non-zero wavenumber modes available for binning")
    upper = float(k_max) if k_max is not None else float(positive_k.max())
    if upper <= 0.0:
        raise ValueError("k_max must be positive")
    return np.linspace(0.0, upper, n_k_bins + 1)


def _binned_mean(values: np.ndarray, sample_k: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Average ``values`` in spherical ``|k|`` bins.

    Parameters
    ----------
    values : np.ndarray
        Flattened values aligned with ``sample_k``.
    sample_k : np.ndarray
        Flattened wavenumber magnitudes.
    bin_edges : np.ndarray
        Monotonic bin edges.

    Returns
    -------
    np.ndarray
        Mean value per bin; empty bins are ``nan``.
    """
    bin_index = np.digitize(sample_k, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    means = np.full(n_bins, np.nan, dtype=np.float64)
    for index in range(n_bins):
        mask = bin_index == index
        if np.any(mask):
            means[index] = float(values[mask].mean())
    return means


def spherical_power_spectra(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    box_size: float,
    n_k_bins: int = 64,
    k_max: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """
    Compute auto and cross power spectra and their correlation ``r(k)``.

    For overdensity fields ``delta_a`` and ``delta_b`` on the same cubic grid,

    ``r(k) = P_ab(k) / sqrt(P_aa(k) * P_bb(k))``.

    Parameters
    ----------
    delta_a : np.ndarray
        First overdensity field.
    delta_b : np.ndarray
        Second overdensity field on the same grid.
    box_size : float
        Periodic box side length in Mpc/h.
    n_k_bins : int, optional
        Number of linear bins in ``|k|``.
    k_max : Optional[float], optional
        Maximum wavenumber for bin edges in ``h Mpc^-1``.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``k``, ``pk_a``, ``pk_b``, ``cross_pk``, and ``r_k``.
    """
    if delta_a.shape != delta_b.shape:
        raise ValueError("delta_a and delta_b must share the same shape")
    n_grid = delta_a.shape[0]
    volume = box_size**3

    delta_k_a = _delta_fourier_modes(delta_a)
    delta_k_b = _delta_fourier_modes(delta_b)
    kx, ky, kz = kgrid(n_grid, box_size)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    auto_a = np.abs(delta_k_a) ** 2 * volume
    auto_b = np.abs(delta_k_b) ** 2 * volume
    cross = np.real(delta_k_a * np.conj(delta_k_b)) * volume

    valid = k_mag > 0.0
    if k_max is not None:
        valid &= k_mag <= k_max

    bin_edges = _spherical_bin_edges(k_mag, n_k_bins, k_max)
    sample_k = k_mag[valid]
    pk_a = _binned_mean(auto_a[valid], sample_k, bin_edges)
    pk_b = _binned_mean(auto_b[valid], sample_k, bin_edges)
    cross_pk = _binned_mean(cross[valid], sample_k, bin_edges)

    denominator = np.sqrt(pk_a * pk_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_k = cross_pk / denominator
    invalid = (~np.isfinite(denominator)) | (denominator <= 0.0)
    r_k[invalid] = np.nan

    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "k": k_centers,
        "pk_a": pk_a,
        "pk_b": pk_b,
        "cross_pk": cross_pk,
        "r_k": r_k,
    }


def median_r_k_below_k(
    k: np.ndarray,
    r_k: np.ndarray,
    k_threshold: float,
) -> float:
    """
    Median ``r(k)`` for bins with ``k < k_threshold``.

    Parameters
    ----------
    k : np.ndarray
        Bin-center wavenumbers in ``h Mpc^-1``.
    r_k : np.ndarray
        Correlation coefficient per bin.
    k_threshold : float
        Upper ``k`` limit for the summary statistic.

    Returns
    -------
    float
        Median of finite ``r_k`` values in the selected bins, or ``nan``.
    """
    mask = (k < k_threshold) & np.isfinite(r_k)
    if not np.any(mask):
        return float("nan")
    return float(np.median(r_k[mask]))
