"""Log-mass bin helpers for Haloscope fit and predict."""

import numpy as np


def default_mass_bin_edges(log_m200b_max: float) -> np.ndarray:
    """
    Default log10(M200b) bin edges for Haloscope fits.

    Parameters
    ----------
    log_m200b_max : float
        Upper edge taken from the training sample (log10 Msun/h).

    Returns
    -------
    np.ndarray
        Bin edges with length ``n_bins + 1``.
    """
    return np.array([10.0, 10.8, 11.6, 12.6, log_m200b_max])


def mask_mass_bin(log_mass: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Boolean mask for halos in one log10 mass bin ``[low, high)``.

    Parameters
    ----------
    log_mass : np.ndarray
        log10(M200b) per halo.
    low : float
        Lower bin edge in log10.
    high : float
        Upper bin edge in log10.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    return (log_mass >= low) & (log_mass < high)
