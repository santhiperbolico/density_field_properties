"""Log-mass bin helpers for Haloscope fit and predict."""

import numpy as np

DEFAULT_LOG_MASS_BIN_MIN = 11.5
DEFAULT_LOG_MASS_N_EDGES = 10


def default_mass_bin_edges(log_m200b_max: float) -> np.ndarray:
    """
    Default log10(M200b) bin edges for Haloscope fits and assembly-bias panels.

    Nine equal-width bins in log10 from ``DEFAULT_LOG_MASS_BIN_MIN`` to the
    sample maximum, aligned with the assembly-bias diagnostic in
    Ramakrishnan et al. (2025).

    Parameters
    ----------
    log_m200b_max : float
        Upper edge taken from the training sample (log10 Msun/h).

    Returns
    -------
    np.ndarray
        Bin edges with length ``DEFAULT_LOG_MASS_N_EDGES``.
    """
    return np.linspace(
        DEFAULT_LOG_MASS_BIN_MIN,
        log_m200b_max,
        DEFAULT_LOG_MASS_N_EDGES,
    )


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
