"""Mass calibration between FastPM and high-resolution training catalogs."""

import numpy as np
from scipy import stats


def abundance_match_mass(mass_target: np.ndarray, mass_reference: np.ndarray) -> np.ndarray:
    """
    Monotonic abundance matching of ``mass_target`` onto the mass function of ``mass_reference``.

    Parameters
    ----------
    mass_target : np.ndarray
        Masses to remap (e.g. FastPM M200b).
    mass_reference : np.ndarray
        Reference mass sample (e.g. UNIT SIM M200b).

    Returns
    -------
    np.ndarray
        Masses on the reference cumulative distribution, same shape as ``mass_target``.
    """
    quantile_target = stats.rankdata(mass_target) / (len(mass_target) + 1)
    reference_sorted = np.sort(mass_reference)
    quantile_reference = (np.arange(len(reference_sorted)) + 1) / (len(reference_sorted) + 1)
    return np.interp(quantile_target, quantile_reference, reference_sorted)
