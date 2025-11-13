import numpy as np


def gaussian_filter(k: np.ndarray, r_scale: float) -> np.ndarray:
    """
    Compute a Gaussian filter.

    This function calculates a Gaussian filter based on the input frequency domain array `k`
    and a given scale parameter `r_scale`. The resulting Gaussian filter is useful for smooth
    filtering in signal or image processing, with values computed using the formula
    `exp(-|k|**2 * r_scale**2 / 2)`.

    Parameters
    ----------
    k : np.ndarray
        Array representing frequencies in the spatial domain.
    r_scale : float
        Scaling factor that determines the spread of the Gaussian in the frequency domain.

    Returns
    -------
    np.ndarray
        Computed Gaussian filter with the same shape as `k`.
    """
    k_2 = np.power(k, 2).sum()
    return np.exp(-k_2 * r_scale**2 / 2)
