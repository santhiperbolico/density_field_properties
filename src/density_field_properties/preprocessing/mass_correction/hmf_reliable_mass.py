"""HMF-based lower mass bound for LR mass correction."""

import numpy as np

from density_field_properties.preprocessing.mass_correction.config import (
    DEFAULT_HMF_ELBOW_BIN_COUNT,
    DEFAULT_HMF_ELBOW_MIN_COUNT,
)


def estimate_fastpm_hmf_elbow_min_mass_msun_h(
    lr_masses: np.ndarray,
    volume_mpc_h3: float,
    min_halo_count_per_bin: int = DEFAULT_HMF_ELBOW_MIN_COUNT,
    bin_count: int = DEFAULT_HMF_ELBOW_BIN_COUNT,
) -> float | None:
    """
    Estimate the FastPM completeness knee from the peak of the LR HMF.

    Uses the bin center where ``log10(n/V)`` is maximal among bins with at
    least ``min_halo_count_per_bin`` halos. That peak is the classic elbow
    before the power-law decline at higher mass.

    Parameters
    ----------
    lr_masses : np.ndarray
        Raw LR (FastPM) masses in Msun/h.
    volume_mpc_h3 : float
        Simulation volume in (Mpc/h)^3.
    min_halo_count_per_bin : int, optional
        Minimum count in a bin for it to qualify as the knee peak.
    bin_count : int, optional
        Number of log-spaced mass bins.

    Returns
    -------
    float or None
        Knee mass in Msun/h, or ``None`` when LR masses are missing.
    """
    lr_positive = lr_masses[np.isfinite(lr_masses) & (lr_masses > 0.0)]
    if lr_positive.size == 0:
        return None

    log_min = float(np.log10(lr_positive.min()))
    log_max = float(np.log10(lr_positive.max()))
    if log_min >= log_max:
        raise ValueError("LR masses are constant; cannot estimate knee mass.")

    bin_edges = np.logspace(log_min, log_max, bin_count + 1)
    counts, _ = np.histogram(lr_positive, bins=bin_edges)
    density = counts.astype(np.float64) / volume_mpc_h3
    qualified = counts >= min_halo_count_per_bin
    if not np.any(qualified):
        raise ValueError(
            f"No bins with at least {min_halo_count_per_bin} halos; cannot estimate knee mass."
        )

    safe_density = np.where(density > 0.0, density, np.nan)
    with np.errstate(divide="ignore"):
        log_density = np.log10(safe_density)

    scored = np.where(qualified, log_density, -np.inf)
    peak_index = int(np.argmax(scored))
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return float(bin_centers[peak_index])
