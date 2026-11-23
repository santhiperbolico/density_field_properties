"""HALOSCOPE halo-by-halo linear bias (Ramakrishnan et al.; UAM analyse_sim convention)."""

import numpy as np
from numpy.fft import rfftn

SMALL_POWER = 1e-30
DEFAULT_K_BINS_H_MPC = np.array([0.02, 0.04, 0.06, 0.08, 0.10])


def _sparse_k_axes(n_grid: int, boxsize_mpc_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build broadcast-ready wavevector axes matching ``rfftn`` layout.

    Parameters
    ----------
    n_grid : int
        Cells per dimension.
    boxsize_mpc_h : float
        Box side length in Mpc/h.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``kx``, ``ky``, ``kz`` with shapes ``(n,1,1)``, ``(1,n,1)``, ``(1,1,n_r)``.
    """
    cell = boxsize_mpc_h / n_grid
    k1d = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=cell)
    kz1d = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=cell)
    return (
        k1d.reshape(-1, 1, 1),
        k1d.reshape(1, -1, 1),
        kz1d.reshape(1, 1, -1),
    )


def _cic_window_square(
    kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, n_grid: int, boxsize: float
) -> np.ndarray:
    """
    Squared CIC transfer function used for Fourier-space deconvolution.

    Parameters
    ----------
    kx, ky, kz : np.ndarray
        Sparse wavevector components.
    n_grid : int
        Grid size.
    boxsize : float
        Box side length in Mpc/h.

    Returns
    -------
    np.ndarray
        Window factor ``W_CIC^2`` with the same broadcast shape as ``kx``.
    """
    factor = boxsize / (2.0 * np.pi * n_grid)
    wx = np.sinc(kx * factor)
    wy = np.sinc(ky * factor)
    wz = np.sinc(kz * factor)
    return (wx * wy * wz) ** 2


def _deltak_halo_cic(
    positions_mpc_h: np.ndarray,
    n_grid: int,
    boxsize_mpc_h: float,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
) -> np.ndarray:
    """
    Fourier transform of a CIC cloud at each halo position.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    n_grid : int
        Grid cells per dimension.
    boxsize_mpc_h : float
        Periodic box side length.
    kx, ky, kz : np.ndarray
        Sparse wavevector axes from ``_sparse_k_axes``.

    Returns
    -------
    np.ndarray
        Complex modes with shape ``(N, n_grid, n_grid, n_grid//2 + 1)``.
    """
    dx = boxsize_mpc_h / n_grid
    x = positions_mpc_h[:, 0]
    y = positions_mpc_h[:, 1]
    z = positions_mpc_h[:, 2]
    i_cell = np.mod(np.floor(x / dx).astype(int), n_grid)
    j_cell = np.mod(np.floor(y / dx).astype(int), n_grid)
    k_cell = np.mod(np.floor(z / dx).astype(int), n_grid)
    dx_frac = x - i_cell * dx
    dy_frac = y - j_cell * dx
    dz_frac = z - k_cell * dx

    n_halos = positions_mpc_h.shape[0]
    deltak = np.zeros((n_halos, n_grid, n_grid, n_grid // 2 + 1), dtype=np.complex128)
    for corner in range(8):
        di = corner % 2
        dj = (corner // 2) % 2
        dk = corner // 4
        weight = (
            (dx_frac if di else dx - dx_frac)
            * (dy_frac if dj else dx - dy_frac)
            * (dz_frac if dk else dx - dz_frac)
        ) / dx**3
        corner_x = np.mod(i_cell + di, n_grid) * dx
        corner_y = np.mod(j_cell + dj, n_grid) * dx
        corner_z = np.mod(k_cell + dk, n_grid) * dx
        phase = np.exp(
            -1j
            * (
                kx * corner_x.reshape(-1, 1, 1, 1)
                + ky * corner_y.reshape(-1, 1, 1, 1)
                + kz * corner_z.reshape(-1, 1, 1, 1)
            )
        )
        deltak += weight.reshape(-1, 1, 1, 1) * phase
    return deltak


def _log_k_bin_edges(k_max_h_mpc: float) -> np.ndarray:
    """
    Return default logarithmic k bin edges up to ``k_max``.

    Parameters
    ----------
    k_max_h_mpc : float
        Maximum wavenumber in h Mpc^-1.

    Returns
    -------
    np.ndarray
        Monotonic bin edges; scaled from HALOSCOPE defaults when ``k_max`` differs.
    """
    if np.isclose(k_max_h_mpc, DEFAULT_K_BINS_H_MPC[-1]):
        return DEFAULT_K_BINS_H_MPC.copy()
    scale = k_max_h_mpc / DEFAULT_K_BINS_H_MPC[-1]
    return DEFAULT_K_BINS_H_MPC * scale


def _k3_weighted_bias_batch(
    p_hm: np.ndarray,
    p_mm: np.ndarray,
    k_mag: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    k^3-weighted average of in-bin ``<P_hm>/<P_mm>`` ratios (Paranjape eq. 8).

    Parameters
    ----------
    p_hm : np.ndarray
        Cross power per halo and mode, shape ``(batch, n_modes_flat)``.
    p_mm : np.ndarray
        Matter auto power per mode, shape ``(n_modes_flat,)``.
    k_mag : np.ndarray
        Mode magnitudes, shape ``(n_modes_flat,)``.
    bin_edges : np.ndarray
        Low-k bin edges in h Mpc^-1.

    Returns
    -------
    np.ndarray
        Bias estimate per halo in the batch.
    """
    n_bins = len(bin_edges) - 1
    k_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    weights = k_centers**3
    ratios = []
    used_weights = []
    for bin_index in range(n_bins):
        in_bin = (k_mag >= bin_edges[bin_index]) & (k_mag < bin_edges[bin_index + 1])
        if not np.any(in_bin):
            continue
        phm_mean = p_hm[:, in_bin].mean(axis=1)
        pmm_mean = float(p_mm[in_bin].mean())
        ratios.append(phm_mean / (pmm_mean + SMALL_POWER))
        used_weights.append(weights[bin_index])
    if not ratios:
        raise ValueError("No Fourier modes inside log-k bins; check n_grid and k_max.")
    ratio_stack = np.stack(ratios, axis=1)
    weight_array = np.asarray(used_weights, dtype=np.float64)
    return (ratio_stack * weight_array).sum(axis=1) / weight_array.sum()


def haloscope_halo_by_halo_bias(
    positions_mpc_h: np.ndarray,
    delta_field: np.ndarray,
    boxsize_mpc_h: float,
    k_max_h_mpc: float = 0.1,
    log_k_bin_edges: np.ndarray | None = None,
    deconvolve_cic: bool = True,
    halo_batch_size: int = 64,
) -> np.ndarray:
    """
    HALOSCOPE low-k halo-by-halo linear bias (Ramakrishnan et al. 2025, eq. 3).

    Matches the UAM ``analyse_simulation.halo_by_halo_bias`` primary estimator:
    in each log-k shell average ``P_hm`` and ``P_mm``, form the ratio, then
    combine shells with weights proportional to k^3.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    delta_field : np.ndarray
        Real-space matter overdensity on a cubic grid from a CIC field.
    boxsize_mpc_h : float
        Box side length in Mpc/h.
    k_max_h_mpc : float, optional
        Maximum wavenumber included (default 0.1).
    log_k_bin_edges : np.ndarray or None, optional
        Bin edges in h Mpc^-1; defaults to scaled ``DEFAULT_K_BINS_H_MPC``.
    deconvolve_cic : bool, optional
        Apply CIC deconvolution to the matter field in Fourier space.
    halo_batch_size : int, optional
        Halos processed per batch to limit memory.

    Returns
    -------
    np.ndarray
        Per-halo linear bias, shape ``(N,)``.
    """
    n_grid = delta_field.shape[0]
    if delta_field.shape != (n_grid, n_grid, n_grid):
        raise ValueError("delta_field must be a cubic 3D array")

    bin_edges = log_k_bin_edges
    if bin_edges is None:
        bin_edges = _log_k_bin_edges(k_max_h_mpc)
    if bin_edges[-1] > k_max_h_mpc + 1e-12:
        raise ValueError("log_k_bin_edges must not extend beyond k_max_h_mpc")

    volume = boxsize_mpc_h**3
    matter_k = rfftn(delta_field) / n_grid**3
    kx, ky, kz = _sparse_k_axes(n_grid, boxsize_mpc_h)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    if deconvolve_cic:
        window = _cic_window_square(kx, ky, kz, n_grid, boxsize_mpc_h)
        matter_k = matter_k / window

    k_mask_shell = (k_mag >= bin_edges[0]) & (k_mag <= bin_edges[-1]) & (k_mag > 0.0)
    if not np.any(k_mask_shell):
        raise ValueError("No modes in the requested k range; increase n_grid or k_max.")

    flat_mask = k_mask_shell.ravel()
    k_flat = k_mag.ravel()[flat_mask]
    matter_flat = matter_k.ravel()[flat_mask]
    p_mm_flat = (np.abs(matter_flat) ** 2) * volume

    n_halos = positions_mpc_h.shape[0]
    bias = np.zeros(n_halos, dtype=np.float64)
    for start in range(0, n_halos, halo_batch_size):
        end = min(start + halo_batch_size, n_halos)
        halo_k = _deltak_halo_cic(
            positions_mpc_h[start:end],
            n_grid,
            boxsize_mpc_h,
            kx,
            ky,
            kz,
        )
        p_hm = halo_k * np.conj(matter_k) * volume
        p_hm_flat = p_hm.reshape(end - start, -1)[:, flat_mask].real
        bias[start:end] = _k3_weighted_bias_batch(p_hm_flat, p_mm_flat, k_flat, bin_edges)
    return bias


def haloscope_ratio_of_means_bias(
    positions_mpc_h: np.ndarray,
    delta_field: np.ndarray,
    boxsize_mpc_h: float,
    k_max_h_mpc: float = 0.1,
    deconvolve_cic: bool = True,
    halo_batch_size: int = 64,
) -> np.ndarray:
    """
    Auxiliary estimator ``mean(P_hm) / mean(P_mm)`` over all modes below ``k_max``.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    delta_field : np.ndarray
        Matter overdensity grid.
    boxsize_mpc_h : float
        Box size in Mpc/h.
    k_max_h_mpc : float, optional
        Maximum wavenumber in h Mpc^-1.
    deconvolve_cic : bool, optional
        CIC deconvolution on the matter field.
    halo_batch_size : int, optional
        Batch size for halo Fourier transforms.

    Returns
    -------
    np.ndarray
        Per-halo bias, shape ``(N,)``.
    """
    n_grid = delta_field.shape[0]
    volume = boxsize_mpc_h**3
    matter_k = rfftn(delta_field) / n_grid**3
    kx, ky, kz = _sparse_k_axes(n_grid, boxsize_mpc_h)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    if deconvolve_cic:
        matter_k = matter_k / _cic_window_square(kx, ky, kz, n_grid, boxsize_mpc_h)

    k_mask = (k_mag > 0.0) & (k_mag <= k_max_h_mpc)
    flat_mask = k_mask.ravel()
    matter_flat = matter_k.ravel()[flat_mask]
    p_mm_mean = float((np.abs(matter_flat) ** 2).mean() * volume)

    n_halos = positions_mpc_h.shape[0]
    bias = np.zeros(n_halos, dtype=np.float64)
    for start in range(0, n_halos, halo_batch_size):
        end = min(start + halo_batch_size, n_halos)
        halo_k = _deltak_halo_cic(positions_mpc_h[start:end], n_grid, boxsize_mpc_h, kx, ky, kz)
        p_hm = halo_k * np.conj(matter_k) * volume
        phm_mean = p_hm.reshape(end - start, -1)[:, flat_mask].real.mean(axis=1)
        bias[start:end] = phm_mean / (p_mm_mean + SMALL_POWER)
    return bias
