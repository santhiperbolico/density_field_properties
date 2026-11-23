"""Tests for HALOSCOPE halo-by-halo linear bias (Ramakrishnan / analyse_sim convention)."""

import numpy as np
import pytest
from numpy.fft import irfftn, rfftn

from density_field_properties.validation.haloscope_halo_bias import (
    DEFAULT_K_BINS_H_MPC,
    _cic_window_square,
    _deltak_halo_cic,
    _k3_weighted_bias_batch,
    _sparse_k_axes,
    haloscope_halo_by_halo_bias,
    haloscope_ratio_of_means_bias,
)

SMALL_POWER = 1e-30


def _legacy_ifft_estimator(
    positions_mpc_h: np.ndarray,
    delta_field: np.ndarray,
    boxsize_mpc_h: float,
    k_max_h_mpc: float,
) -> np.ndarray:
    """
    Previous estimator: IFFT of ``delta_k / P_mm`` sampled with trilinear interpolation.

    Kept only as a regression contrast; not the HALOSCOPE target.
    """
    n_grid = delta_field.shape[0]
    volume = boxsize_mpc_h**3
    delta_k = rfftn(delta_field) / n_grid**3
    kx, ky, kz = _sparse_k_axes(n_grid, boxsize_mpc_h)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    low_k = (k_mag > 0.0) & (k_mag <= k_max_h_mpc)
    power = np.abs(delta_k) ** 2 * volume
    weighted = np.zeros_like(delta_k, dtype=np.complex128)
    weighted[low_k] = delta_k[low_k] / np.maximum(power[low_k], SMALL_POWER)
    bias_field = irfftn(weighted, s=(n_grid, n_grid, n_grid)).real * n_grid**3
    cell = boxsize_mpc_h / n_grid
    wrapped = np.mod(positions_mpc_h, boxsize_mpc_h)
    grid_coord = wrapped / cell
    i0 = np.floor(grid_coord).astype(int) % n_grid
    frac = grid_coord - np.floor(grid_coord)
    samples = np.zeros(len(positions_mpc_h), dtype=np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                weight = (
                    (frac[:, 0] if di else 1.0 - frac[:, 0])
                    * (frac[:, 1] if dj else 1.0 - frac[:, 1])
                    * (frac[:, 2] if dk else 1.0 - frac[:, 2])
                )
                ix = (i0[:, 0] + di) % n_grid
                iy = (i0[:, 1] + dj) % n_grid
                iz = (i0[:, 2] + dk) % n_grid
                samples += weight * bias_field[ix, iy, iz]
    return samples


def _reference_k3_bias(
    positions_mpc_h: np.ndarray,
    delta_field: np.ndarray,
    boxsize_mpc_h: float,
    bin_edges: np.ndarray,
    deconvolve_cic: bool = True,
) -> np.ndarray:
    """
    Reference implementation mirroring ``analyse_simulation.halo_by_halo_bias`` (b1 output).
    """
    n_grid = delta_field.shape[0]
    volume = boxsize_mpc_h**3
    matter_k = rfftn(delta_field) / n_grid**3
    kx, ky, kz = _sparse_k_axes(n_grid, boxsize_mpc_h)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    if deconvolve_cic:
        matter_k = matter_k / _cic_window_square(kx, ky, kz, n_grid, boxsize_mpc_h)

    shell = (k_mag >= bin_edges[0]) & (k_mag <= bin_edges[-1]) & (k_mag > 0.0)
    flat_mask = shell.ravel()
    k_flat = k_mag.ravel()[flat_mask]
    matter_flat = matter_k.ravel()[flat_mask]
    p_mm_flat = (np.abs(matter_flat) ** 2) * volume

    n_halos = positions_mpc_h.shape[0]
    bias = np.zeros(n_halos, dtype=np.float64)
    for halo_index in range(n_halos):
        halo_k = _deltak_halo_cic(
            positions_mpc_h[halo_index : halo_index + 1],
            n_grid,
            boxsize_mpc_h,
            kx,
            ky,
            kz,
        )
        p_hm = halo_k * np.conj(matter_k) * volume
        p_hm_flat = p_hm.reshape(1, -1)[:, flat_mask].real
        bias[halo_index] = _k3_weighted_bias_batch(p_hm_flat, p_mm_flat, k_flat, bin_edges)[0]
    return bias


def _plane_wave_delta(n_grid: int, boxsize: float, k0: float, amplitude: float) -> np.ndarray:
    """
    Build a real single-mode overdensity on the FFT grid.

    Parameters
    ----------
    n_grid : int
        Grid size.
    boxsize : float
        Box side length.
    k0 : float
        Wavenumber along x in h Mpc^-1.
    amplitude : float
        Mode amplitude.

    Returns
    -------
    np.ndarray
        Real overdensity field.
    """
    cell = boxsize / n_grid
    x = np.arange(n_grid, dtype=np.float64) * cell
    return amplitude * np.cos(k0 * x)[:, np.newaxis, np.newaxis]


@pytest.mark.parametrize("n_grid", [16, 24])
def test_haloscope_k3_matches_reference_implementation(n_grid):
    """Primary estimator must match a one-halo-at-a-time reference loop."""
    rng = np.random.default_rng(3)
    boxsize = 100.0
    delta_field = rng.normal(scale=0.05, size=(n_grid, n_grid, n_grid))
    positions = rng.uniform(0.0, boxsize, size=(12, 3))
    bin_edges = DEFAULT_K_BINS_H_MPC

    expected = _reference_k3_bias(positions, delta_field, boxsize, bin_edges)
    result = haloscope_halo_by_halo_bias(
        positions,
        delta_field,
        boxsize,
        k_max_h_mpc=bin_edges[-1],
        log_k_bin_edges=bin_edges,
        halo_batch_size=4,
    )
    assert np.allclose(result, expected, rtol=0.0, atol=1e-10)


def test_haloscope_uniform_delta_near_zero():
    """A spatially uniform matter field yields zero linear bias."""
    n_grid = 16
    boxsize = 50.0
    delta_field = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    positions = np.array([[10.0, 20.0, 30.0], [40.0, 41.0, 42.0]])
    bias = haloscope_halo_by_halo_bias(positions, delta_field, boxsize, k_max_h_mpc=0.2)
    assert np.allclose(bias, 0.0, atol=1e-10)


def test_haloscope_ratio_of_means_finite_on_single_mode_field():
    """
    A single-mode overdensity must yield finite bias values below ``k_max``.
    """
    n_grid = 32
    boxsize = 64.0
    k0 = 2.0 * np.pi / boxsize
    amplitude = 0.12
    delta_field = _plane_wave_delta(n_grid, boxsize, k0, amplitude)
    positions = np.array([[8.0, 8.0, 8.0], [30.0, 12.0, 20.0]])
    k_max = 3.0 * k0
    bias = haloscope_ratio_of_means_bias(
        positions,
        delta_field,
        boxsize,
        k_max_h_mpc=k_max,
        deconvolve_cic=True,
    )
    assert np.all(np.isfinite(bias))


def test_haloscope_k3_differs_from_legacy_ifft_estimator():
    """The HALOSCOPE estimator must not coincide with the old IFFT shortcut."""
    rng = np.random.default_rng(7)
    n_grid = 20
    boxsize = 80.0
    delta_field = rng.normal(scale=0.08, size=(n_grid, n_grid, n_grid))
    positions = rng.uniform(0.0, boxsize, size=(8, 3))
    k_max = 0.2

    haloscope = haloscope_halo_by_halo_bias(
        positions, delta_field, boxsize, k_max_h_mpc=k_max, halo_batch_size=8
    )
    legacy = _legacy_ifft_estimator(positions, delta_field, boxsize, k_max)
    assert not np.allclose(haloscope, legacy, rtol=0.0, atol=1e-3)


def test_cic_deconvolution_changes_bias_when_grid_is_sharp():
    """CIC deconvolution must change the estimate on a non-trivial field."""
    rng = np.random.default_rng(11)
    n_grid = 16
    boxsize = 32.0
    delta_field = rng.normal(scale=0.1, size=(n_grid, n_grid, n_grid))
    positions = rng.uniform(0.0, boxsize, size=(5, 3))
    bias_on = haloscope_halo_by_halo_bias(
        positions, delta_field, boxsize, k_max_h_mpc=0.25, deconvolve_cic=True
    )
    bias_off = haloscope_halo_by_halo_bias(
        positions, delta_field, boxsize, k_max_h_mpc=0.25, deconvolve_cic=False
    )
    assert not np.allclose(bias_on, bias_off, atol=1e-4)
