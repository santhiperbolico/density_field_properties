import numpy as np
from numpy.fft import fftfreq


def kgrid(n_grid: int, box_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the 3D Fourier-space wavevector grids for a real-valued field.

    Builds the arrays of wavenumbers (kx, ky, kz) corresponding to a real-to-complex
    FFT (`rfftn`) of a cubic field of size `box_size`. The resulting grids can be used
    to compute Fourier derivatives or to solve Poisson’s equation.

    Parameters
    ----------
    n_grid : int
        Number of grid cells per dimension.
    box_size : float
        Physical size of the simulation box.

    Returns
    -------
    kx3, ky3, kz3 : ndarray
        3D arrays with the Cartesian components of the wavevector for each Fourier mode.
    """
    kspace = 2 * np.pi * fftfreq(n_grid, d=box_size / n_grid)
    kx3, ky3, kz3 = np.meshgrid(kspace, kspace, kspace[: n_grid // 2 + 1], indexing="ij")
    return kx3, ky3, kz3
