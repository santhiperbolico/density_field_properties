from typing import Optional, Self

import numpy as np
from numpy.fft import irfftn, rfftn

from density_field_properties.density_field.fourrier_transformations import kgrid


def sinc(x: np.ndarray) -> np.ndarray:
    """
    Compute the normalized sinc function element-wise for the input array.

    The normalized sinc function is defined as sinc(x) = sin(x) / x for x != 0,
    and sinc(0) = 1. This function computes the values for each element in a
    given input array.

    Parameters
    ----------
    x : np.ndarray
        Input array containing numerical values. Each element of the array is
        processed to compute its normalized sinc function value.

    Returns
    -------
    np.ndarray
        An array of the same shape as the input array, where each element contains
        the computed value of the normalized sinc function for the corresponding
        input value.
    """
    out = np.ones_like(x)
    m = x != 0
    out[m] = np.sin(x[m]) / x[m]
    return out


class TidalTensor:
    def __init__(
        self,
        t_xx: np.ndarray,
        t_xy: np.ndarray,
        t_xz: np.ndarray,
        t_yy: np.ndarray,
        t_yz: np.ndarray,
        t_zz: np.ndarray,
        gaussian_scale: Optional[float | int] = None,
    ):
        self.tidal_tensor = {
            (0, 0): t_xx,
            (0, 1): t_xy,
            (0, 2): t_xz,
            (1, 0): t_xy,
            (1, 1): t_yy,
            (1, 2): t_yz,
            (2, 0): t_xz,
            (2, 1): t_yz,
            (2, 2): t_zz,
        }
        self.gaussian_scale = gaussian_scale

    @classmethod
    def tidal_tensor_fft_from_delta(
        cls,
        delta: np.ndarray,
        box_size: float,
        gaussian_scale: Optional[float | int] = None,
        deconvolve_cic: bool = False,
    ) -> Self:
        """
        Computes the tidal tensor in Fourier space from a given density contrast field.

        The tidal tensor is derived from the gravitational potential tensor by taking
        its second derivatives. This function uses fast Fourier transforms (FFTs) to
        efficiently perform the calculations in Fourier space, and then converts
        the results back to real space. Optionally, corrections for Cloud-In-Cell (CIC)
        deconvolution can be applied alongside smoothing with a Gaussian kernel.

        Parameters
        ----------
        delta : np.ndarray
            The density contrast field, a 3D array representing the spatial structure
            of density fluctuations.
        box_size : float
            The physical size of the simulation box in the same units as the spatial
            dimensions of `delta`.
        gaussian_scale : Optional[float | int]=None
            The Gaussian smoothing scale in the same units as the spatial dimensions of
            `delta`. If not specified or set to None, smoothing is omitted.
        deconvolve_cic : bool, default=False
            Whether to apply a Cloud-In-Cell (CIC) kernel deconvolution to correct
            for grid effects in the density field.

        Returns
        -------
        TidalTensor
            TidalTensor object containing the tidal tensor in real space.
        """
        n_grid = delta.shape[0]
        delt_k = rfftn(delta)
        kx, ky, kz = kgrid(n_grid, box_size)
        k2 = kx**2 + ky**2 + kz**2

        if deconvolve_cic:
            d = box_size / n_grid
            w = sinc(0.5 * kx * d) * sinc(0.5 * ky * d) * sinc(0.5 * kz * d)
            delt_k /= w**2 + 1e-7

        if gaussian_scale is not None and gaussian_scale > 0:
            delt_k *= np.exp(-0.5 * k2 * gaussian_scale**2)

        denom = np.where(k2 > 0, k2, np.inf)

        # TODO: In the Sujatha's code appears a GridSize**3 factor in each term. Why?
        tidal_tensor = cls(
            t_xx=irfftn((kx * kx / denom) * delt_k, s=delta.shape),
            t_yy=irfftn((ky * ky / denom) * delt_k, s=delta.shape),
            t_zz=irfftn((kz * kz / denom) * delt_k, s=delta.shape),
            t_xy=irfftn((kx * ky / denom) * delt_k, s=delta.shape),
            t_xz=irfftn((kx * kz / denom) * delt_k, s=delta.shape),
            t_yz=irfftn((ky * kz / denom) * delt_k, s=delta.shape),
            gaussian_scale=gaussian_scale,
        )

        return tidal_tensor

    def get_tidal_tensor(
        self, cell_x: np.ndarray, cell_y: np.ndarray, cell_z: np.ndarray
    ) -> np.ndarray:
        """
        Computes the tidal tensor for given grid cell indices.

        This function calculates a 3x3 tidal tensor for each specified grid point,
        where the tidal tensor elements are determined by precomputed data. The
        inputs must be NumPy arrays of the same shape, representing the indices of
        the grid along the x, y, and z dimensions for which the tidal tensor is to
        be computed.

        Parameters
        ----------
        cell_x : np.ndarray
            The x-coordinates of the grid cell indices.
        cell_y : np.ndarray
            The y-coordinates of the grid cell indices.
        cell_z : np.ndarray
            The z-coordinates of the grid cell indices.

        Returns
        -------
        np.ndarray
            A 3D NumPy array representing the tidal tensor for the specified grid
            cells. Its shape will be `(N, 3, 3)`, where N is the total number of
            grid points specified by the input arrays.
        """

        if cell_x.shape != cell_y.shape:
            raise ValueError("cell_x and cell_y must have the same shape")

        if cell_x.shape != cell_z.shape:
            raise ValueError("cell_x and cell_z must have the same shape")

        if cell_y.shape != cell_z.shape:
            raise ValueError("cell_y and cell_z must have the same shape")

        tidal_tensor = np.zeros([np.size(cell_x), 3, 3], dtype="f4")
        for key, values in self.tidal_tensor.items():
            tidal_tensor[:, key[0], key[1]] = values[cell_x, cell_y, cell_z]

        return tidal_tensor

    def eigenvalues(
        self,
        cell_x: np.ndarray,
        cell_y: np.ndarray,
        cell_z: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the eigenvalues of the tidal tensor at the specified grid cell.

        This function calculates the eigenvalues of the tidal tensor constructed
        at the spatial location defined by the input cell indices. The tidal tensor
        provides insights into the gravitational interactions at that point in the grid.

        Parameters
        ----------
        cell_x : np.ndarray
            The x-coordinates of the grid cells where the tidal tensor is evaluated.
        cell_y : np.ndarray
            The y-coordinates of the grid cells where the tidal tensor is evaluated.
        cell_z : np.ndarray
            The z-coordinates of the grid cells where the tidal tensor is evaluated.

        Returns
        -------
        np.ndarray
            An array of eigenvalues corresponding to the tidal tensor for each input grid cell.
        """
        tidal_tensor = self.get_tidal_tensor(cell_x, cell_y, cell_z)
        eigval, _ = np.linalg.eigh(tidal_tensor)
        return eigval
