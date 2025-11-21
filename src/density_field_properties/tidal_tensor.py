import logging
import os
from typing import Optional, Self

import h5py
import numpy as np
from numpy.fft import irfftn, rfftn

from density_field_properties.density_field.fourrier_transformations import kgrid

NAME_HD5 = "tidaltensor"
TIDAL_TENSOR_PATH = "tidal_tensor"
GAUSSIAN_SCALE_DEFAULT = "none"


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


def tidal_tensor_component_calculation(
    delta: np.ndarray,
    box_size: float,
    path: str,
    component: tuple[int, int],
    gaussian_scale: Optional[float | int] = None,
) -> str:
    """
    Calculates a specific component of the tidal tensor from input density field data
    and saves the resulting tensor component to an HDF5 file. This function computes
    the tidal tensor component by transforming the density field into Fourier space,
    applying relevant operations, and then transforming back to real space.

    Parameters
    ----------
    delta : np.ndarray
        3D density field data array represented as a numpy array.
    box_size : float
        The size of the box containing the 3D density field in physical space.
    path : str
        The directory path where the resulting HDF5 file will be saved.
    component : tuple[int, int]
        A tuple specifying the indices representing the component of the tidal
        tensor to calculate (e.g., (0, 1) for T_01).
    gaussian_scale : float or int, optional
        The scale factor for a Gaussian smoothing operation applied in Fourier
        space. If None or non-positive, Gaussian smoothing is not applied.

    Returns
    -------
    str
        The path to the output HDF5 file where the tidal tensor component is saved.
    """
    n_grid = delta.shape[0]
    grid_shape = (n_grid, n_grid, n_grid)
    delt_k = rfftn(delta)
    kgrid_components = list(kgrid(n_grid, box_size))
    kmodule = kgrid_components[0] ** 2 + kgrid_components[1] ** 2 + kgrid_components[2] ** 2
    k1, k2 = kgrid_components[component[0]], kgrid_components[component[1]]

    del delta
    del kgrid_components

    if gaussian_scale is not None and gaussian_scale > 0:
        delt_k *= np.exp(-0.5 * kmodule * gaussian_scale**2)

    f_deltak = k1 * k2
    del k1, k2
    f_deltak = f_deltak / np.where(kmodule > 0, kmodule, np.inf)
    del kmodule
    f_deltak = f_deltak * delt_k
    del delt_k

    # In the Sujatha's code appears a GridSize**3 factor in each term.
    # The reason is that she's divided the delt_k by GridSize**3.
    t_component = irfftn(f_deltak, s=grid_shape)
    del f_deltak

    if gaussian_scale is None:
        gaussian_scale = GAUSSIAN_SCALE_DEFAULT

    output_file = os.path.join(
        path, f"TidalTensor_{component[0] + 3 * component[1]}_{gaussian_scale}.h5"
    )
    with h5py.File(output_file, "w") as f:
        f.create_dataset(NAME_HD5, data=t_component, dtype="float32")

    del t_component

    return output_file


class TidalTensor:
    def __init__(
        self,
        t_xx: str,
        t_xy: str,
        t_xz: str,
        t_yy: str,
        t_yz: str,
        t_zz: str,
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
    def from_folder(cls, path: str, gaussian_scale: Optional[float | int] = None) -> Self:
        """
        Create an instance of the class by loading tidal tensor components from a folder.

        The method searches for files corresponding to specific tidal tensor components
        in the provided folder and constructs an instance of the class using these
        components. If a file for a required component is not found, a
        FileNotFoundError is raised.

        Parameters
        ----------
        path : str
            Path to the folder containing the tidal tensor component files.
        gaussian_scale : Optional[float | int]=None
            The Gaussian scale used in the file names. If not provided, defaults
            to a predefined value (`GAUSSIAN_SCALE_DEFAULT`).

        Returns
        -------
        TidalTensor
            An instance of the class, initialized with the loaded tidal tensor
            components.

        Raises
        ------
        FileNotFoundError
            If any of the required tidal tensor component files are missing in the
            specified path.
        """
        gaussian_scale_name = gaussian_scale
        if gaussian_scale is None:
            gaussian_scale_name = GAUSSIAN_SCALE_DEFAULT

        components_list = [0, 3, 4, 6, 7, 8]
        tidal_params = ["t_xx", "t_xy", "t_xz", "t_yy", "t_yz", "t_zz"]
        tidal_params = dict(zip(components_list, tidal_params))
        tidal_files = {}
        for component in components_list:
            file_name = f"TidalTensor_{component}_{gaussian_scale_name}.h5"
            if os.path.isfile(os.path.join(path, file_name)):
                tidal_files[tidal_params.get(component)] = os.path.join(path, file_name)
            else:
                raise FileNotFoundError(f"File {file_name} not found in {path}")

        tidal_tensor = cls(gaussian_scale=gaussian_scale, **tidal_files)
        return tidal_tensor

    @classmethod
    def from_delta(
        cls,
        delta: np.ndarray,
        box_size: float,
        path: str,
        gaussian_scale: Optional[float | int] = None,
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
        path : str
            The directory path where the resulting HDF5 file will be saved.
        gaussian_scale : Optional[float | int]=None
            The Gaussian smoothing scale in the same units as the spatial dimensions of
            `delta`. If not specified or set to None, smoothing is omitted.

        Returns
        -------
        TidalTensor
            TidalTensor object containing the tidal tensor in real space.
        """
        components_list = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]
        tidal_params = ["t_xx", "t_xy", "t_xz", "t_yy", "t_yz", "t_zz"]
        tidal_params = dict(zip(components_list, tidal_params))
        tidal_files = {}
        output_path = os.path.join(path, f"{TIDAL_TENSOR_PATH}_none")
        if gaussian_scale is not None:
            output_path = os.path.join(path, f"{TIDAL_TENSOR_PATH}_{gaussian_scale: .0f}")
        os.makedirs(output_path, exist_ok=True)
        for component in components_list:
            name_tidal_tensor = tidal_tensor_component_calculation(
                delta=delta,
                component=component,
                path=output_path,
                box_size=box_size,
                gaussian_scale=gaussian_scale,
            )
            tidal_files[tidal_params.get(component)] = name_tidal_tensor
            logging.info(f"Tidal Tensor component {component} saved in {name_tidal_tensor}")

        tidal_tensor = cls(gaussian_scale=gaussian_scale, **tidal_files)
        return tidal_tensor

    def _get_tidal_tensor(
        self, component: tuple[int, int], cell_x: int, cell_y: int, cell_z: int
    ) -> np.ndarray:
        """
        Gets the tidal tensor value for a specified tensor component and cell coordinates.

        This function retrieves a specific component of the tidal tensor from
        an HDF5 dataset based on the provided component tuple and the cell
        coordinates (x, y, z). The tidal tensor component is extracted through
        indexed access to the dataset.

        Parameters
        ----------
        component : tuple[int, int]
            A tuple of integers specifying the component of the tidal tensor
            to retrieve, such as (0, 0), (0, 1), etc.
        cell_x : int
            The x-coordinate of the cell in the tidal tensor dataset.
        cell_y : int
            The y-coordinate of the cell in the tidal tensor dataset.
        cell_z : int
            The z-coordinate of the cell in the tidal tensor dataset.

        Returns
        -------
        numpy.ndarray
            The specific component value from the tidal tensor dataset
            corresponding to the provided cell coordinates and component tuple.
        """

        with h5py.File(self.tidal_tensor.get(component), "r") as f:
            dataset = f[NAME_HD5]
            t_component_value = dataset[cell_x, cell_y, cell_z]
        return t_component_value

    def get_tidal_tensor(
        self, cell_x: np.ndarray | int, cell_y: np.ndarray | int, cell_z: np.ndarray | int
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
        cell_x : np.ndarray | int
            The x-coordinates of the grid cell indices.
        cell_y : np.ndarray | int
            The y-coordinates of the grid cell indices.
        cell_z : np.ndarray | int
            The z-coordinates of the grid cell indices.

        Returns
        -------
        np.ndarray
            A 3D NumPy array representing the tidal tensor for the specified grid
            cells. Its shape will be `(N, 3, 3)`, where N is the total number of
            grid points specified by the input arrays.
        """
        if isinstance(cell_x, int):
            cell_x = np.array([cell_x])

        if isinstance(cell_y, int):
            cell_y = np.array([cell_y])

        if isinstance(cell_z, int):
            cell_z = np.array([cell_z])

        if cell_x.shape != cell_y.shape:
            raise ValueError("cell_x and cell_y must have the same shape")

        if cell_x.shape != cell_z.shape:
            raise ValueError("cell_x and cell_z must have the same shape")

        if cell_y.shape != cell_z.shape:
            raise ValueError("cell_y and cell_z must have the same shape")

        size = np.size(cell_x)

        tidal_tensor = np.zeros([size, 3, 3], dtype="f4")
        for key, values in self.tidal_tensor.items():
            for item_pos in range(size):
                tidal_tensor[item_pos, key[0], key[1]] = self._get_tidal_tensor(
                    component=key,
                    cell_x=cell_x[item_pos],
                    cell_y=cell_y[item_pos],
                    cell_z=cell_z[item_pos],
                )

        return tidal_tensor

    def eigenvalues(
        self,
        cell_x: np.ndarray | int,
        cell_y: np.ndarray | int,
        cell_z: np.ndarray | int,
    ) -> np.ndarray:
        """
        Computes the eigenvalues of the tidal tensor at the specified grid cell.

        This function calculates the eigenvalues of the tidal tensor constructed
        at the spatial location defined by the input cell indices. The tidal tensor
        provides insights into the gravitational interactions at that point in the grid.

        Parameters
        ----------
        cell_x : np.ndarray | int
            The x-coordinates of the grid cells where the tidal tensor is evaluated.
        cell_y : np.ndarray | int
            The y-coordinates of the grid cells where the tidal tensor is evaluated.
        cell_z : np.ndarray | int
            The z-coordinates of the grid cells where the tidal tensor is evaluated.

        Returns
        -------
        np.ndarray
            An array of eigenvalues corresponding to the tidal tensor for each input grid cell.
        """
        tidal_tensor = self.get_tidal_tensor(cell_x, cell_y, cell_z)
        eigval, _ = np.linalg.eigh(tidal_tensor)
        return eigval
