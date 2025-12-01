import logging
import os
from typing import Callable, Optional, Self

import h5py
import numpy as np
from numpy.fft import irfftn, rfftn

from density_field_properties.density_field.fourrier_transformations import kgrid

NAME_HD5 = "tidaltensor"
TIDAL_TENSOR_PATH = "tidal_tensor"
GAUSSIAN_SCALE_DEFAULT = "none"


def interpolate_array_generator(
    array_0: np.ndarray, array_1: np.ndarray, gaussian_scale_0: float, gaussian_scale_1: float
) -> Callable[[np.ndarray | int | float], np.ndarray]:
    def init(gaussian_scale: int | float | np.ndarray) -> np.ndarray:
        slope = (array_1 - array_0) / (gaussian_scale_0 - gaussian_scale_1)
        return (gaussian_scale_1 - gaussian_scale) * slope + array_1

    return init


def tidal_tensor_component_calculation(
    delta: np.ndarray,
    box_size: float,
    path: str,
    component: tuple[int, int],
    gaussian_scale: Optional[float | int] = None,
) -> str:
    """
    Calculates a specific component of the tidal tensor from input density field halo_catalog
    and saves the resulting tensor component to an HDF5 file. This function computes
    the tidal tensor component by transforming the density field into Fourier space,
    applying relevant operations, and then transforming back to real space.

    Parameters
    ----------
    delta : np.ndarray
        3D density field halo_catalog array represented as a numpy array.
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
    """
    Represents a Tidal Tensor for analyzing gravitational potential structures in 3D space.

    The TidalTensor class provides functionality for working with tidal tensors, which describe
    the second derivatives of the gravitational potential in a defined space. The class allows
    for constructing tidal tensors from file halo_catalog, density contrast fields, and retrieving
    specific tensor components for grid coordinates. The tensors are stored as a dictionary
    of components, which may be read or computed as needed. This implementation is especially
    valuable in cosmological simulations and modeling large-scale structures in astrophysical
     studies.

    Attributes
    ----------
    tidal_tensor : dict[tuple[int, int], str]
        Dictionary mapping tensor components (specified as 2D tuples) to the file paths of
        their associated datasets. Keys represent the component indices, and values are
        the file paths storing the corresponding halo_catalog.
    gaussian_scale : Optional[float | int]
        The Gaussian smoothing scale used during tidal tensor computation or representation.
        If not specified, smoothing or scale adjustments are omitted.
    """

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
            output_path = f"{TIDAL_TENSOR_PATH}_{gaussian_scale}".replace(".", "*")
            output_path = os.path.join(path, output_path)
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
        if cell_x % 1 > 0:
            raise ValueError("cell_x must be an integer")
        if cell_y % 1 > 0:
            raise ValueError("cell_y must be an integer")
        if cell_z % 1 > 0:
            raise ValueError("cell_z must be an integer")

        with h5py.File(self.tidal_tensor.get(component), "r") as f:
            dataset = f[NAME_HD5]
            t_component_value = dataset[int(cell_x), int(cell_y), int(cell_z)]
        return t_component_value

    def get_tidal_tensor(
        self, cell_x: np.ndarray | int, cell_y: np.ndarray | int, cell_z: np.ndarray | int
    ) -> np.ndarray:
        """
        Computes the tidal tensor for given grid cell indices.

        This function calculates a 3x3 tidal tensor for each specified grid point,
        where the tidal tensor elements are determined by precomputed halo_catalog. The
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
        eigval = np.zeros((tidal_tensor.shape[0], 3))
        for i in range(tidal_tensor.shape[0]):
            eigval[i] = np.linalg.eigvalsh(tidal_tensor[i])
        return eigval


class TidalTensorArray:
    def __init__(
        self, tidal_tensor_list: list[TidalTensor], gaussian_scale_list: list[float | int]
    ):
        self.tidal_tensors = dict(zip(gaussian_scale_list, tidal_tensor_list))

    @property
    def gaussian_scale_list(self) -> list[float | int]:
        """
        Gets the list of Gaussian scale values sorted in ascending order.

        The Gaussian scales are obtained from the keys of the `tidal_tensors`
        attribute, which represent the available scales defined in the object.
        These values are then sorted before being returned.

        Returns
        -------
        list of float or int
            A sorted list of Gaussian scale values.
        """
        gs_list = list(self.tidal_tensors.keys())
        gs_list.sort()
        return gs_list

    def get_gaussian_scale_bin(
        self, gaussian_scale: np.ndarray | float | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding bin of a given Gaussian scale from a predefined Gaussian scale list.

        The function searches for two consecutive Gaussian scales in the list such that the given
        scale falls between them. If the given scale is not within the range of the Gaussian scale
        list, an error is raised.

        Parameters
        ----------
        gaussian_scale : np.ndarray | float | int
            The Gaussian scale to locate in the list. Must be within the bounds of the predefined
            list of Gaussian scales.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            A tuple containing two consecutive Gaussian scales that bound the input scale.

        Raises
        ------
        ValueError
            If the Gaussian scale is not found within the ranges of the predefined list.
        """
        if isinstance(gaussian_scale, (int, float)):
            gaussian_scale = np.array([gaussian_scale])

        gs_0 = np.zeros(gaussian_scale.shape)
        gs_1 = np.zeros(gaussian_scale.shape)
        for i in range(gaussian_scale.shape[0]):
            gs_scales = np.array(self.gaussian_scale_list)
            diff_gs = gs_scales - gaussian_scale[i]
            if (diff_gs < 0).all() or (diff_gs > 0).all():
                raise ValueError(f"Gaussian scale {gaussian_scale[i]} not found in the list")
            gs_i = int(np.where((diff_gs == max(diff_gs[diff_gs <= 0])))[0])
            gs_0[i] = self.gaussian_scale_list[gs_i]
            gs_1[i] = self.gaussian_scale_list[gs_i + 1]

        return gs_0, gs_1

    @classmethod
    def from_folder(cls, path: str) -> Self:
        """
        Creates an instance of the class from the contents of a folder.

        This class method examines the contents of the specified folder to identify
        subfolders containing tidal tensor halo_catalog and parses their associated Gaussian
        scale values. It then constructs and initializes the class using a list of
        `TidalTensor` instances and their corresponding Gaussian scale values, derived
        from the detected subfolder structure.

        Parameters
        ----------
        path : str
            The file system path to the folder containing the tidal tensor halo_catalog
            subdirectories.

        Returns
        -------
        TidalTensorArray
            An instance of the class initialized with tidal tensor halo_catalog derived
            from the specified folder.
        """
        tidal_tensor_list = []
        gaussian_scale_list = []
        for file in os.listdir(path):
            path_r = os.path.join(path, file)
            if file.startswith(TIDAL_TENSOR_PATH) and os.path.isdir(path_r):
                gaussian_scale = file.split("_")[-1]
                if gaussian_scale == "none":
                    gaussian_scale = None
                else:
                    gaussian_scale = float(gaussian_scale.replace("*", "."))
                tidal_tensor_list.append(TidalTensor.from_folder(path_r, gaussian_scale))
                gaussian_scale_list.append(gaussian_scale)
        return cls(tidal_tensor_list, gaussian_scale_list)

    @classmethod
    def from_delta(
        cls, delta: np.ndarray, box_size: float, path: str, gaussian_scale_list: list[float | int]
    ) -> Self:
        """
        Create an instance of the class from a given set of density field halo_catalog
        and scales.

        This method processes a density field to compute tidal tensors for
        provided Gaussian smoothing scales and stores them at a specified path.

        Parameters
        ----------
        delta : numpy.ndarray
            The density field halo_catalog to be processed.
        box_size : float
            The size of the simulation box in physical units.
        path : str
            The directory path where the computed tidal tensors will be saved.
        gaussian_scale_list : list of float or int
            List of Gaussian smoothing scales for tensor computation.

        Returns
        -------
        TidalTensorArray
            An instance of the class initialized with computed tidal tensors and
            the provided Gaussian smoothing scales.
        """
        output_path = os.path.join(path, f"{TIDAL_TENSOR_PATH}")
        os.makedirs(output_path, exist_ok=True)
        tidal_tensor_list = []
        for gaussian_scale in gaussian_scale_list:
            tidal_tensor = TidalTensor.from_delta(delta, box_size, output_path, gaussian_scale)
            tidal_tensor_list.append(tidal_tensor)
        return cls(tidal_tensor_list, gaussian_scale_list)

    def _get_tidal_tensor(
        self, cell_x: int, cell_y: int, cell_z: int, gaussian_scale: float | int
    ) -> np.ndarray:
        """
        Computes and retrieves the tidal tensor for a given set of cell coordinates and
        gaussian scale. The tidal tensor provides information related to the second
        derivatives of the gravitational potential in a simulated cosmological field.

        Parameters
        ----------
        cell_x : int
            The x-coordinate(s) of the cell(s) for which the tidal tensor is computed.
        cell_y : int
            The y-coordinate(s) of the cell(s) for which the tidal tensor is computed.
        cell_z : int
            The z-coordinate(s) of the cell(s) for which the tidal tensor is computed.
        gaussian_scale : float | int
            The scale used for Gaussian filtering of the gravitational field during tidal
            tensor computation. This scale determines the smoothing applied.

        Returns
        -------
        np.ndarray
            The tidal tensor at the specified coordinates and gaussian scale.
        """
        tt_object = self.tidal_tensors[gaussian_scale]
        return tt_object.get_tidal_tensor(int(cell_x), int(cell_y), int(cell_z))

    def get_tidal_tensor(
        self,
        cell_x: np.ndarray | int,
        cell_y: np.ndarray | int,
        cell_z: np.ndarray | int,
        gaussian_scale: np.ndarray | float | int,
    ) -> np.ndarray:
        """
        Calculates the tidal tensor for given spatial coordinates and Gaussian scale.

        This function computes the tidal tensor by interpolating values between two different
        Gaussian scale bins derived from the given scale value. It utilizes two intermediate
        internal computations to fetch tidal tensors corresponding to neighboring Gaussian
        scale bins, and then applies a linear interpolation to determine the resultant
        tidal tensor.

        Parameters
        ----------
        cell_x : np.ndarray | int
            The x-coordinate(s) (in grid space) where the tidal tensor is evaluated.
        cell_y : np.ndarray | int
            The y-coordinate(s) (in grid space) where the tidal tensor is evaluated.
        cell_z : np.ndarray | int
            The z-coordinate(s) (in grid space) where the tidal tensor is evaluated.
        gaussian_scale : np.ndarray | float | int
            The Gaussian scale value to determine the scale-specific components
            of the tidal tensor.

        Returns
        -------
        np.ndarray
            A multidimensional array representing the tidal tensor at the specified coordinates
            and scale.
        """
        if isinstance(gaussian_scale, (int, float)):
            gaussian_scale = np.array([gaussian_scale])

        if isinstance(cell_x, (int, float)):
            cell_x = np.array([cell_x])

        if isinstance(cell_y, (int, float)):
            cell_y = np.array([cell_y])

        if isinstance(cell_z, (int, float)):
            cell_z = np.array([cell_z])

        if cell_x.shape != cell_y.shape:
            raise ValueError("cell_x and cell_y must have the same shape")

        if cell_x.shape != cell_z.shape:
            raise ValueError("cell_x and cell_z must have the same shape")

        if cell_y.shape != cell_z.shape:
            raise ValueError("cell_y and cell_z must have the same shape")

        if gaussian_scale.shape != cell_x.shape:
            raise ValueError("gaussian_scale and cell_x must have the same shape")

        tidal_tensor = np.zeros([cell_x.shape[0], 3, 3], dtype="f4")
        for i_pos, gs in enumerate(gaussian_scale):
            gs_limits = self.get_gaussian_scale_bin(gs)
            gs_0, gs_1 = gs_limits[0][0], gs_limits[1][0]
            tidal_tensor_i0 = self._get_tidal_tensor(
                cell_x=cell_x[i_pos],
                cell_y=cell_y[i_pos],
                cell_z=cell_z[i_pos],
                gaussian_scale=gs_0,
            )
            tidal_tensor_i1 = self._get_tidal_tensor(
                cell_x=cell_x[i_pos],
                cell_y=cell_y[i_pos],
                cell_z=cell_z[i_pos],
                gaussian_scale=gs_1,
            )
            interpolator = interpolate_array_generator(
                array_0=tidal_tensor_i0,
                array_1=tidal_tensor_i1,
                gaussian_scale_0=gs_0,
                gaussian_scale_1=gs_1,
            )
            tidal_tensor[i_pos] = interpolator(gs)
        return tidal_tensor

    def _get_eigenvalue(
        self, cell_x: int, cell_y: int, cell_z: int, gaussian_scale: float | int
    ) -> np.ndarray:
        """
        Computes and retrieves the eigenvalue of the tidal tensor at the specified for the given
        spatial coordinates and Gaussian scale.

        Parameters
        ----------
        cell_x : int
            The x-coordinate(s) of the cell(s) for which the tidal tensor is computed.
        cell_y : int
            The y-coordinate(s) of the cell(s) for which the tidal tensor is computed.
        cell_z : int
            The z-coordinate(s) of the cell(s) for which the tidal tensor is computed.
        gaussian_scale : float | int
            The scale used for Gaussian filtering of the gravitational field during tidal
            tensor computation. This scale determines the smoothing applied.

        Returns
        -------
        np.ndarray
            The eigenvalue of the tidal tensor at the specified coordinates and scale.
        """
        tt_object = self.tidal_tensors[gaussian_scale]
        return tt_object.eigenvalues(int(cell_x), int(cell_y), int(cell_z))

    def eigenvalues(
        self,
        cell_x: np.ndarray | int,
        cell_y: np.ndarray | int,
        cell_z: np.ndarray | int,
        gaussian_scale: np.ndarray | float | int,
    ) -> np.ndarray:
        """
        Compute the eigenvalues of the tidal tensor.

        This method calculates the eigenvalues of the tidal tensor for a specified
        cell given its coordinates and a Gaussian smoothing scale. The tidal tensor
        is obtained internally, and its eigenvalues are derived using a numerical
        calculation.

        Parameters
        ----------
        cell_x : np.ndarray or int
            The x-coordinate of the cell where the tidal tensor is computed.
        cell_y : np.ndarray or int
            The y-coordinate of the cell where the tidal tensor is computed.
        cell_z : np.ndarray or int
            The z-coordinate of the cell where the tidal tensor is computed.
        gaussian_scale : np.ndarray | float | int
            The Gaussian smoothing scale to filter the halo_catalog before computing
            the tidal tensor.

        Returns
        -------
        np.ndarray
            A 1D array containing the eigenvalues of the derived tidal tensor.
        """
        if isinstance(gaussian_scale, (int, float)):
            gaussian_scale = np.array([gaussian_scale])

        if isinstance(cell_x, (int, float)):
            cell_x = np.array([cell_x])

        if isinstance(cell_y, (int, float)):
            cell_y = np.array([cell_y])

        if isinstance(cell_z, (int, float)):
            cell_z = np.array([cell_z])

        if cell_x.shape != cell_y.shape:
            raise ValueError("cell_x and cell_y must have the same shape")

        if cell_x.shape != cell_z.shape:
            raise ValueError("cell_x and cell_z must have the same shape")

        if cell_y.shape != cell_z.shape:
            raise ValueError("cell_y and cell_z must have the same shape")

        if gaussian_scale.shape != cell_x.shape:
            raise ValueError("gaussian_scale and cell_x must have the same shape")

        n_rows = cell_x.shape[0]
        eigval = np.zeros((n_rows, 3))
        for i_pos, gs in enumerate(gaussian_scale):
            gs_limits = self.get_gaussian_scale_bin(gs)
            gs_0, gs_1 = gs_limits[0][0], gs_limits[1][0]
            eigenvalue_i0 = self._get_eigenvalue(
                cell_x=cell_x[i_pos],
                cell_y=cell_y[i_pos],
                cell_z=cell_z[i_pos],
                gaussian_scale=gs_0,
            )
            eigenvalue_i1 = self._get_eigenvalue(
                cell_x=cell_x[i_pos],
                cell_y=cell_y[i_pos],
                cell_z=cell_z[i_pos],
                gaussian_scale=gs_1,
            )
            interpolator = interpolate_array_generator(
                array_0=eigenvalue_i0,
                array_1=eigenvalue_i1,
                gaussian_scale_0=gs_0,
                gaussian_scale_1=gs_1,
            )
            eigval[i_pos] = interpolator(gs)

        return eigval

    def get_tidal_anisotropy_and_overdensity(
        self,
        cell_x: np.ndarray | int,
        cell_y: np.ndarray | int,
        cell_z: np.ndarray | int,
        gaussian_scale: np.ndarray | float | int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the tidal anisotropy and overdensity for a specified cell and gaussian scale.

        The method computes the tidal anisotropy and overdensity using eigenvalues of the
        tidal tensor, obtained based on the given cell coordinates and gaussian scale parameter.
        The tidal anisotropy is determined from the eigenvalues to measure the local tidal
        distortion, while the overdensity is calculated as their sum. More information on the
            The multi-dimensional halo assembly bias can be preserved when enhancing
            halo properties with HALOSCOPE, Ramakrishnan et al., 2024

        Parameters
        ----------
        cell_x : np.ndarray or int
            The x-coordinate(s) of the cell(s) in the grid.
        cell_y : np.ndarray or int
            The y-coordinate(s) of the cell(s) in the grid.
        cell_z : np.ndarray or int
            The z-coordinate(s) of the cell(s) in the grid.
        gaussian_scale : np.ndarray | float | int
            Smoothing scale applied for evaluating the tidal tensor.

        Returns
        -------
        tidal_anisotropy: np.ndarray
            Tidal anisotropy, quantified using eigenvalue differences.
        delta_s: np.ndarray
            Overdensity, calculated as the sum of eigenvalues.
        """
        eigval = self.eigenvalues(cell_x, cell_y, cell_z, gaussian_scale)
        delta_s = eigval.sum(axis=1)
        q2 = 0.5 * (
            (eigval[:, 0] - eigval[:, 1]) ** 2
            + (eigval[:, 1] - eigval[:, 2]) ** 2
            + (eigval[:, 2] - eigval[:, 0]) ** 2
        )
        tidal_anisotropy = np.sqrt(q2) / (1.0 + delta_s + 1e-15)
        return tidal_anisotropy, delta_s
