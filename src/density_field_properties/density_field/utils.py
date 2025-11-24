from typing import Optional, Self

import numpy as np


def get_grid_cell(data: np.ndarray, box_size: float | int, n_grid: int) -> np.ndarray:
    """
    Compute the grid cell indices for points in a dataset.

    This function determines which grid cell each point in the provided dataset
    belongs to, based on the specified box size and grid resolution. The result
    is returned as an array of grid cell indices.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape `(N, M)` where `N` is the number of points and `M` is the
        dimensionality of each point.
    box_size : float or int
        The size of the box within which the grid is defined.
    n_grid : int
        The resolution of the grid, representing the number of divisions along one
        axis of the box.

    Returns
    -------
    numpy.ndarray
        An array of shape `(N, M)` that contains the grid cell indices for each
        point along each dimension.
    """
    n_cols = 1
    if data.shape[1] > 1:
        n_cols = data.shape[1]
    cells = np.zeros((data.shape[0], n_cols))
    dx = box_size / n_grid
    for col_i in range(n_cols):
        cells[:, col_i] = np.floor(data[:, 0] / dx).astype(int) % n_grid
    return cells.astype(int)


class DensityFieldInfo:
    """
    Represents metadata information for a density field.

    This class is designed to encapsulate the critical parameters of a density
    field, which are necessary for data analysis and simulations. It stores
    information such as the physical dimensions of the field, the number of
    grids in the field, the particle mass, and the total number of particles.

    Attributes
    ----------
    box_size : float
        The size of the simulation box or field in Mpc/h units.
    n_grid : int
        The number of grid cells in the density field.
    n_particles : int
        The total number of particles within the field.
    mass_particle : float
        The mass of a single particle in the simulation or field in M_sun units.
    process_duration : Optional[float] = None
        Time taken to process the density field in seconds. Defaults to None.
    """

    def __init__(
        self,
        box_size: float,
        n_grid: int,
        n_particles: int,
        mass_particle: float,
        process_duration: Optional[float] = None,
    ):
        self.box_size = box_size
        if box_size % 1 == 0:
            self.box_size = int(box_size)
        self.n_grid = int(n_grid)
        self.n_particles = int(n_particles)
        self.mass_particle = mass_particle
        self.process_duration = process_duration

    def __str__(self):
        attributes = (
            f"box_size={self.box_size}, "
            f"n_grid={self.n_grid}, "
            f"n_particles={self.n_particles}, "
            f"mass_particle={self.mass_particle}"
        )
        if self.process_duration is not None:
            attributes += f", process_duration={self.process_duration}"
        return f"DensityFieldInfo({attributes})"

    def __repr__(self):
        return self.__str__()

    def add_process_duration(self, duration: float) -> None:
        self.process_duration = duration
        return None

    def save_information(self, output_info_file: str) -> None:
        row_information = "%f,%i,%i,%f," % (
            self.box_size,
            self.n_grid,
            self.n_particles,
            self.mass_particle,
        )
        if self.process_duration is not None:
            row_information += "%f" % self.process_duration
        row_information += "\n"
        with open(output_info_file, "w") as f:
            f.write("# box_size,n_grid,n_particles,mass_particle, process_duration\n")
            f.write(row_information)
        return None

    @classmethod
    def load_information(cls, density_info_file: str) -> Self:
        density_info = np.loadtxt(density_info_file, delimiter=",")
        return cls(*density_info)


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
