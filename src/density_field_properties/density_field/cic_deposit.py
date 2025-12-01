import gc
import logging
import os
from typing import Optional

import numpy as np

from density_field_properties.density_field.utils import DensityFieldInfo


def mass_field_cic(
    data: np.ndarray,
    mass_particle: float,
    box_size: float,
    n_grid: int,
    mass_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Function that calculates the accumulated mass in a grid of size Ngrid, Ngrid, Ngrid following
    the CIC scheme.

    Parameters
    ----------
    data: np.ndarray,
        Array with dark matter particle simulations, where the first three
        columns indicate the x, y, z positions respectively in Mpc/h.
    mass_particle: float,
        Mass of the particle used in the simulation in Solar Masses.
    box_size: float,
        Size of the simulation box in Mpc/h.
    n_grid: int,
        Size of the grid to calculate.
    mass_field: Optional[np.ndarray] = None
        Array with masses associated to the Ngrid, Ngrid, Ngrid grid precalculated.
        If provided, masses will be added to this parameter.

    Returns
    -------
    mass_field: np.ndarray,
        Array with masses added according to CIC methodology. If mass_field parameter is provided,
        new values will be added to this element.
    """
    dx = box_size / n_grid
    if not isinstance(mass_field, np.ndarray):
        mass_field = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    grid_position_x = data[:, 0] / dx
    grid_position_y = data[:, 1] / dx
    grid_position_z = data[:, 2] / dx

    cell_i = np.floor(grid_position_x).astype(int) % n_grid
    cell_j = np.floor(grid_position_y).astype(int) % n_grid
    cell_k = np.floor(grid_position_z).astype(int) % n_grid

    distance_fraction_x = grid_position_x - cell_i
    distance_fraction_y = grid_position_y - cell_j
    distance_fraction_z = grid_position_z - cell_k

    for di in [0, 1]:
        wx = (1 - distance_fraction_x) * (1 - di) + distance_fraction_x * di
        ii = (cell_i + di) % n_grid
        for dj in (0, 1):
            wy = (1 - distance_fraction_y) * (1 - dj) + distance_fraction_y * dj
            jj = (cell_j + dj) % n_grid
            wxy = wx * wy
            for dk in (0, 1):
                wz = (1 - distance_fraction_z) * (1 - dk) + distance_fraction_z * dk
                kk = (cell_k + dk) % n_grid
                w = wxy * wz
                np.add.at(mass_field, (ii, jj, kk), mass_particle * w)

    return mass_field


def density_field_cic_main(
    dm_particles_file: str,
    mass_particle: float,
    box_size: float,
    n_grid: int,
    batch_size: Optional[int] = None,
) -> tuple[np.ndarray, DensityFieldInfo]:
    """
    Computes the density field using Cloud-In-Cell (CIC) interpolation and processes
    halo_catalog in batches for memory efficiency. The procedure reads particle halo_catalog from
    a file, processes it via the CIC method, and returns the resulting density
    field and total number of particles used.

    Parameters
    ----------
    dm_particles_file : str
        Path to the file containing particle halo_catalog.
    mass_particle : float
        Mass of a single particle.
    box_size : float
        Size of the simulation box.
    n_grid : int
        Number of grid cells along one dimension.
    batch_size : Optional[int] = None
        Number of particle halo_catalog rows to process in a single batch. If None,
        the entire file is processed in one go.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
            - The resulting density field as a NumPy array.
            - The total number of particles used in the computation.
    """
    batch = 0
    mass_field = None
    n_particles = 0
    dx = box_size / n_grid

    n_iter = 0
    condition = True
    logging.info("- Computing density field by batches from file %s" % dm_particles_file)

    while condition:
        data = np.loadtxt(dm_particles_file, skiprows=batch, max_rows=batch_size)
        n_particles_batch = data.shape[0]
        condition = n_particles_batch > 0
        if condition:
            mass_field = mass_field_cic(data, mass_particle, box_size, n_grid, mass_field)
            batch += n_particles_batch
            n_particles += n_particles_batch
            logging.info(
                "\t -- %i Lines %i to %i have been loaded."
                % (n_iter, batch - n_particles_batch, batch)
            )
            n_iter += 1
            del data
            gc.collect()

    density = mass_field / (dx**3)
    density_info = DensityFieldInfo(
        box_size=box_size, n_grid=n_grid, n_particles=n_particles, mass_particle=mass_particle
    )
    return density, density_info


def save_density_field_cic(
    density: np.ndarray, path: str, dm_particles_file: str, density_info: DensityFieldInfo
) -> str:
    """
    Saves a density field array to a file in binary format using the specified
    filename and number of particles. The file name is formatted to include
    the number of particles and adjusts for existing file extensions.

    Parameters
    ----------
    density : numpy.ndarray
        A NumPy array representing the density field to be saved.
    path: str
        Path to the directory where the file will be saved.
    dm_particles_file : str
        The base name of the file that links to the density field.
    density_info : DensityFieldInfo
        Information about the density field.

    Returns
    -------
    output_file: str
        Outpu file name.
    """
    dm_particles_file = dm_particles_file.split(".")[0]
    output_data_file = "%s_density" % os.path.basename(dm_particles_file).replace(".dat", "")
    output_data_file = os.path.join(path, output_data_file)
    density.tofile(output_data_file)

    output_info_file = "%s_density_info.txt" % os.path.basename(dm_particles_file).replace(
        ".dat", ""
    )

    output_info_file = os.path.join(path, output_info_file)
    density_info.save_information(output_info_file)

    return output_data_file


def load_density_field_cic(
    density_field_cic_file: str, density_field_cic_info_file: Optional[str] = None
) -> tuple[np.ndarray, DensityFieldInfo | None]:
    """
    Load a density field calculated using the Cloud-In-Cell (CIC) method from a provided file.
    The function reads the density field from a binary file and extracts the number of
    particles from the filename. The filename is expected to follow a specific convention
    where the number of particles is indicated at the end of the filename, prior to the
    file extension.

    Parameters
    ----------
    density_field_cic_file : str
        Path to the binary file containing the CIC density field halo_catalog. The filename should
        include the number of particles as a suffix before the file extension.
    density_field_cic_info_file: Optional[str] = None,
        Path to the text file containing the density field information.

    Returns
    -------
    density: np.ndarray
        A numpy array representing the density field read from the file.
    density_info: Optional[DensityFieldInfo] = None
        Information about the density field.
    """
    density = np.fromfile(density_field_cic_file)
    density_info = None
    if density_field_cic_info_file is not None:
        density_info = DensityFieldInfo.load_information(density_field_cic_info_file)
        n_grid = density_info.n_grid
        density = density.reshape((n_grid, n_grid, n_grid))
    return density, density_info


def get_delta_density(
    density_field: np.ndarray, n_particles: int, mass_particle: float, box_size: int
) -> np.ndarray:
    """
    Computes the density contrast (delta density field).

    The function calculates the density contrast, also known as the delta density field,
    which measures the deviation of a given density field from the mean density.
    The calculation involves normalizing the density field by the mean density
    and subtracting one.

    Parameters
    ----------
    density_field : np.ndarray
        The input 3D array representing the density field.
    n_particles : int
        The number of particles in the system.
    mass_particle : float
        The mass of each particle.
    box_size : int
        The length of one side of the cubic box containing the system.

    Returns
    -------
    np.ndarray
        A 3D array representing the density contrast field, where each value
        indicates the deviation from the mean density normalized by the mean density.
    """
    density_bar = n_particles * mass_particle / (box_size**3)
    return density_field / density_bar - 1
