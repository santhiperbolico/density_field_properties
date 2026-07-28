import gc
import logging
import os
from typing import Optional

import numpy as np

from density_field_properties.density_field.particle_io import iter_dm_particle_batches
from density_field_properties.density_field.utils import DensityFieldInfo


def weighted_field_cic(
    data: np.ndarray,
    weights: np.ndarray,
    box_size: float,
    n_grid: int,
    mass_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Accumulate per-tracer weights on a cubic grid with the CIC scheme.

    Parameters
    ----------
    data : np.ndarray
        Positions with shape ``(N, 3)`` in Mpc/h.
    weights : np.ndarray
        Non-negative weight per tracer (e.g. particle mass or ``M200b``).
    box_size : float
        Periodic box side length in Mpc/h.
    n_grid : int
        Cells per dimension.
    mass_field : Optional[np.ndarray], optional
        Existing grid to accumulate into.

    Returns
    -------
    np.ndarray
        Grid of accumulated weights, shape ``(n_grid, n_grid, n_grid)``.
    """
    if data.shape[0] != weights.shape[0]:
        raise ValueError("positions and weights must have the same length")
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
                np.add.at(mass_field, (ii, jj, kk), weights * w)

    return mass_field


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
    weights = np.full(data.shape[0], mass_particle, dtype=np.float64)
    return weighted_field_cic(data, weights, box_size, n_grid, mass_field)


def overdensity_from_cic_grid(cic_grid: np.ndarray) -> np.ndarray:
    """
    Convert a CIC-deposited scalar grid to the overdensity ``delta = rho / mean - 1``.

    Parameters
    ----------
    cic_grid : np.ndarray
        Non-negative accumulated weights on a cubic grid.

    Returns
    -------
    np.ndarray
        Overdensity field with the same shape as ``cic_grid``.
    """
    mean_density = cic_grid.mean()
    if mean_density <= 0.0:
        raise ValueError("CIC grid has zero mean; cannot form overdensity.")
    return cic_grid / mean_density - 1.0


def delta_field_from_dm_particles(
    dm_particles_file: str,
    mass_particle: float,
    box_size: float,
    n_grid: int,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Build the matter overdensity field from DM particle positions (text or FastPM BigFile).

    Parameters
    ----------
    dm_particles_file : str
        Path passed to ``iter_dm_particle_batches`` (text file or FastPM block directory).
    mass_particle : float
        DM particle mass in Msun/h.
    box_size : float
        Box side length in Mpc/h.
    n_grid : int
        Grid cells per dimension.
    batch_size : Optional[int], optional
        Particle batch size for I/O; ``None`` reads all particles in one batch.

    Returns
    -------
    np.ndarray
        Matter overdensity ``delta`` on a cubic grid.
    """
    density, density_info = density_field_cic_main(
        dm_particles_file=dm_particles_file,
        mass_particle=mass_particle,
        box_size=box_size,
        n_grid=n_grid,
        batch_size=batch_size,
    )
    return get_delta_density(
        density,
        density_info.n_particles,
        mass_particle,
        box_size,
    )


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
    mass_field = None
    n_particles = 0
    dx = box_size / n_grid

    n_iter = 0
    logging.info("- Computing density field by batches from file %s" % dm_particles_file)

    for positions, start_idx, end_idx in iter_dm_particle_batches(
        dm_particles_file, batch_size=batch_size
    ):
        mass_field = mass_field_cic(positions, mass_particle, box_size, n_grid, mass_field)
        n_batch = positions.shape[0]
        n_particles += n_batch
        logging.info(
            "\t -- %i particles %i to %i have been loaded." % (n_iter, start_idx, end_idx)
        )
        n_iter += 1
        del positions
        gc.collect()

    if mass_field is None:
        mass_field = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

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
    normalized = os.path.normpath(dm_particles_file)
    base = os.path.basename(normalized)
    if os.path.isdir(normalized):
        if base.isdigit():
            output_stem = os.path.basename(os.path.dirname(normalized))
        else:
            output_stem = base
    else:
        output_stem, _ = os.path.splitext(base)

    output_data_file = os.path.join(path, "%s_density" % output_stem)
    density.tofile(output_data_file)

    output_info_file = os.path.join(path, "%s_density_info.txt" % output_stem)
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


def delta_field_from_saved_cic(
    density_field_cic_file: str,
    density_field_cic_info_file: str,
    expected_box_size: Optional[float] = None,
) -> np.ndarray:
    """
    Load a precomputed CIC density grid and convert it to matter overdensity ``delta``.

    Parameters
    ----------
    density_field_cic_file : str
        Binary density file written by ``save_density_field_cic``.
    density_field_cic_info_file : str
        Companion ``*_density_info.txt`` metadata file.
    expected_box_size : Optional[float], optional
        If set, require a matching ``box_size`` in the info file (Mpc/h).

    Returns
    -------
    np.ndarray
        Matter overdensity on the stored grid.

    Raises
    ------
    ValueError
        If metadata is missing or ``expected_box_size`` does not match.
    """
    density, density_info = load_density_field_cic(
        density_field_cic_file, density_field_cic_info_file
    )
    if density_info is None:
        raise ValueError("density_field_cic_info_file is required for saved CIC grids")
    if expected_box_size is not None and not np.isclose(
        float(density_info.box_size), float(expected_box_size)
    ):
        raise ValueError(
            "Saved CIC box_size "
            f"{density_info.box_size} does not match expected {expected_box_size}"
        )
    return get_delta_density(
        density,
        density_info.n_particles,
        density_info.mass_particle,
        density_info.box_size,
    )


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
