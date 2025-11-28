import logging
import os
from typing import Optional

import numpy as np

from density_field_properties.data.read_data import (
    iter_rockstar_halo_catalog_by_offset,
    read_rockstar_halo_catalog,
)
from density_field_properties.density_field.utils import get_grid_cell
from density_field_properties.tidal_tensor import TIDAL_TENSOR_PATH, TidalTensorArray

ANISOTROPY_PATH = "tidal_anisotropy"


def _tidal_anisotropy_and_overdensity_from_halo_calaog_batches(
    output_path: str,
    tidal_tensor_array: TidalTensorArray,
    rockstar_halo_catalog: str,
    n_grid: int,
    box_size: int,
    n_lines: Optional[int] = None,
    xyz_position: Optional[list[int]] = None,
    r_position: int = 4,
    batch_size: Optional[int] = None,
) -> str:
    r_min = np.array(tidal_tensor_array.gaussian_scale_list).min()
    r_max = np.array(tidal_tensor_array.gaussian_scale_list).max()

    offset = 0
    batch_index = 0
    lines_readed = 0
    for halo_catalog, batch_start_offset, next_offset in iter_rockstar_halo_catalog_by_offset(
        rockstar_halo_catalog,
        batch_size=batch_size,
        start_offset=offset,
    ):
        logging.info(
            f"Batch %i: %i halos, offset start=%i"
            % (batch_index, halo_catalog.shape[0], batch_start_offset)
        )
        halo_catalog[:, xyz_position] = get_grid_cell(
            data=halo_catalog[:, xyz_position], box_size=box_size, n_grid=n_grid
        )

        halo_selected = (halo_catalog[:, r_position] > r_min) & (
            halo_catalog[:, r_position] < r_max
        )
        logging.info(
            f"Number of halos in the selected radius: {np.sum(halo_selected)} of "
            f"{halo_selected.size}"
        )

        halo_catalog = halo_catalog[halo_selected]
        tidal_anisotropy, overdensity = tidal_tensor_array.get_tidal_anisotropy_and_overdensity(
            halo_catalog[:, xyz_position[0]],
            halo_catalog[:, xyz_position[1]],
            halo_catalog[:, xyz_position[2]],
            halo_catalog[:, r_position],
        )

        np.savetxt(output_path + f"{batch_index}_anisotropy.txt", tidal_anisotropy)
        np.savetxt(output_path + f"{batch_index}_overdensity.txt", overdensity)

        lines_readed += batch_size
        batch_index += 1
        offset = next_offset

        if lines_readed > n_lines:
            break

    return output_path


def _tidal_anisotropy_and_overdensity_from_halo_calaog_complete(
    output_path: str,
    tidal_tensor_array: TidalTensorArray,
    rockstar_halo_catalog: str,
    n_grid: int,
    box_size: int,
    n_lines: Optional[int] = None,
    xyz_position: Optional[list[int]] = None,
    r_position: int = 4,
) -> str:
    r_min = np.array(tidal_tensor_array.gaussian_scale_list).min()
    r_max = np.array(tidal_tensor_array.gaussian_scale_list).max()

    halo_catalog = read_rockstar_halo_catalog(path=rockstar_halo_catalog, n_lines=n_lines)

    halo_catalog[:, xyz_position] = get_grid_cell(
        data=halo_catalog[:, xyz_position], box_size=box_size, n_grid=n_grid
    )

    halo_selected = (halo_catalog[:, r_position] > r_min) & (halo_catalog[:, r_position] < r_max)
    logging.info(
        f"Number of halos in the selected radius: {np.sum(halo_selected)} of {halo_selected.size}"
    )

    halo_catalog = halo_catalog[halo_selected]
    tidal_anisotropy, overdensity = tidal_tensor_array.get_tidal_anisotropy_and_overdensity(
        halo_catalog[:, xyz_position[0]],
        halo_catalog[:, xyz_position[1]],
        halo_catalog[:, xyz_position[2]],
        halo_catalog[:, r_position],
    )

    np.savetxt(output_path + f"anisotropy.txt", tidal_anisotropy)
    np.savetxt(output_path + f"overdensity.txt", overdensity)

    return output_path


def tidal_anisotropy_and_overdensity_from_halo_calaog(
    path: str,
    rockstar_halo_catalog: str,
    n_grid: int,
    box_size: int,
    n_lines: Optional[int] = None,
    xyz_position: Optional[list[int]] = None,
    r_position: int = 4,
    batch_size: Optional[int] = None,
) -> str:
    """
    Calculates tidal anisotropy and overdensity from a halo catalog, iterating through
    batches and saving the results to output files.

    Parameters
    ----------
    path : str
        Path to the main directory where the tidal tensor is stored.
    rockstar_halo_catalog : str
        Path to the Rockstar halo catalog file to process.
    n_grid : int
        Number of grid cells in one dimension of the simulation grid.
    box_size : int
        The length of one side of the simulation box.
    xyz_position : list of int, optional
        Indices of the spatial coordinates in the halo catalog. Defaults to [1, 2, 3].
    batch_size : int, optional
        The size of each batch to process from the halo catalog.

    Returns
    -------
    output_path: str
        Path to the directory where the tidal anisotropy an overdensity are stored.

    Raises
    ------
    ValueError
        If the tidal tensor folder path does not exist.
    """
    if xyz_position is None:
        xyz_position = [1, 2, 3]

    tidal_path = os.path.join(path, f"{TIDAL_TENSOR_PATH}")
    if not os.path.exists(tidal_path):
        raise ValueError(
            "%s path doesn' exist. Before you need to calculate the Tidals Tensors." % tidal_path
        )

    tidal_tensor_array = TidalTensorArray.from_folder(path=tidal_path)
    output_path = os.path.join(path, f"{ANISOTROPY_PATH}/")
    os.makedirs(output_path, exist_ok=True)

    if batch_size:
        return _tidal_anisotropy_and_overdensity_from_halo_calaog_batches(
            output_path=output_path,
            tidal_tensor_array=tidal_tensor_array,
            rockstar_halo_catalog=rockstar_halo_catalog,
            n_grid=n_grid,
            box_size=box_size,
            n_lines=n_lines,
            xyz_position=xyz_position,
            r_position=r_position,
            batch_size=batch_size,
        )

    return _tidal_anisotropy_and_overdensity_from_halo_calaog_complete(
        output_path=output_path,
        tidal_tensor_array=tidal_tensor_array,
        rockstar_halo_catalog=rockstar_halo_catalog,
        n_grid=n_grid,
        box_size=box_size,
        n_lines=n_lines,
        xyz_position=xyz_position,
        r_position=r_position,
    )
