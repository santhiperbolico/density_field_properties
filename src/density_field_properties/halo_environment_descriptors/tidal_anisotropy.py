import gc
import logging
import os
from typing import Optional

import numpy as np

from density_field_properties.density_field.utils import get_grid_cell
from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData, HaloCatalogReader
from density_field_properties.tidal_tensor import TIDAL_TENSOR_PATH, TidalTensorArray

ANISOTROPY_PATH = "tidal_anisotropy"


def format_halo_catalog(
    halo_catalog: HaloCatalogData,
    box_size: int,
    n_grid: int,
    r_min: float | None = None,
    r_max: float | None = None,
) -> HaloCatalogData:
    """
    Formats the halo catalog data by adjusting positions to a grid system and optionally filtering
    halos based on their radius if range limits are provided.

    Parameters
    ----------
    halo_catalog : HaloCatalogData
        The halo catalog data containing position and other attributes of halos.

    box_size : int
        Size of the simulation box used for constructing the grid.

    n_grid : int
        Number of grid cells along each axis in the simulation box.

    r_min : float, optional
        Minimum radius value for filtering halos. If None, no lower limit is applied.

    r_max : float, optional
        Maximum radius value for filtering halos. If None, no upper limit is applied.

    Returns
    -------
    HaloCatalogData
        The formatted halo catalog data with adjusted positions and halos filtered
        based on the specified radius range if provided.
    """
    xyz_positions = [halo_catalog.x_position, halo_catalog.y_position, halo_catalog.z_position]
    halo_catalog.data[:, xyz_positions] = get_grid_cell(
        data=halo_catalog.data[:, xyz_positions], box_size=box_size, n_grid=n_grid
    )

    if r_min is not None and r_max is not None and halo_catalog.rg_position is not None:
        halo_selected = (halo_catalog.data[:, halo_catalog.rg_position] > r_min) & (
            halo_catalog.data[:, halo_catalog.rg_position] < r_max
        )
        logging.info(
            f"Number of halos in the selected radius: {np.sum(halo_selected)} of "
            f"{halo_selected.size}"
        )
        halo_catalog.data = halo_catalog.data[halo_selected]
    return halo_catalog


def _tidal_anisotropy_and_overdensity_from_halo_calaog_batches(
    output_path: str,
    tidal_tensor_array: TidalTensorArray,
    halo_catalog: HaloCatalogReader,
    halo_catalog_path: str,
    n_grid: int,
    box_size: int,
    batch_size: int,
    n_lines: Optional[int] = None,
    **save_params,
) -> str:
    """
    Processes batches of halo catalogs to compute tidal anisotropy and overdensity,
    then saves the results.

    This function iterates through a Rockstar halo catalog in batches, identifies halos within a
    specified radius range, and calculates tidal anisotropy and overdensity based on the tidal
    tensor array. Results are saved to text files for each batch.

    Parameters
    ----------
    output_path : str
        The file path where the results for tidal anisotropy and overdensity will be saved.
    tidal_tensor_array : TidalTensorArray
        Object containing the tidal tensor and related computation methods.
    halo_catalog: HaloCatalogReader,
        Halo catalog object containing methods for reading and formatting halo catalog data.
    halo_catalog_path : str
        Path to the halo catalog to be processed.
    n_grid : int
        Number of grid cells in one dimension of the simulation box.
    box_size : int
        Physical size of the simulation box.
    n_lines : Optional[int]
        Maximum number of halos to process. Iteration stops when this limit is reached.
    batch_size : int
        Number of halos to be processed in each batch.

    Returns
    -------
    str
        The output path where the batch results are saved.
    """
    r_min = np.array(tidal_tensor_array.gaussian_scale_list).min()
    r_max = np.array(tidal_tensor_array.gaussian_scale_list).max()

    if n_lines is None:
        n_lines = np.inf

    offset = 0
    batch_index = 0
    lines_readed = 0
    for halo_data, batch_start_offset, next_offset in halo_catalog.read_catalog_batch_generator(
        halo_catalog_path,
        batch_size=batch_size,
        start_offset=offset,
    ):
        logging.info(
            "Batch %i: %i halos, offset start=%i"
            % (batch_index, halo_data.n_halos, batch_start_offset)
        )
        halo_data = format_halo_catalog(halo_data, box_size, n_grid, r_min, r_max)
        tidal_anisotropy, overdensity = tidal_tensor_array.get_tidal_anisotropy_and_overdensity(
            halo_data.halo_x,
            halo_data.halo_y,
            halo_data.halo_z,
            halo_data.halo_rg,
        )

        halo_data.save_properties(
            path=output_path + f"{batch_index}_halo_environment_descriptors.txt",
            properties_data=(tidal_anisotropy, overdensity),
            properties_header=f"Tidal Anisotropy, Overdensity\n Halo catalog: {halo_catalog_path}",
            **save_params,
        )

        lines_readed += batch_size
        batch_index += 1
        offset = next_offset

        del halo_data, tidal_anisotropy, overdensity
        gc.collect()

        if lines_readed > n_lines:
            break

    return output_path


def _tidal_anisotropy_and_overdensity_from_halo_calaog_complete(
    output_path: str,
    tidal_tensor_array: TidalTensorArray,
    halo_catalog: HaloCatalogReader,
    halo_catalog_path: str,
    n_grid: int,
    box_size: int,
    n_lines: Optional[int] = None,
    **save_params,
) -> str:
    """
    Computes tidal anisotropy and overdensity from a given halo catalog, filters halos
    based on radius constraints, and saves the results to the specified output path.

    Parameters
    ----------
    output_path : str
        The directory path where the computed tidal anisotropy and overdensity results will
        be saved.
    tidal_tensor_array : TidalTensorArray
        An object containing methods and halo_catalog to compute tidal anisotropy and overdensity.
    halo_catalog: HaloCatalogReader,
        Halo catalog object containing methods for reading and formatting halo catalog data.
    halo_catalog_path : str
        Path to the halo catalog to be processed.
    n_grid : int
        Number of grid cells along one dimension in the simulation box.
    box_size : int
        The size of the simulation box.
    n_lines : Optional[int]
        Number of lines to be read from the halo catalog file. If None, all lines
        are read.
    xyz_position : Optional[list[int]] = None
        List of indices in the halo catalog corresponding to the x, y, and z
        spatial positions, respectively. If None, default indices are assumed.
    r_position : int, default=4
        Column index in the halo catalog specifying the radius or size of halos.

    Returns
    -------
    str
        The output path where results are saved.
    """
    r_min = np.array(tidal_tensor_array.gaussian_scale_list).min()
    r_max = np.array(tidal_tensor_array.gaussian_scale_list).max()

    halo_data = halo_catalog.read_catalog(path=halo_catalog_path, n_lines=n_lines)

    halo_data = format_halo_catalog(halo_data, box_size, n_grid, r_min, r_max)
    tidal_anisotropy, overdensity = tidal_tensor_array.get_tidal_anisotropy_and_overdensity(
        halo_data.halo_x,
        halo_data.halo_y,
        halo_data.halo_z,
        halo_data.halo_rg,
    )

    halo_data.save_properties(
        path=output_path + "halo_environment_descriptors.txt",
        properties_data=(tidal_anisotropy, overdensity),
        properties_header=f"Tidal Anisotropy, Overdensity\n Halo catalog: {halo_catalog_path}",
        **save_params,
    )

    return output_path


def tidal_anisotropy_and_overdensity_from_halo_calaog(
    path: str,
    halo_catalog: HaloCatalogReader,
    halo_catalog_path: str,
    n_grid: int,
    box_size: int,
    n_lines: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> str:
    """
    Calculates tidal anisotropy and overdensity from a halo catalog, iterating through
    batches and saving the results to output files.

    Parameters
    ----------
    path : str
        Path to the main directory where the tidal tensor is stored.
    halo_catalog: HaloCatalogReader,
        Halo catalog object containing methods for reading and formatting halo catalog data.
    halo_catalog_path : str
        Path to the halo catalog to be processed.
    n_grid : int
        Number of grid cells in one dimension of the simulation grid.
    box_size : int
        The physical size of the simulation box in Mpc/h units.
    n_lines : Optional[int]
        Number of lines to be read from the halo catalog file. If None, all lines
        are read.
    batch_size : Optional[int] = None
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
    tidal_path = os.path.join(path, f"{TIDAL_TENSOR_PATH}")
    if not os.path.exists(tidal_path):
        raise ValueError(
            "%s path doesn' exist. Before you need to calculate the Tidals Tensors." % tidal_path
        )

    tidal_tensor_array = TidalTensorArray.from_folder(path=tidal_path)
    output_path = os.path.join(path, f"{ANISOTROPY_PATH}/")
    os.makedirs(output_path, exist_ok=True)

    if batch_size is not None:
        return _tidal_anisotropy_and_overdensity_from_halo_calaog_batches(
            output_path=output_path,
            tidal_tensor_array=tidal_tensor_array,
            halo_catalog=halo_catalog,
            halo_catalog_path=halo_catalog_path,
            n_grid=n_grid,
            box_size=box_size,
            n_lines=n_lines,
            batch_size=batch_size,
        )

    return _tidal_anisotropy_and_overdensity_from_halo_calaog_complete(
        output_path=output_path,
        tidal_tensor_array=tidal_tensor_array,
        halo_catalog=halo_catalog,
        halo_catalog_path=halo_catalog_path,
        n_grid=n_grid,
        box_size=box_size,
        n_lines=n_lines,
    )
