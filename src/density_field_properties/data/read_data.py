import logging
from typing import Optional

import numpy as np

from density_field_properties.cosmology import Cosmology

ROCKSTAR_HALO_COLUMNS_POSITION = {
    "halo_id": 1,
    "halo_x": 17,
    "halo_y": 18,
    "halo_z": 19,
    "m200b": 39,
}


def read_rockstar_halo_catalog(
    path: str,
    n_lines: Optional[int] = None,
    halo_id_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"],
    halo_x_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"],
    halo_y_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"],
    halo_z_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"],
    halo_m200b_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["m200b"],
) -> np.ndarray:
    """
    Reads a Rockstar halo catalog from a file and extracts selected data.

    This function reads a Rockstar halo catalog from the specified file path,
    parses it line by line, and extracts selected attributes such as halo
    positions (x, y, z) and mass (m200b). It skips lines starting with
    a comment ('#') and stops once the specified number of lines has been
    processed, or the file ends. If the omega_matter is specified
    in the halo file, it returns R200b values instead of m200b.

    Parameters
    ----------
    path : str
        Path to the file containing the Rockstar halo catalog.
    n_lines : int, optional
        Number of lines to read and process from the file. If `None`, it will
        read the entire file.
    halo_id_position : int, optional
        Position of the halo ID in the file.
    halo_x_position : int, optional
        Position of the halo x-coordinate in the file.
    halo_y_position : int, optional
        Position of the halo y-coordinate in the file.
    halo_z_position : int, optional
        Position of the halo z-coordinate in the file.
    halo_m200b_position : int, optional
        Position of the halo M200b in the file.

    Returns
    -------
    np.ndarray
        A Array where each row corresponds to a halo and contains
        the following columns:
        - `id` (int): Halo ID.
        - `halo_x` (float): x-coordinate of the halo.
        - `halo_y` (float): y-coordinate of the halo.
        - `halo_z` (float): z-coordinate of the halo.
        - `m200b` (float): Mass (m200b) of the halo. If omega_matter is specified
        , this value will be converted to 4 * r200b / sqrt(5).
    """
    results = []
    cosmology = None

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#Omega_M"):
                parameters = {}
                for param_line in line[1:].split(";"):
                    param = param_line.split("=")[0].strip().replace(" ", "_").lower()
                    value = float(param_line.split("=")[1].strip())
                    if param.startswith("omega_m"):
                        param = "omega_matter"
                    if param.startswith("omega_l"):
                        param = "omega_lambda"
                    parameters[param] = value
                try:
                    cosmology = Cosmology(**parameters)
                except TypeError:
                    logging.warning(f"Could not parse parameter {list(parameters.keys())}.")
                continue

            elif line.startswith("#"):
                continue
            cols = line.split()

            halo_id = int(cols[halo_id_position])
            halo_x = float(cols[halo_x_position])
            halo_y = float(cols[halo_y_position])
            halo_z = float(cols[halo_z_position])
            m200b = float(cols[halo_m200b_position])

            results.append((halo_id, halo_x, halo_y, halo_z, m200b))

            if len(results) >= n_lines:
                break

    results = np.array(results)
    if cosmology is None:
        return results

    results[:, 4] = 4 * cosmology.convert_m200b_to_r200b(results[:, 4]) / np.sqrt(5)
    return results
