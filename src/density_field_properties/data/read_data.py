from typing import Optional

import numpy as np

ROCKSTAR_HALO_COLUMNS_POSITION = {
    "halo_x": 17,
    "halo_y": 18,
    "halo_z": 19,
    "m200b": 39,
}


def read_rockstar_halo_catalog(
    path: str,
    n_lines: Optional[int] = None,
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
    processed, or the file ends.

    Parameters
    ----------
    path : str
        Path to the file containing the Rockstar halo catalog.
    n_lines : int, optional
        Number of lines to read and process from the file. If `None`, it will
        read the entire file.
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
        - `halo_x` (float): x-coordinate of the halo.
        - `halo_y` (float): y-coordinate of the halo.
        - `halo_z` (float): z-coordinate of the halo.
        - `m200b` (float): Mass (m200b) of the halo.
    """
    results = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split()

            halo_x = float(cols[halo_x_position])
            halo_y = float(cols[halo_y_position])
            halo_z = float(cols[halo_z_position])
            m200b = float(cols[halo_m200b_position])

            results.append((halo_x, halo_y, halo_z, m200b))

            if len(results) >= n_lines:
                break
    return np.array(results)
