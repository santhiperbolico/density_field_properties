import logging
from typing import Generator, Optional, Tuple

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
        - `rg` (float): If omega_matter is specified 4 * r200b / sqrt(5).
        - `m200b` (float): Mass (m200b) of the halo.
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

            row = (halo_id, halo_x, halo_y, halo_z, m200b)
            if cosmology is not None:
                rg = 4.0 * cosmology.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
                row = (halo_id, halo_x, halo_y, halo_z, rg, m200b)

            results.append(row)

            if len(results) >= n_lines:
                break

    results = np.array(results)
    return results


def _read_rockstar_cosmology_header(path: str) -> Optional[Cosmology]:
    """
    Lee solo las líneas de cabecera del catálogo de Rockstar para extraer la cosmología.
    No lee ninguna línea de datos.
    """
    cosmology = None

    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                # En cuanto vemos la primera línea de datos, paramos.
                break

            if line.startswith("#Omega_M"):
                parameters = {}
                for param_line in line[1:].split(";"):
                    if "=" not in param_line:
                        continue
                    key, value = param_line.split("=", 1)
                    param = key.strip().replace(" ", "_").lower()
                    value = float(value.strip())
                    if param.startswith("omega_m"):
                        param = "omega_matter"
                    if param.startswith("omega_l"):
                        param = "omega_lambda"
                    parameters[param] = value

                try:
                    cosmology = Cosmology(**parameters)
                except TypeError:
                    logging.warning(
                        "Could not parse parameter(s) %s from cosmology header.",
                        list(parameters.keys()),
                    )
                # Aunque hubiera varias líneas similares, con una basta.
    return cosmology


def iter_rockstar_halo_catalog_by_offset(
    path: str,
    batch_size: int,
    start_offset: int = 0,
    halo_id_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"],
    halo_x_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"],
    halo_y_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"],
    halo_z_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"],
    halo_m200b_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["m200b"],
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Iterador sobre un catálogo de halos de Rockstar que devuelve batches
    y permite reanudar eficientemente usando offsets de bytes.

    A diferencia de la versión por número de línea, aquí:
      - Podemos empezar directamente en `start_offset` con `f.seek(start_offset)`.
      - No leemos las líneas anteriores de datos.
      - Solo se lee la cabecera una vez (en una pasada muy corta) para obtener la cosmología.

    Parameters
    ----------
    path : str
        Ruta al fichero del catálogo de halos de Rockstar.
    batch_size : int
        Número de halos por batch.
    start_offset : int, optional
        Offset en bytes dentro del fichero donde comenzar a leer halos.
        Se puede usar el `next_offset` devuelto en una iteración anterior
        para reanudar exactamente donde lo dejamos.
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

    Yields
    ------
    batch : np.ndarray
        Array de shape (N, 5) con columnas:
            [halo_id, halo_x, halo_y, halo_z, m200b_o_4r200b_sqrt5].
    batch_start_offset : int
        Offset en bytes del fichero donde comienza la primera línea
        de datos de este batch.
    next_offset : int
        Offset en bytes justo después de la última línea leída del batch.
        Útil para reanudar el procesado en otro momento.
    """
    cosmology = _read_rockstar_cosmology_header(path)

    with open(path, "r") as f:
        f.seek(start_offset)

        results = []
        batch_start_offset: Optional[int] = None

        while True:
            line_offset = f.tell()
            line = f.readline()
            if not line:
                break

            if line.startswith("#"):
                continue

            if batch_start_offset is None:
                batch_start_offset = line_offset

            cols = line.split()
            halo_id = int(cols[halo_id_position])
            halo_x = float(cols[halo_x_position])
            halo_y = float(cols[halo_y_position])
            halo_z = float(cols[halo_z_position])
            m200b = float(cols[halo_m200b_position])
            row = (halo_id, halo_x, halo_y, halo_z, m200b)
            if cosmology is not None:
                rg = 4.0 * cosmology.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
                row = (halo_id, halo_x, halo_y, halo_z, rg, m200b)

            results.append(row)

            if len(results) == batch_size:
                batch = np.array(results, dtype=float)

                next_offset = f.tell()

                yield batch, batch_start_offset, next_offset

                results = []
                batch_start_offset = None

        if results:
            batch = np.array(results, dtype=float)

            if batch_start_offset is None:
                batch_start_offset = start_offset

            next_offset = f.tell()
            yield batch, batch_start_offset, next_offset
