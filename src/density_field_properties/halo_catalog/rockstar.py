import bz2
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Optional, TextIO, Tuple

import numpy as np

from density_field_properties.cosmology import Cosmology
from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData, HaloCatalogReader

ROCKSTAR_HALO_COLUMNS_POSITION = {
    "halo_id": 1,
    "halo_x": 17,
    "halo_y": 18,
    "halo_z": 19,
    "m200b": 39,
}

_COSMOLOGY_REQUIRED_FIELDS = ("omega_matter", "omega_lambda", "h0")
_ROCKSTAR_COSMOLOGY_ALIASES = {
    "omega_matter": frozenset({"om", "omega_m", "omega_matter"}),
    "omega_lambda": frozenset({"ol", "omega_l", "omega_lambda"}),
    "h0": frozenset({"h", "h0"}),
}


@contextmanager
def _open_text_catalog(path: str) -> Iterator[TextIO]:
    """
    Open a Rockstar catalog as a text stream, including ``*.list.bz2`` files.

    Parameters
    ----------
    path : str
        Catalog path.

    Yields
    ------
    Iterator[str]
        Open text handle for line iteration.
    """
    if Path(path).suffix == ".bz2":
        with bz2.open(path, "rt") as handle:
            yield handle
    else:
        with open(path, "r") as handle:
            yield handle


def _parse_box_size_from_header_line(line: str) -> Optional[float]:
    """
    Parse a Rockstar header line for the box side length in Mpc/h.

    Parameters
    ----------
    line : str
        Header line starting with ``#``.

    Returns
    -------
    Optional[float]
        Box size in Mpc/h, or ``None`` if not present.
    """
    if not line.startswith("#"):
        return None
    body = line[1:]
    for segment in body.split(";"):
        segment = segment.strip()
        if "=" in segment:
            key, value = segment.split("=", 1)
        elif ":" in segment:
            key, value = segment.split(":", 1)
        else:
            continue
        normalized = key.strip().replace(" ", "_").lower()
        if normalized in {"box_size", "boxsize", "box"}:
            token = value.strip().split()[0]
            return float(token)
    return None


def _normalize_rockstar_cosmology_field(raw_key: str) -> Optional[str]:
    """
    Map a Rockstar header key to a ``Cosmology`` constructor field name.

    Parameters
    ----------
    raw_key : str
        Key text before ``=`` in a ``#`` header segment.

    Returns
    -------
    Optional[str]
        Canonical field name, or ``None`` if the key is not recognized.
    """
    token = raw_key.strip().replace(" ", "_").lower()
    if token.startswith("omega_m"):
        return "omega_matter"
    if token.startswith("omega_l"):
        return "omega_lambda"
    for field, aliases in _ROCKSTAR_COSMOLOGY_ALIASES.items():
        if token in aliases:
            return field
    return None


def _parse_rockstar_hash_line_cosmology(line: str) -> dict[str, float]:
    """
    Extract cosmology key-value pairs from a single Rockstar ``#`` comment line.

    Parameters
    ----------
    line : str
        One header line, typically starting with ``#``.

    Returns
    -------
    dict[str, float]
        Parsed canonical cosmology fields and numeric values.
    """
    parameters = {}
    body = line[1:] if line.startswith("#") else line
    for segment in body.split(";"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        field = _normalize_rockstar_cosmology_field(key)
        if field is None:
            continue
        parameters[field] = float(value.strip())
    return parameters


def read_rockstar_box_size_header(path: str) -> Optional[float]:
    """
    Parse the Rockstar ``Box size`` header value in Mpc/h.

    Parameters
    ----------
    path : str
        Path to a Rockstar ``.list`` catalog.

    Returns
    -------
    Optional[float]
        Box side length in Mpc/h, or ``None`` if not found.
    """
    with _open_text_catalog(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            box_size = _parse_box_size_from_header_line(line)
            if box_size is not None:
                return box_size
    return None


def read_rockstar_cosmology_header(path: str) -> Optional[Cosmology]:
    """
    Reads a Rockstar cosmology header file and extracts cosmological parameters.

    This function parses the contents of a Rockstar header file to retrieve
    cosmological parameters such as omega matter and omega lambda. The parameters
    are extracted from the header section marked with '#', and the corresponding
    Cosmology object is returned based on the parsed values. If the parsing
    encounters unrecognized or invalid structures, a warning is logged.

    Parameters
    ----------
    path : str
        Path to the Rockstar cosmology header file.

    Returns
    -------
    Optional[Cosmology]
        A `Cosmology` instance populated with parsed parameters from the file,
        or None if the file could not be parsed successfully.
    """
    merged = {}

    with _open_text_catalog(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            merged.update(_parse_rockstar_hash_line_cosmology(line))

    if not all(field in merged for field in _COSMOLOGY_REQUIRED_FIELDS):
        return None

    try:
        return Cosmology(
            omega_matter=merged["omega_matter"],
            omega_lambda=merged["omega_lambda"],
            h0=merged["h0"],
        )
    except TypeError:
        logging.warning(
            "Could not parse parameter(s) %s from cosmology header.",
            list(merged.keys()),
        )
        return None


class RockstarCatalogReader(HaloCatalogReader):
    calog_name: str = "rockstar"

    @staticmethod
    def read_catalog(
        path: str,
        n_lines: Optional[int] = None,
        halo_id_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"],
        halo_x_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"],
        halo_y_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"],
        halo_z_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"],
        halo_m200b_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["m200b"],
    ) -> HaloCatalogData:
        """
        Reads a Rockstar halo catalog from a file and extracts selected halo_catalog.

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
        n_lines : Optional[int] = None
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
        HaloCatalogData
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
        cosmology = read_rockstar_cosmology_header(path)
        positions = (0, 1, 2, 3, 4)
        if isinstance(cosmology, Cosmology):
            positions = (0, 1, 2, 3, 4, 5)

        with _open_text_catalog(path) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                cols = line.split()

                halo_id = int(cols[halo_id_position])
                halo_x = float(cols[halo_x_position])
                halo_y = float(cols[halo_y_position])
                halo_z = float(cols[halo_z_position])
                m200b = float(cols[halo_m200b_position])

                row = (halo_id, halo_x, halo_y, halo_z, m200b)
                if isinstance(cosmology, Cosmology):
                    rg = 4.0 * cosmology.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
                    row = (halo_id, halo_x, halo_y, halo_z, m200b, rg)

                results.append(row)
                if n_lines is not None and len(results) >= n_lines:
                    break

        results = HaloCatalogData(np.array(results), *positions)
        return results

    @staticmethod
    def read_catalog_batch_generator(
        path: str,
        batch_size: int,
        start_offset: int = 0,
        halo_id_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"],
        halo_x_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"],
        halo_y_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"],
        halo_z_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"],
        halo_m200b_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["m200b"],
    ) -> Generator[Tuple[HaloCatalogData, int, int], None, None]:
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
        batch : HaloCatalogData
            Array de shape (N, 5) con columnas:
                [halo_id, halo_x, halo_y, halo_z, m200b_o_4r200b_sqrt5].
        batch_start_offset : int
            Offset en bytes del fichero donde comienza la primera línea
            de datos de este batch.
        next_offset : int
            Offset en bytes justo después de la última línea leída del batch.
            Útil para reanudar el procesado en otro momento.
        """
        cosmology = read_rockstar_cosmology_header(path)
        positions = (0, 1, 2, 3, 4)
        if isinstance(cosmology, Cosmology):
            positions = (0, 1, 2, 3, 4, 5)

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
                if isinstance(cosmology, Cosmology):
                    rg = 4.0 * cosmology.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
                    row = (halo_id, halo_x, halo_y, halo_z, m200b, rg)

                results.append(row)

                if len(results) == batch_size:
                    batch = np.array(results, dtype=float)
                    batch = HaloCatalogData(batch, *positions)
                    next_offset = f.tell()
                    yield batch, batch_start_offset, next_offset
                    results = []
                    batch_start_offset = None

            if results:
                batch = np.array(results, dtype=float)
                batch = HaloCatalogData(batch, *positions)
                if batch_start_offset is None:
                    batch_start_offset = start_offset
                next_offset = f.tell()
                yield batch, batch_start_offset, next_offset
