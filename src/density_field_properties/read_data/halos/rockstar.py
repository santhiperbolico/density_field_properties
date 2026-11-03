import bz2
import logging
import re
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np

from density_field_properties.cosmology import Cosmology
from density_field_properties.read_data.halos.base import HaloCatalogData, HaloCatalogReader
from density_field_properties.read_data.halos.column_maps import ROCKSTAR_LIST_COLUMNS

ROCKSTAR_HALO_COLUMNS_POSITION = {
    "halo_id": 1,
    "halo_x": 17,
    "halo_y": 18,
    "halo_z": 19,
    "pid": 5,
    "m200b": 39,
}

_COSMOLOGY_REQUIRED_FIELDS = ("omega_matter", "omega_lambda", "h0")
_ROCKSTAR_COSMOLOGY_ALIASES = {
    "omega_matter": frozenset({"om", "omega_m", "omega_matter"}),
    "omega_lambda": frozenset({"ol", "omega_l", "omega_lambda"}),
    "h0": frozenset({"h", "h0"}),
}


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


def _open_rockstar_catalog_file(path: str):
    """
    Open a Rockstar catalog path, including ``.bz2`` compressed hlists.

    Parameters
    ----------
    path : str
        Catalog file path.

    Returns
    -------
    file
        Text-mode file object.
    """
    if path.endswith(".bz2"):
        return bz2.open(path, "rt")
    return open(path, "r")


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

    with _open_rockstar_catalog_file(path) as handle:
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


_HLIST_FIELD_TOKEN = re.compile(r"^(?P<name>.+)\((?P<index>\d+)\)$", re.IGNORECASE)
_HLIST_SCHEMA_START = re.compile(r"^\w+\(\d+\)$", re.IGNORECASE)


def _is_hlist_schema_header_line(tokens: list[str]) -> bool:
    """
    Return whether tokens form a consistent-trees ``name(index)`` schema line.

    Parameters
    ----------
    tokens : list[str]
        Header tokens after the leading ``#``.

    Returns
    -------
    bool
        True when the first token matches ``scale(0)``-style hlist fields.
    """
    return bool(tokens) and _HLIST_SCHEMA_START.match(tokens[0]) is not None


def _parse_hlist_field_token(token: str) -> Optional[tuple[str, int]]:
    """
    Parse one consistent-trees header token into a field name and column index.

    Parameters
    ----------
    token : str
        Token such as ``pid(5)`` or ``M200b(39)``.

    Returns
    -------
    Optional[tuple[str, int]]
        Lowercase field name and 0-based index, or ``None`` when not matched.
    """
    match = _HLIST_FIELD_TOKEN.match(token)
    if match is None:
        return None
    return match.group("name").lower(), int(match.group("index"))


def _rockstar_first_data_row_width(path: str) -> Optional[int]:
    """
    Return the number of whitespace-separated fields in the first data row.

    Parameters
    ----------
    path : str
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.

    Returns
    -------
    Optional[int]
        Column count for the first non-comment row, or ``None`` when missing.
    """
    with _open_rockstar_catalog_file(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            return len(line.split())
    return None


def read_rockstar_id_header_column_names(path: str) -> Optional[list[str]]:
    """
    Return column names from the catalog schema header line.

    Supports Rockstar ``#ID ...`` lines and consistent-trees ``#scale(0) id(1) ...``.

    Parameters
    ----------
    path : str
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.

    Returns
    -------
    Optional[list[str]]
        Header tokens after the leading ``#``, or ``None`` when no schema line exists.
    """
    with _open_rockstar_catalog_file(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            tokens = line[1:].strip().split()
            if not tokens:
                continue
            if tokens[0].upper() == "ID":
                return tokens
            if _is_hlist_schema_header_line(tokens):
                return tokens
    return None


def rockstar_pid_column_index(path: str) -> Optional[int]:
    """
    Return the 0-based parent-halo column index when the catalog header defines it.

    Parameters
    ----------
    path : str
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.

    Returns
    -------
    Optional[int]
        Index of ``PID`` / ``pid`` in data rows, or ``None`` when absent.
    """
    header = read_rockstar_id_header_column_names(path)
    if header is None:
        return None
    if _is_hlist_schema_header_line(header):
        for token in header:
            parsed = _parse_hlist_field_token(token)
            if parsed is not None and parsed[0] == "pid":
                return parsed[1]
        return None
    try:
        header_pid = header.index("PID")
    except ValueError:
        return None
    data_width = _rockstar_first_data_row_width(path)
    short_pid_index = ROCKSTAR_LIST_COLUMNS["halo_pid"]
    if data_width is not None and data_width <= header_pid:
        return short_pid_index
    return header_pid


def resolve_rockstar_pid_filter(
    path: str,
    halo_pid_position: int,
    filter_by_pid: bool,
) -> tuple[bool, int]:
    """
    Decide whether to filter subhalos and which column holds the parent ID.

    When a ``#ID`` header line exists but omits ``PID`` (typical FastPM ``out_*.list``
    files) and ``filter_by_pid`` is True, raises ``ValueError``. Pass
    ``filter_by_pid=False`` for those catalogs. Catalogs without a ``#ID`` line keep
    the caller-supplied ``halo_pid_position``.

    Parameters
    ----------
    path : str
        Path to a Rockstar catalog file.
    halo_pid_position : int
        Default zero-based PID column index from layout maps.
    filter_by_pid : bool
        Whether the caller requested host-only filtering.

    Returns
    -------
    tuple[bool, int]
        Effective ``filter_by_pid`` flag and PID column index for data rows.
    """
    if not filter_by_pid:
        return False, halo_pid_position

    header_pid = rockstar_pid_column_index(path)
    if header_pid is not None:
        return True, header_pid

    if read_rockstar_id_header_column_names(path) is None:
        logging.warning(
            "Rockstar catalog %s has no column names; using PID index %i.",
            Path(path).name,
            halo_pid_position,
        )
        return True, halo_pid_position

    raise ValueError(
        f"Rockstar catalog {Path(path).name} has no PID column; "
        "cannot filter hosts with filter_by_pid=True."
    )


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
        halo_pid_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["pid"],
        filter_by_pid: bool = True,
        extra_column_positions: Optional[dict[str, int]] = None,
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
        halo_pid_position : int, optional
            Position of the halo PID in the file.
        filter_by_pid : bool, optional
            Whether to filter halos by PID.
        extra_column_positions : Optional[dict[str, int]], optional
            Optional mapping of output column names to file column indices.
            Parsed values are stored on ``HaloCatalogData.extra_columns``.

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
        extra_lists = (
            {name: [] for name in extra_column_positions}
            if extra_column_positions is not None
            else None
        )
        cosmology = read_rockstar_cosmology_header(path)
        positions = (0, 1, 2, 3, 4)
        if isinstance(cosmology, Cosmology):
            positions = (0, 1, 2, 3, 4, 5)

        apply_pid_filter, pid_column = resolve_rockstar_pid_filter(
            path,
            halo_pid_position,
            filter_by_pid,
        )

        with _open_rockstar_catalog_file(path) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                cols = line.split()

                halo_id = int(cols[halo_id_position])
                halo_x = float(cols[halo_x_position])
                halo_y = float(cols[halo_y_position])
                halo_z = float(cols[halo_z_position])
                m200b = float(cols[halo_m200b_position])
                if apply_pid_filter:
                    pid = int(cols[pid_column])
                    if pid > -1:
                        continue

                row = (halo_id, halo_x, halo_y, halo_z, m200b)
                if isinstance(cosmology, Cosmology):
                    rg = 4.0 * cosmology.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
                    row = (halo_id, halo_x, halo_y, halo_z, m200b, rg)

                results.append(row)
                if extra_lists is not None:
                    for name, column_index in extra_column_positions.items():
                        extra_lists[name].append(float(cols[column_index]))
                if n_lines is not None and len(results) >= n_lines:
                    break

        extra_columns = None
        if extra_lists is not None:
            extra_columns = {
                name: np.array(values, dtype=np.float64) for name, values in extra_lists.items()
            }
        return HaloCatalogData(np.array(results), *positions, extra_columns=extra_columns)

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
        halo_pid_position: int = ROCKSTAR_HALO_COLUMNS_POSITION["pid"],
        filter_by_pid: bool = True,
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
        halo_pid_position : int, optional
            Position of the halo PID in the file.
        filter_by_pid : bool, optional
            Whether to filter halos by PID.
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

        apply_pid_filter, pid_column = resolve_rockstar_pid_filter(
            path,
            halo_pid_position,
            filter_by_pid,
        )

        with _open_rockstar_catalog_file(path) as handle:
            handle.seek(start_offset)
            results = []
            batch_start_offset: Optional[int] = None

            while True:
                line_offset = handle.tell()
                line = handle.readline()
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
                if apply_pid_filter:
                    pid = int(cols[pid_column])
                    if pid > -1:
                        continue
                row = (halo_id, halo_x, halo_y, halo_z, m200b)
                if isinstance(cosmology, Cosmology):
                    rg = 4.0 * cosmology.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
                    row = (halo_id, halo_x, halo_y, halo_z, m200b, rg)

                results.append(row)

                if len(results) == batch_size:
                    batch = np.array(results, dtype=float)
                    batch = HaloCatalogData(batch, *positions)
                    next_offset = handle.tell()
                    yield batch, batch_start_offset, next_offset
                    results = []
                    batch_start_offset = None

            if results:
                batch = np.array(results, dtype=float)
                batch = HaloCatalogData(batch, *positions)
                if batch_start_offset is None:
                    batch_start_offset = start_offset
                next_offset = handle.tell()
                yield batch, batch_start_offset, next_offset
