"""Load UNIT consistent-trees and FastPM Rockstar catalogs into pandas."""

import bz2
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from density_field_properties.halo_catalog.rockstar import RockstarCatalogReader
from density_field_properties.haloscope.sim_to_fastpm.config import (
    EXTENDED_ROCKSTAR_MIN_COLUMNS,
    ROCKSTAR_LIST_COLUMNS,
    ROCKSTAR_RESERVOIR_SEED,
    UNIT_HLIST_COLUMNS,
    UNIT_ROCKSTAR_LIST_COLUMNS,
)


def _read_whitespace_table_lines(
    path: Path, max_data_rows: Optional[int] = None
) -> list[list[str]]:
    """
    Read non-comment whitespace-separated rows from a text or bz2 catalog file.

    Parameters
    ----------
    path : Path
        Catalog file path.
    max_data_rows : Optional[int], optional
        Stop after this many data rows; ``None`` reads the full file.

    Returns
    -------
    list[list[str]]
        Tokenized rows (column indices match Rockstar / consistent-trees headers).
    """
    opener = bz2.open if path.suffix == ".bz2" else open
    mode = "rt" if path.suffix == ".bz2" else "r"
    rows: list[list[str]] = []
    with opener(path, mode) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            rows.append(line.split())
            if max_data_rows is not None and len(rows) >= max_data_rows:
                break
    return rows


def _rows_to_frame(rows: list[list[str]], column_map: dict[str, int]) -> pd.DataFrame:
    """
    Build a DataFrame from tokenized rows using 0-based column indices.

    Parameters
    ----------
    rows : list[list[str]]
        Data rows from a catalog file.
    column_map : dict[str, int]
        Mapping from output column name to 0-based index in each row.

    Returns
    -------
    pd.DataFrame
        Selected columns cast to float where appropriate.
    """
    data = {}
    for name, index in column_map.items():
        values = [row[index] for row in rows]
        if name in ("id", "pid", "halo_id"):
            data[name] = np.array(values, dtype=np.int64)
        else:
            data[name] = np.array(values, dtype=np.float64)
    return pd.DataFrame(data)


def _rockstar_data_column_count(list_path: Path) -> int:
    """
    Return the number of whitespace-separated columns in the first data row.

    Parameters
    ----------
    list_path : Path
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.

    Returns
    -------
    int
        Column count of the first data row, or zero if the file has no data rows.
    """
    path = Path(list_path)
    opener = bz2.open if path.suffix == ".bz2" else open
    mode = "rt" if path.suffix == ".bz2" else "r"
    with opener(path, mode) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            return len(line.split())
    return 0


def _rockstar_pid_column_index(list_path: Path, pid_index: int) -> Optional[int]:
    """
    Return the PID column index when the catalog uses the extended Rockstar layout.

    Compact FastPM ``.list`` files have about 34 columns and no host/subhalo PID field.
    Extended catalogs with at least ``EXTENDED_ROCKSTAR_MIN_COLUMNS`` columns include
    ``PID`` at the given index (host halos have ``PID == -1``).

    Parameters
    ----------
    list_path : Path
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.
    pid_index : int
        Expected 0-based PID column index in extended catalogs.

    Returns
    -------
    Optional[int]
        ``pid_index`` for extended catalogs, otherwise ``None``.
    """
    column_count = _rockstar_data_column_count(list_path)
    if column_count >= EXTENDED_ROCKSTAR_MIN_COLUMNS:
        return pid_index
    return None


def _reservoir_sample_halo(
    reservoir: list[tuple[float, float, float, float]],
    sample_size: int,
    seen_count: int,
    position_x: float,
    position_y: float,
    position_z: float,
    mass: float,
    rng: np.random.Generator,
) -> None:
    """
    Update a fixed-size reservoir with one accepted halo row.

    Parameters
    ----------
    reservoir : list[tuple[float, float, float, float]]
        In-place reservoir of ``(x, y, z, M200b)`` tuples.
    sample_size : int
        Target reservoir capacity.
    seen_count : int
        One-based count of accepted halos seen so far in the stream.
    position_x : float
        Halo x coordinate in Mpc/h.
    position_y : float
        Halo y coordinate in Mpc/h.
    position_z : float
        Halo z coordinate in Mpc/h.
    mass : float
        Halo ``M200b`` in Msun/h.
    rng : np.random.Generator
        Random generator for uniform reservoir updates.
    """
    entry = (position_x, position_y, position_z, mass)
    if seen_count <= sample_size:
        reservoir.append(entry)
        return
    replace_index = int(rng.integers(0, seen_count))
    if replace_index < sample_size:
        reservoir[replace_index] = entry


def _halos_to_dataframe(halos: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """
    Build a halo DataFrame from ``(x, y, z, M200b)`` tuples.

    Parameters
    ----------
    halos : list[tuple[float, float, float, float]]
        Accepted halo records.

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z``, and ``M200b``.
    """
    if not halos:
        raise ValueError("Cannot build an empty halo DataFrame")
    array = np.array(halos, dtype=np.float64)
    return pd.DataFrame(
        {
            "x": array[:, 0],
            "y": array[:, 1],
            "z": array[:, 2],
            "M200b": array[:, 3],
        }
    )


def _collect_rockstar_halos(
    list_path: Path,
    column_map: dict[str, int],
    max_halos: Optional[int],
    central_id_column: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream a Rockstar catalog and collect halo positions and ``M200b``.

    Parameters
    ----------
    list_path : Path
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.
    column_map : dict[str, int]
        Mapping with ``halo_x``, ``halo_y``, ``halo_z``, and ``halo_m200b`` indices.
    max_halos : Optional[int]
        When set, keep a uniform random subset of this size via reservoir
        sampling over the full catalog stream; ``None`` reads every accepted row.
    central_id_column : Optional[int], optional
        When set, keep only rows with this column equal to ``-1`` (host halos).

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z``, and ``M200b``.

    Raises
    ------
    ValueError
        If no halos are found before end-of-file.
    """
    path = Path(list_path)
    opener = bz2.open if path.suffix == ".bz2" else open
    mode = "rt" if path.suffix == ".bz2" else "r"
    x_index = column_map["halo_x"]
    y_index = column_map["halo_y"]
    z_index = column_map["halo_z"]
    mass_index = column_map["halo_m200b"]

    reservoir: list[tuple[float, float, float, float]] = []
    rng = np.random.default_rng(ROCKSTAR_RESERVOIR_SEED)
    seen_count = 0

    with opener(path, mode) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.split()
            if central_id_column is not None and int(columns[central_id_column]) != -1:
                continue
            mass = float(columns[mass_index])
            if mass <= 0.0:
                continue
            position_x = float(columns[x_index])
            position_y = float(columns[y_index])
            position_z = float(columns[z_index])
            seen_count += 1
            if max_halos is None:
                reservoir.append((position_x, position_y, position_z, mass))
                continue
            _reservoir_sample_halo(
                reservoir,
                max_halos,
                seen_count,
                position_x,
                position_y,
                position_z,
                mass,
                rng,
            )

    if seen_count == 0:
        raise ValueError(f"No halos found in catalog {list_path}")

    return _halos_to_dataframe(reservoir)


def load_unit_sim_training_catalog(
    hlist_path: Path,
    max_halos: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load UNIT (consistent-trees) halos for Haloscope training.

    Parameters
    ----------
    hlist_path : Path
        Path to ``hlist_*.list`` or ``hlist_*.list.bz2``.
    max_halos : Optional[int], optional
        Limit number of data rows read for development runs.

    Returns
    -------
    pd.DataFrame
        Host halos with ``cv``, positions, spins, shapes, and ``M200b``.
    """
    rows = _read_whitespace_table_lines(hlist_path, max_data_rows=max_halos)
    frame = _rows_to_frame(rows, UNIT_HLIST_COLUMNS)
    frame = frame.rename(columns={"id": "halo_id"})
    frame["cv"] = frame["Rvir"] / frame["Rs_Klypin"]
    hosts = frame[(frame["pid"] == -1) & (frame["M200b"] > 0) & np.isfinite(frame["cv"])].copy()
    hosts = hosts.rename(
        columns={
            "halo_id": "id",
            "ba": "ba",
            "ca": "ca",
        }
    )
    return hosts.reset_index(drop=True)


def rockstar_halo_catalog_to_dataframe(
    path: Path,
    n_lines: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a Rockstar ``.list`` file via ``RockstarCatalogReader``.

    Parameters
    ----------
    path : Path
        Path to ``out_*.list``.
    n_lines : Optional[int], optional
        Maximum halos to read.

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z``, ``M200b`` (and ``id`` if present in reader).
    """
    catalog = RockstarCatalogReader.read_catalog(
        str(path),
        n_lines=n_lines,
        halo_id_position=ROCKSTAR_LIST_COLUMNS["halo_id"],
        halo_x_position=ROCKSTAR_LIST_COLUMNS["halo_x"],
        halo_y_position=ROCKSTAR_LIST_COLUMNS["halo_y"],
        halo_z_position=ROCKSTAR_LIST_COLUMNS["halo_z"],
        halo_m200b_position=ROCKSTAR_LIST_COLUMNS["halo_m200b"],
    )
    frame = pd.DataFrame(
        {
            "x": catalog.halo_x,
            "y": catalog.halo_y,
            "z": catalog.halo_z,
            "M200b": catalog.halo_m200b,
        }
    )
    if catalog.m200b_position is not None:
        frame["id"] = catalog.halo_id.astype(np.int64)
    return frame


def load_fastpm_target_catalog(
    list_path: Path,
    max_halos: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load FastPM Rockstar halos to enrich (positions and mass only).

    Parameters
    ----------
    list_path : Path
        Path to ``out_*.list`` under ``rockstar_out_pm``.
    max_halos : Optional[int], optional
        Subset size for quick runs.

    Returns
    -------
    pd.DataFrame
        Positive-mass halos with ``x``, ``y``, ``z``, ``M200b``.
    """
    frame = rockstar_halo_catalog_to_dataframe(list_path, n_lines=max_halos)
    frame = frame[frame["M200b"] > 0].reset_index(drop=True)
    return frame


def load_fastpm_central_target_catalog(
    list_path: Path,
    max_centrals: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load FastPM halos for IC or enrichment checks.

    When the catalog has the extended Rockstar layout (``>= 55`` columns), keep
    host halos with ``PID == -1``. Compact FastPM ``.list`` files lack ``PID``;
    in that case all positive-mass halos are returned and a warning is logged.

    Parameters
    ----------
    list_path : Path
        Path to ``out_*.list`` under ``rockstar_out_pm``.
    max_centrals : Optional[int], optional
        Maximum number of halos to collect with uniform reservoir sampling;
        ``None`` reads the full catalog.

    Returns
    -------
    pd.DataFrame
        Halos with ``x``, ``y``, ``z``, and ``M200b``.
    """
    pid_column = _rockstar_pid_column_index(list_path, ROCKSTAR_LIST_COLUMNS["pid"])
    if pid_column is None:
        logging.warning(
            "FastPM catalog %s has no PID column; loading all halos with M200b > 0.",
            list_path,
        )
    return _collect_rockstar_halos(
        list_path,
        ROCKSTAR_LIST_COLUMNS,
        max_halos=max_centrals,
        central_id_column=pid_column,
    )


def load_unit_rockstar_target_catalog(
    list_path: Path,
    max_centrals: Optional[int] = None,
    central_only: bool = True,
) -> pd.DataFrame:
    """
    Load UNIT halos from a Rockstar ``out_*p.list.bz2``.

    Parameters
    ----------
    list_path : Path
        Path to a compressed UNIT Rockstar catalog at scale factor ``a = 1``.
    max_centrals : Optional[int], optional
        Maximum number of halos to collect with uniform reservoir sampling;
        ``None`` reads the full catalog.
    central_only : bool, optional
        When True, keep host halos with ``PID == -1`` only.

    Returns
    -------
    pd.DataFrame
        Halos with ``x``, ``y``, ``z``, and ``M200b``.
    """
    central_column = UNIT_ROCKSTAR_LIST_COLUMNS["pid"] if central_only else None
    return _collect_rockstar_halos(
        list_path,
        UNIT_ROCKSTAR_LIST_COLUMNS,
        max_halos=max_centrals,
        central_id_column=central_column,
    )
