"""Load UNIT consistent-trees and FastPM Rockstar catalogs into pandas."""

import bz2
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from density_field_properties.halo_catalog.rockstar import RockstarCatalogReader
from density_field_properties.haloscope.sim_to_fastpm.config import (
    ROCKSTAR_LIST_COLUMNS,
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


def _collect_rockstar_central_halos(
    list_path: Path,
    column_map: dict[str, int],
    central_id_column: int,
    max_centrals: int,
) -> pd.DataFrame:
    """
    Stream a Rockstar catalog and collect central host halos (``central_id == -1``).

    Parameters
    ----------
    list_path : Path
        Path to a Rockstar ``.list`` or ``.list.bz2`` file.
    column_map : dict[str, int]
        Mapping with ``halo_x``, ``halo_y``, ``halo_z``, and ``halo_m200b`` indices.
    central_id_column : int
        Column index for DescID or PID; host halos have value ``-1``.
    max_centrals : int
        Stop after this many central halos with ``M200b > 0``.

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z``, and ``M200b``.

    Raises
    ------
    ValueError
        If no central halos are found before end-of-file.
    """
    opener = bz2.open if list_path.suffix == ".bz2" else open
    mode = "rt" if list_path.suffix == ".bz2" else "r"
    x_index = column_map["halo_x"]
    y_index = column_map["halo_y"]
    z_index = column_map["halo_z"]
    mass_index = column_map["halo_m200b"]

    positions_x: list[float] = []
    positions_y: list[float] = []
    positions_z: list[float] = []
    masses: list[float] = []

    with opener(list_path, mode) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.split()
            if int(columns[central_id_column]) != -1:
                continue
            mass = float(columns[mass_index])
            if mass <= 0.0:
                continue
            positions_x.append(float(columns[x_index]))
            positions_y.append(float(columns[y_index]))
            positions_z.append(float(columns[z_index]))
            masses.append(mass)
            if len(masses) >= max_centrals:
                break

    if not masses:
        raise ValueError(f"No central halos found in catalog {list_path}")

    return pd.DataFrame(
        {
            "x": np.array(positions_x, dtype=np.float64),
            "y": np.array(positions_y, dtype=np.float64),
            "z": np.array(positions_z, dtype=np.float64),
            "M200b": np.array(masses, dtype=np.float64),
        }
    )


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
    max_centrals: int,
) -> pd.DataFrame:
    """
    Load up to ``max_centrals`` FastPM host halos (Rockstar ``DescID == -1``).

    Parameters
    ----------
    list_path : Path
        Path to ``out_*.list`` under ``rockstar_out_pm``.
    max_centrals : int
        Maximum number of central halos to collect.

    Returns
    -------
    pd.DataFrame
        Host halos with ``x``, ``y``, ``z``, and ``M200b``.
    """
    return _collect_rockstar_central_halos(
        list_path,
        ROCKSTAR_LIST_COLUMNS,
        central_id_column=ROCKSTAR_LIST_COLUMNS["desc_id"],
        max_centrals=max_centrals,
    )


def load_unit_rockstar_target_catalog(
    list_path: Path,
    max_centrals: int,
) -> pd.DataFrame:
    """
    Load up to ``max_centrals`` UNIT host halos from a Rockstar ``out_*p.list.bz2``.

    Parameters
    ----------
    list_path : Path
        Path to a compressed UNIT Rockstar catalog at scale factor ``a = 1``.
    max_centrals : int
        Maximum number of central halos (``PID == -1``) to collect.

    Returns
    -------
    pd.DataFrame
        Host halos with ``x``, ``y``, ``z``, and ``M200b``.
    """
    return _collect_rockstar_central_halos(
        list_path,
        UNIT_ROCKSTAR_LIST_COLUMNS,
        central_id_column=UNIT_ROCKSTAR_LIST_COLUMNS["pid"],
        max_centrals=max_centrals,
    )
