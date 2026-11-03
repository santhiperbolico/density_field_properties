"""Load HR/LR halo catalogs and apply preprocessing filters."""

import bz2
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from density_field_properties.pipelines.catalog_columns import (
    ROCKSTAR_LIST_CATALOG_LAYOUT,
    rockstar_catalog_column_kwargs,
)
from density_field_properties.preprocessing.filters import filter_positive_mass
from density_field_properties.read_data.halos.column_maps import (
    ROCKSTAR_LIST_COLUMNS,
    UNIT_HLIST_TRAINING_EXTRA_COLUMN_POSITIONS,
)
from density_field_properties.read_data.halos.fastpm import FastPMCatalogReader
from density_field_properties.read_data.halos.rockstar import (
    RockstarCatalogReader,
    resolve_rockstar_pid_filter,
)


def _read_host_catalog_rows(
    path: Path,
    pid_index: int,
    max_hosts: Optional[int] = None,
    filter_by_pid: bool = True,
) -> list[list[str]]:
    """
    Read data rows for host halos only from a Rockstar-style catalog file.

    Parameters
    ----------
    path : Path
        Catalog file path.
    pid_index : int
        Zero-based parent-ID column index; ``-1`` marks a host halo.
    max_hosts : Optional[int], optional
        Stop after this many host rows.
    filter_by_pid : bool, optional
        When False, every data row is kept (FastPM lists without a PID column).

    Returns
    -------
    list[list[str]]
        Tokenized host rows in file order.
    """
    apply_pid_filter, pid_column = resolve_rockstar_pid_filter(
        str(path),
        pid_index,
        filter_by_pid,
    )
    opener = bz2.open if path.suffix == ".bz2" else open
    mode = "rt" if path.suffix == ".bz2" else "r"
    rows: list[list[str]] = []
    with opener(path, mode) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.split()
            if apply_pid_filter and int(cols[pid_column]) != -1:
                continue
            rows.append(cols)
            if max_hosts is not None and len(rows) >= max_hosts:
                break
    return rows


def _halo_catalog_data_to_frame(catalog) -> pd.DataFrame:
    """
    Convert ``HaloCatalogData`` positions and mass into a preprocessing table.

    Parameters
    ----------
    catalog : HaloCatalogData
        Rockstar catalog subset from ``read_catalog``.

    Returns
    -------
    pd.DataFrame
        Table with ``id``, ``x``, ``y``, ``z``, and ``M200b``.
    """
    return pd.DataFrame(
        {
            "id": catalog.halo_id.astype(np.int64),
            "x": catalog.halo_x,
            "y": catalog.halo_y,
            "z": catalog.halo_z,
            "M200b": catalog.halo_m200b,
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
    catalog = RockstarCatalogReader.read_catalog(
        str(hlist_path),
        n_lines=max_halos,
        filter_by_pid=True,
        extra_column_positions=UNIT_HLIST_TRAINING_EXTRA_COLUMN_POSITIONS,
    )
    frame = _halo_catalog_data_to_frame(catalog)
    for column_name in UNIT_HLIST_TRAINING_EXTRA_COLUMN_POSITIONS:
        frame[column_name] = catalog.extra_columns[column_name]
    frame["cv"] = frame["Rvir"] / frame["Rs_Klypin"]
    frame = filter_positive_mass(frame)
    frame = frame.loc[np.isfinite(frame["cv"])].copy()
    return frame.reset_index(drop=True)


ROCKSTAR_VRMS_COLUMN_INDEX = 4


def _attach_rockstar_sigma_v_column(
    frame: pd.DataFrame,
    path: Path,
    catalog_column_kwargs: dict[str, int],
    max_rows: Optional[int],
    filter_by_pid: bool,
) -> pd.DataFrame:
    """
    Attach ``sigma_v`` from the Rockstar ``Vrms`` column (Forero ML feature proxy).

    Parameters
    ----------
    frame : pd.DataFrame
        Halo table aligned with the Rockstar reader row order.
    path : Path
        Path to the ``.list`` catalog.
    catalog_column_kwargs : dict[str, int]
        Column layout kwargs for PID indexing.
    max_rows : Optional[int]
        Maximum rows read from the catalog file.
    filter_by_pid : bool
        Whether host-only filtering was applied when building ``frame``.

    Returns
    -------
    pd.DataFrame
        ``frame`` with a ``sigma_v`` column in km/s.

    Raises
    ------
    ValueError
        If the list row count does not match ``frame``.
    """
    pid_index = catalog_column_kwargs["halo_pid_position"]
    rows = _read_host_catalog_rows(
        path,
        pid_index=pid_index,
        max_hosts=max_rows,
        filter_by_pid=filter_by_pid,
    )
    if len(rows) != len(frame):
        raise ValueError("Row count mismatch when attaching sigma_v from Rockstar list.")
    frame["sigma_v"] = np.array(
        [float(row[ROCKSTAR_VRMS_COLUMN_INDEX]) for row in rows],
        dtype=np.float64,
    )
    return frame


def rockstar_halo_catalog_to_dataframe(
    path: Path,
    n_lines: Optional[int] = None,
    catalog_column_kwargs: Optional[dict[str, int]] = None,
    filter_by_pid: bool = True,
) -> pd.DataFrame:
    """
    Load a Rockstar ``.list`` file via ``RockstarCatalogReader``.

    Parameters
    ----------
    path : Path
        Path to ``out_*.list``.
    n_lines : Optional[int], optional
        Maximum host halos to read.
    catalog_column_kwargs : Optional[dict[str, int]], optional
        Column index kwargs for ``read_catalog``; FastPM layouts use
        ``rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)``.
    filter_by_pid : bool, optional
        Keep only host halos when the catalog exposes a ``PID`` column.

    Returns
    -------
    pd.DataFrame
        Columns ``x``, ``y``, ``z``, ``M200b`` (and ``id`` when present).
    """
    if catalog_column_kwargs is None:
        catalog_column_kwargs = rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)
    catalog = RockstarCatalogReader.read_catalog(
        str(path),
        n_lines=n_lines,
        filter_by_pid=filter_by_pid,
        **catalog_column_kwargs,
    )
    frame = _halo_catalog_data_to_frame(catalog)
    if ROCKSTAR_LIST_COLUMNS.get("t_over_u") is not None:
        t_over_u_index = ROCKSTAR_LIST_COLUMNS["t_over_u"]
        pid_index = catalog_column_kwargs["halo_pid_position"]
        rows = _read_host_catalog_rows(
            path,
            pid_index=pid_index,
            max_hosts=n_lines,
            filter_by_pid=filter_by_pid,
        )
        frame["t_over_u"] = np.array(
            [row[t_over_u_index] for row in rows],
            dtype=np.float64,
        )
    return frame


def rockstar_list_supports_host_pid_filter(
    list_path: Path,
    catalog_column_kwargs: Optional[dict[str, int]] = None,
) -> bool:
    """
    Return whether a Rockstar list can filter to host halos (``PID == -1``).

    Parameters
    ----------
    list_path : Path
        Path to ``out_*.list``.
    catalog_column_kwargs : Optional[dict[str, int]], optional
        Column layout kwargs; defaults to FastPM Rockstar layout.

    Returns
    -------
    bool
        True when ``filter_by_pid=True`` is supported for this file.
    """
    if catalog_column_kwargs is None:
        catalog_column_kwargs = rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)
    pid_index = catalog_column_kwargs["halo_pid_position"]
    try:
        resolve_rockstar_pid_filter(str(list_path), pid_index, filter_by_pid=True)
    except ValueError:
        return False
    return True


def load_fastpm_rockstar_for_mass_calibration(
    list_path: Path,
    max_halos: Optional[int] = None,
    catalog_column_kwargs: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Load FastPM Rockstar halos for mass-calibration benchmarks.

    Keeps host halos only (``PID == -1``) when the catalog exposes a PID column;
    otherwise loads every row (FoF-style lists without PID).

    Parameters
    ----------
    list_path : Path
        Path to ``out_*.list``.
    max_halos : Optional[int], optional
        Maximum host halos (or rows when PID filtering is unavailable).
    catalog_column_kwargs : Optional[dict[str, int]], optional
        Column index kwargs for ``read_catalog``.

    Returns
    -------
    pd.DataFrame
        Table with ``x``, ``y``, ``z``, ``M200b``, and ``sigma_v``.
    """
    if catalog_column_kwargs is None:
        catalog_column_kwargs = rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)
    filter_by_pid = rockstar_list_supports_host_pid_filter(list_path, catalog_column_kwargs)
    frame = rockstar_halo_catalog_to_dataframe(
        list_path,
        n_lines=max_halos,
        catalog_column_kwargs=catalog_column_kwargs,
        filter_by_pid=filter_by_pid,
    )
    frame = _attach_rockstar_sigma_v_column(
        frame,
        list_path,
        catalog_column_kwargs,
        max_halos,
        filter_by_pid=filter_by_pid,
    )
    return filter_positive_mass(frame)


def load_fastpm_target_catalog(
    list_path: Path,
    max_halos: Optional[int] = None,
    catalog_column_kwargs: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Load FastPM Rockstar halos to enrich (positions and mass only).

    Parameters
    ----------
    list_path : Path
        Path to ``out_*.list`` under ``rockstar_out_pm``.
    max_halos : Optional[int], optional
        Subset size for quick runs.
    catalog_column_kwargs : Optional[dict[str, int]], optional
        Column index kwargs for ``read_catalog``; FastPM layouts use
        ``rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)``.
    Returns
    -------
    pd.DataFrame
        Positive-mass host halos with ``x``, ``y``, ``z``, ``M200b``.
    """
    if catalog_column_kwargs is None:
        catalog_column_kwargs = rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)
    frame = rockstar_halo_catalog_to_dataframe(
        list_path,
        n_lines=max_halos,
        catalog_column_kwargs=catalog_column_kwargs,
        filter_by_pid=False,
    )
    frame = _attach_rockstar_sigma_v_column(
        frame,
        list_path,
        catalog_column_kwargs,
        max_halos,
        filter_by_pid=False,
    )
    return filter_positive_mass(frame)


def load_fastpm_fof_target_catalog(
    fof_group_path: Path,
    max_halos: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load FastPM FoF groups for LR mass calibration and benchmarking.

    Parameters
    ----------
    fof_group_path : Path
        Path to the FoF BigFile group directory (for example ``.../LL-0.200``).
    max_halos : Optional[int], optional
        Maximum number of groups to read.

    Returns
    -------
    pd.DataFrame
        Positive-mass groups with ``id``, ``x``, ``y``, ``z``, and ``M200b``.
    """
    catalog = FastPMCatalogReader.read_catalog(str(fof_group_path), n_lines=max_halos)
    frame = _halo_catalog_data_to_frame(catalog)
    return filter_positive_mass(frame)
