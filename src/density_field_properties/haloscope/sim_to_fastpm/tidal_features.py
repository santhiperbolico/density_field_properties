"""Tidal anisotropy descriptors and Rockstar T/|U| for Haloscope conditioning."""

import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from density_field_properties.density_field.utils import get_grid_cell

DESCRIPTOR_FILE_PATTERN = re.compile(r"^(\d+)_halo_environment_descriptors\.txt$")
DESCRIPTOR_COLUMNS = [
    "descriptor_halo_id",
    "cell_x",
    "cell_y",
    "cell_z",
    "descriptor_m200b",
    "halo_rg",
    "tidal_anisotropy",
    "overdensity",
]


def _descriptor_batch_paths(
    descriptor_dir: Path,
    max_batch_files: Optional[int] = None,
) -> list[Path]:
    """
    List tidal descriptor batch files sorted by batch index.

    Parameters
    ----------
    descriptor_dir : Path
        Directory containing ``*_halo_environment_descriptors.txt`` files.
    max_batch_files : Optional[int], optional
        If set, keep only the first N batch files after sorting.

    Returns
    -------
    list[Path]
        Sorted batch file paths.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or contains no batch files.
    """
    if not descriptor_dir.is_dir():
        raise FileNotFoundError(f"Tidal descriptor directory not found: {descriptor_dir}")

    batch_paths: list[tuple[int, Path]] = []
    for path in descriptor_dir.iterdir():
        match = DESCRIPTOR_FILE_PATTERN.match(path.name)
        if match is not None:
            batch_paths.append((int(match.group(1)), path))
    if not batch_paths:
        raise FileNotFoundError(f"No tidal descriptor batch files in {descriptor_dir}")
    batch_paths.sort(key=lambda item: item[0])
    paths = [path for _, path in batch_paths]
    if max_batch_files is not None:
        paths = paths[:max_batch_files]
    return paths


def load_halo_environment_descriptor_table(
    descriptor_dir: Union[str, Path],
    max_batch_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load tidal anisotropy and overdensity batches into one table.

    Parameters
    ----------
    descriptor_dir : str or Path
        Directory with ``*_halo_environment_descriptors.txt`` outputs.
    max_batch_files : Optional[int], optional
        Cap on batch files for smoke runs; ``None`` loads all batches.

    Returns
    -------
    pd.DataFrame
        Descriptor rows with grid-cell coordinates and ``tidal_anisotropy``.
    """
    directory = Path(descriptor_dir)
    batch_paths = _descriptor_batch_paths(directory, max_batch_files=max_batch_files)
    tables = [pd.read_csv(path, comment="#", sep=r"\s+", header=None) for path in batch_paths]
    frame = pd.concat(tables, ignore_index=True)
    frame.columns = DESCRIPTOR_COLUMNS
    return frame


def _grid_merge_keys(
    positions_mpc_h: np.ndarray,
    boxsize_mpc_h: float,
    n_grid: int,
) -> pd.DataFrame:
    """
    Build integer grid-cell keys for halo positions.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    n_grid : int
        CIC grid resolution used for tidal descriptors.

    Returns
    -------
    pd.DataFrame
        Columns ``cell_x``, ``cell_y``, ``cell_z``.
    """
    cells = get_grid_cell(positions_mpc_h, boxsize_mpc_h, n_grid).astype(np.int64)
    return pd.DataFrame(
        {
            "cell_x": cells[:, 0],
            "cell_y": cells[:, 1],
            "cell_z": cells[:, 2],
        }
    )


def attach_tidal_anisotropy(
    catalog: pd.DataFrame,
    descriptor_dir: Union[str, Path],
    boxsize_mpc_h: float,
    n_grid: int,
    max_batch_files: Optional[int] = None,
    position_columns: Sequence[str] = ("x", "y", "z"),
) -> pd.DataFrame:
    """
    Merge precomputed tidal anisotropy onto a halo catalog by grid-cell key.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table with position columns in Mpc/h.
    descriptor_dir : str or Path
        Directory with tidal descriptor batch files.
    boxsize_mpc_h : float
        Box side length in Mpc/h.
    n_grid : int
        Grid resolution used when the descriptors were computed.
    max_batch_files : Optional[int], optional
        Smoke cap on descriptor batch files.
    position_columns : Sequence[str], optional
        Position column names in ``catalog``.

    Returns
    -------
    pd.DataFrame
        Copy of ``catalog`` with a ``tidal_anisotropy`` column.
    """
    descriptors = load_halo_environment_descriptor_table(
        descriptor_dir,
        max_batch_files=max_batch_files,
    )
    descriptors = descriptors.drop_duplicates(
        subset=["cell_x", "cell_y", "cell_z"],
        keep="first",
    )
    enriched = catalog.copy()
    grid_keys = _grid_merge_keys(
        enriched[list(position_columns)].to_numpy(dtype=np.float64),
        boxsize_mpc_h,
        n_grid,
    )
    enriched = pd.concat([enriched.reset_index(drop=True), grid_keys], axis=1)
    merged = enriched.merge(
        descriptors[["cell_x", "cell_y", "cell_z", "tidal_anisotropy"]],
        on=["cell_x", "cell_y", "cell_z"],
        how="left",
    )
    return merged.drop(columns=["cell_x", "cell_y", "cell_z"])


def filter_finite_input_features(
    catalog: pd.DataFrame,
    input_features: Sequence[str],
) -> pd.DataFrame:
    """
    Drop rows with non-finite values in the Haloscope input feature columns.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table.
    input_features : Sequence[str]
        Column names required for Haloscope ``fit`` / ``predict``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with a reset index.
    """
    mask = np.ones(len(catalog), dtype=bool)
    for column in input_features:
        mask &= np.isfinite(catalog[column].to_numpy(dtype=np.float64))
    return catalog.loc[mask].reset_index(drop=True)
