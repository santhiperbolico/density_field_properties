"""Tidal anisotropy input feature from precomputed descriptor batches."""

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from density_field_properties.environment_properties.cic.utils import get_grid_cell
from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.base import InputFeatureAttacher

TIDAL_ANISOTROPY_FEATURE_NAME = "tidal_anisotropy"
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
DEFAULT_POSITION_COLUMNS = ("x", "y", "z")


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


def _load_halo_environment_descriptor_table(
    descriptor_dir: Path,
    max_batch_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load tidal anisotropy and overdensity batches into one table.

    Parameters
    ----------
    descriptor_dir : Path
        Directory with ``*_halo_environment_descriptors.txt`` outputs.
    max_batch_files : Optional[int], optional
        Cap on batch files for smoke runs; ``None`` loads all batches.

    Returns
    -------
    pd.DataFrame
        Descriptor rows with grid-cell coordinates and ``tidal_anisotropy``.
    """
    batch_paths = _descriptor_batch_paths(descriptor_dir, max_batch_files=max_batch_files)
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


def _attach_tidal_anisotropy(
    catalog: pd.DataFrame,
    descriptor_dir: Path,
    boxsize_mpc_h: float,
    n_grid: int,
    max_batch_files: Optional[int] = None,
    position_columns: Sequence[str] = DEFAULT_POSITION_COLUMNS,
) -> pd.DataFrame:
    """
    Merge precomputed tidal anisotropy onto a halo catalog by grid-cell key.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table with position columns in Mpc/h.
    descriptor_dir : Path
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
        The same ``catalog`` instance with a ``tidal_anisotropy`` column.
    """
    descriptors = _load_halo_environment_descriptor_table(
        descriptor_dir,
        max_batch_files=max_batch_files,
    )
    descriptors = descriptors.drop_duplicates(
        subset=["cell_x", "cell_y", "cell_z"],
        keep="first",
    )
    grid_keys = _grid_merge_keys(
        catalog[list(position_columns)].to_numpy(dtype=np.float64),
        boxsize_mpc_h,
        n_grid,
    )
    merged = grid_keys.merge(
        descriptors[["cell_x", "cell_y", "cell_z", "tidal_anisotropy"]],
        on=["cell_x", "cell_y", "cell_z"],
        how="left",
    )
    catalog[TIDAL_ANISOTROPY_FEATURE_NAME] = merged["tidal_anisotropy"].to_numpy()
    return catalog


class TidalAnisotropyFeature(InputFeatureAttacher):
    """Attach ``tidal_anisotropy`` by merging grid-cell descriptor batches."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Return the tidal anisotropy column name.

        Returns
        -------
        tuple[str, ...]
            Single-element tuple with ``tidal_anisotropy``.
        """
        return (TIDAL_ANISOTROPY_FEATURE_NAME,)

    def attach(
        self,
        catalog: pd.DataFrame,
        run_context: SimulationRunContext,
    ) -> pd.DataFrame:
        """
        Merge tidal anisotropy descriptors onto a halo catalog.

        Parameters
        ----------
        catalog : pd.DataFrame
            Halo table with position columns in Mpc/h.
        run_context : SimulationRunContext
            Descriptor directory and grid resolution.

        Returns
        -------
        pd.DataFrame
            The same ``catalog`` instance with ``tidal_anisotropy``.

        Raises
        ------
        ValueError
            If ``tidal_descriptors_dir`` is not configured.
        """
        if run_context.tidal_descriptors_dir is None:
            raise ValueError("tidal_anisotropy requires tidal_descriptors_dir in the run context")
        return _attach_tidal_anisotropy(
            catalog,
            run_context.tidal_descriptors_dir,
            run_context.boxsize_mpc_h,
            run_context.tidal_n_grid,
            max_batch_files=run_context.max_descriptor_batch_files,
        )
