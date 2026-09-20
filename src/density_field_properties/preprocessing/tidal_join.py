"""Deprecated shim — use preprocessing.input_features.tidal_anisotropy."""

import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

warnings.warn(
    "preprocessing.tidal_join is deprecated; "
    "use density_field_properties.preprocessing.input_features.tidal_anisotropy",
    DeprecationWarning,
    stacklevel=2,
)


def load_halo_environment_descriptor_table(
    descriptor_dir: Union[str, Path],
    max_batch_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Deprecated wrapper around tidal descriptor table loading.

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
    from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
        _load_halo_environment_descriptor_table,
    )

    return _load_halo_environment_descriptor_table(
        Path(descriptor_dir),
        max_batch_files=max_batch_files,
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
    Deprecated wrapper around tidal anisotropy attachment.

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
        The same ``catalog`` instance with a ``tidal_anisotropy`` column.
    """
    from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
        _attach_tidal_anisotropy,
    )

    return _attach_tidal_anisotropy(
        catalog,
        Path(descriptor_dir),
        boxsize_mpc_h,
        n_grid,
        max_batch_files=max_batch_files,
        position_columns=position_columns,
    )
