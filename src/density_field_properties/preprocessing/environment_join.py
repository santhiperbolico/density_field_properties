"""Deprecated shim — use preprocessing.input_features.n_halos_env."""

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from density_field_properties.preprocessing.input_features.n_halos_env import (
    _attach_env_column,
    _local_environment,
)

warnings.warn(
    "preprocessing.environment_join is deprecated; "
    "use density_field_properties.preprocessing.input_features.n_halos_env",
    DeprecationWarning,
    stacklevel=2,
)


def local_environment(
    positions_mpc_h: np.ndarray,
    boxsize_mpc_h: float,
    radius_mpc_h: float = 5.0,
    workers: int = 5,
    log10_transform: bool = True,
) -> np.ndarray:
    """
    Deprecated wrapper around ``NHalosEnvironmentFeature`` kernel logic.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    radius_mpc_h : float, optional
        Search radius in Mpc/h.
    workers : int, optional
        Number of workers passed to ``cKDTree.query_ball_point``.
    log10_transform : bool, optional
        If True, return ``log10(1 + neighbor_count)`` excluding self.

    Returns
    -------
    np.ndarray
        Environment feature per halo, shape ``(N,)``.
    """
    return _local_environment(
        positions_mpc_h,
        boxsize_mpc_h,
        radius_mpc_h,
        workers=workers,
        log10_transform=log10_transform,
    )


def attach_env_column(
    catalog: pd.DataFrame,
    boxsize_mpc_h: float,
    radius_mpc_h: float,
    position_columns: Sequence[str] = ("x", "y", "z"),
    env_column: str = "env",
) -> pd.DataFrame:
    """
    Deprecated wrapper around ``NHalosEnvironmentFeature.attach``.

    Parameters
    ----------
    catalog : pd.DataFrame
        Halo table with position columns in Mpc/h.
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    radius_mpc_h : float
        Neighbor search radius in Mpc/h.
    position_columns : Sequence[str], optional
        Position column names.
    env_column : str, optional
        Output environment column name.

    Returns
    -------
    pd.DataFrame
        The same ``catalog`` instance with ``env_column`` attached.
    """
    return _attach_env_column(
        catalog,
        boxsize_mpc_h,
        radius_mpc_h,
        position_columns=position_columns,
        env_column=env_column,
    )
