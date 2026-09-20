"""N-halo local environment input feature."""

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.spatial

from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.base import InputFeatureAttacher

ENV_FEATURE_NAME = "env"
DEFAULT_POSITION_COLUMNS = ("x", "y", "z")


def _local_environment(
    positions_mpc_h: np.ndarray,
    boxsize_mpc_h: float,
    radius_mpc_h: float,
    workers: int = 5,
    log10_transform: bool = True,
) -> np.ndarray:
    """
    Count halos within a sphere using a periodic KD-tree (Haloscope ``env`` proxy).

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    radius_mpc_h : float
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
    tree = scipy.spatial.cKDTree(positions_mpc_h, boxsize=boxsize_mpc_h)
    counts = tree.query_ball_point(
        positions_mpc_h, r=radius_mpc_h, workers=workers, return_length=True
    )
    counts = counts - 1
    if log10_transform:
        return np.log10(1.0 + counts)
    return counts.astype(float)


def _attach_env_column(
    catalog: pd.DataFrame,
    boxsize_mpc_h: float,
    radius_mpc_h: float,
    position_columns: Sequence[str] = DEFAULT_POSITION_COLUMNS,
    env_column: str = ENV_FEATURE_NAME,
) -> pd.DataFrame:
    """
    Add the Haloscope ``env`` column from halo positions.

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
    catalog[env_column] = _local_environment(
        catalog[list(position_columns)].to_numpy(),
        boxsize_mpc_h,
        radius_mpc_h=radius_mpc_h,
    )
    return catalog


class NHalosEnvironmentFeature(InputFeatureAttacher):
    """Attach the Haloscope ``env`` proxy from periodic halo counts."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Return the environment column name.

        Returns
        -------
        tuple[str, ...]
            Single-element tuple with ``env``.
        """
        return (ENV_FEATURE_NAME,)

    def attach(
        self,
        catalog: pd.DataFrame,
        run_context: SimulationRunContext,
    ) -> pd.DataFrame:
        """
        Compute ``env`` from halo positions in a periodic box.

        Parameters
        ----------
        catalog : pd.DataFrame
            Halo table with ``x``, ``y``, ``z`` in Mpc/h.
        run_context : SimulationRunContext
            Box size and search radius for the KD-tree count.

        Returns
        -------
        pd.DataFrame
            The same ``catalog`` instance with an ``env`` column.
        """
        return _attach_env_column(
            catalog,
            run_context.boxsize_mpc_h,
            run_context.env_radius_mpc_h,
        )
