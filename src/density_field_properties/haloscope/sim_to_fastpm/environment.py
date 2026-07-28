"""Local halo environment proxy for Haloscope conditioning."""

import numpy as np
import scipy.spatial


def local_environment(
    positions_mpc_h: np.ndarray,
    boxsize_mpc_h: float,
    radius_mpc_h: float = 5.0,
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
    tree = scipy.spatial.cKDTree(positions_mpc_h, boxsize=boxsize_mpc_h)
    counts = tree.query_ball_point(
        positions_mpc_h, r=radius_mpc_h, workers=workers, return_length=True
    )
    counts = counts - 1
    if log10_transform:
        return np.log10(1.0 + counts)
    return counts.astype(float)
