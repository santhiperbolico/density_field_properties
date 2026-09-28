"""Periodic HR-LR halo matching kernel (Forero et al. style, section 3.1)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.spatial

from density_field_properties.preprocessing.mass_calibration_config import (
    MatchingHyperparameters,
)

DEFAULT_POSITION_COLUMNS = ("x", "y", "z")
KD_TREE_WORKERS = 5
UNMATCHED_INDEX = -1


@dataclass(frozen=True)
class HaloMatchingResult:
    """
    One-to-one HR-LR matching outcome indexed by catalog row order.

    Parameters
    ----------
    lr_to_hr_index : np.ndarray
        HR row index paired with each LR halo, or ``UNMATCHED_INDEX``.
    hr_to_lr_index : np.ndarray
        LR row index paired with each HR halo, or ``UNMATCHED_INDEX``.
    lr_spatial_distance_mpc_h : np.ndarray
        Periodic spatial separation at match time; NaN when unmatched.
    """

    lr_to_hr_index: np.ndarray
    hr_to_lr_index: np.ndarray
    lr_spatial_distance_mpc_h: np.ndarray


def _assign_lr_to_hr(
    hr_index: int,
    lr_index: int,
    spatial_distance: float,
    hr_to_lr: np.ndarray,
    lr_to_hr: np.ndarray,
    lr_distance: np.ndarray,
) -> None:
    """
    Record a mutual HR–LR pair in the assignment arrays.

    Parameters
    ----------
    hr_index : int
        HR row index.
    lr_index : int
        LR row index.
    spatial_distance : float
        Periodic separation in Mpc/h.
    hr_to_lr : np.ndarray
        HR → LR assignment buffer updated in place.
    lr_to_hr : np.ndarray
        LR → HR assignment buffer updated in place.
    lr_distance : np.ndarray
        LR spatial distance buffer updated in place.
    """
    hr_to_lr[hr_index] = lr_index
    lr_to_hr[lr_index] = hr_index
    lr_distance[lr_index] = spatial_distance


def _clear_lr_assignment(
    lr_index: int,
    hr_to_lr: np.ndarray,
    lr_to_hr: np.ndarray,
    lr_distance: np.ndarray,
) -> int:
    """
    Remove the HR partner currently assigned to one LR halo.

    Parameters
    ----------
    lr_index : int
        LR row index to unassign.
    hr_to_lr : np.ndarray
        HR → LR assignment buffer updated in place.
    lr_to_hr : np.ndarray
        LR → HR assignment buffer updated in place.
    lr_distance : np.ndarray
        LR spatial distance buffer updated in place.

    Returns
    -------
    int
        Former HR index, or ``UNMATCHED_INDEX`` when the LR was free.
    """
    former_hr = int(lr_to_hr[lr_index])
    if former_hr == UNMATCHED_INDEX:
        return UNMATCHED_INDEX
    hr_to_lr[former_hr] = UNMATCHED_INDEX
    lr_to_hr[lr_index] = UNMATCHED_INDEX
    lr_distance[lr_index] = np.nan
    return former_hr


def _try_match_hr_to_lr_candidates(
    hr_index: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    mass_hr: np.ndarray,
    mass_lr: np.ndarray,
    max_mass_distance: float,
    hr_to_lr: np.ndarray,
    lr_to_hr: np.ndarray,
    lr_distance: np.ndarray,
) -> bool:
    """
    Match one HR halo to the best available LR among k periodic neighbors.

    When an LR partner is reassigned to a closer HR, the displaced HR is left
    unmatched until a later pass of the outer iteration loop (Forero-style
    conflict resolution, no in-pass recursion).

    Parameters
    ----------
    hr_index : int
        HR row index to match.
    neighbor_indices : np.ndarray
        LR indices returned by ``cKDTree.query`` for this HR halo.
    neighbor_distances : np.ndarray
        Periodic distances aligned with ``neighbor_indices``.
    mass_hr : np.ndarray
        HR masses aligned with the HR catalog row order.
    mass_lr : np.ndarray
        LR masses aligned with the LR catalog row order.
    max_mass_distance : float
        Maximum allowed |M_LR - M_HR| / M_HR.
    hr_to_lr : np.ndarray
        HR → LR assignment buffer updated in place.
    lr_to_hr : np.ndarray
        LR → HR assignment buffer updated in place.
    lr_distance : np.ndarray
        LR spatial distance buffer updated in place.

    Returns
    -------
    bool
        True when a new LR assignment was created or updated for this HR halo.
    """
    for lr_index, spatial_distance in zip(neighbor_indices, neighbor_distances):
        if not np.isfinite(spatial_distance):
            break
        lr_index = int(lr_index)
        if lr_index < 0 or lr_index >= len(mass_lr):
            break

        relative_mass_distance = float(
            abs(mass_lr[lr_index] - mass_hr[hr_index]) / mass_hr[hr_index]
        )
        if relative_mass_distance >= max_mass_distance:
            continue

        current_hr = int(lr_to_hr[lr_index])
        if current_hr == UNMATCHED_INDEX:
            _assign_lr_to_hr(
                hr_index,
                lr_index,
                float(spatial_distance),
                hr_to_lr,
                lr_to_hr,
                lr_distance,
            )
            return True

        if current_hr == hr_index:
            return False

        current_distance = float(lr_distance[lr_index])
        if spatial_distance >= current_distance:
            continue

        _clear_lr_assignment(lr_index, hr_to_lr, lr_to_hr, lr_distance)
        _assign_lr_to_hr(
            hr_index,
            lr_index,
            float(spatial_distance),
            hr_to_lr,
            lr_to_hr,
            lr_distance,
        )
        return True
    return False


def match_hr_to_lr_one_to_one(
    hr_catalog: pd.DataFrame,
    lr_catalog: pd.DataFrame,
    box_size_mpc_h: float,
    params: MatchingHyperparameters,
    position_columns: tuple[str, str, str] = DEFAULT_POSITION_COLUMNS,
) -> HaloMatchingResult:
    """
    Match HR halos to LR halos with periodic k-neighbour search and conflict resolution.

    HR halos are scanned in index order each iteration. If a closer HR takes an
    LR partner, the previous HR is unmatched and may pair with another LR on a
    later iteration, up to ``max_matching_iterations``.

    Complexity is O(N_HR * k * max_matching_iterations) after one kd-tree build;
    suitable for smoke catalogs up to ~1e6 halos without extra tiling.

    Parameters
    ----------
    hr_catalog : pd.DataFrame
        High-resolution halo table with positions and masses.
    lr_catalog : pd.DataFrame
        Low-resolution halo table with positions and masses.
    box_size_mpc_h : float
        Periodic box side length in Mpc/h.
    params : MatchingHyperparameters
        Matching hyperparameters from run configuration.
    position_columns : tuple[str, str, str], optional
        Comoving position columns shared by both catalogs.

    Returns
    -------
    HaloMatchingResult
        Mutual assignment arrays indexed by input row order.

    Raises
    ------
    ValueError
        If catalogs are empty or required columns are missing.
    """
    lr_positions = lr_catalog[list(position_columns)].to_numpy(dtype=float)
    hr_positions = hr_catalog[list(position_columns)].to_numpy(dtype=float)
    mass_lr = lr_catalog[params.lr_mass_column].to_numpy(dtype=float)
    mass_hr = hr_catalog[params.hr_mass_column].to_numpy(dtype=float)

    n_hr = len(hr_catalog)
    n_lr = len(lr_catalog)
    k_query = min(params.k_neighbours, n_lr)

    tree = scipy.spatial.cKDTree(lr_positions, boxsize=box_size_mpc_h)
    distances, neighbor_indices = tree.query(
        hr_positions,
        k=k_query,
        distance_upper_bound=params.r_match_mpc_h,
        workers=KD_TREE_WORKERS,
    )
    if k_query == 1:
        distances = distances.reshape(n_hr, 1)
        neighbor_indices = neighbor_indices.reshape(n_hr, 1)

    hr_to_lr = np.full(n_hr, UNMATCHED_INDEX, dtype=np.int64)
    lr_to_hr = np.full(n_lr, UNMATCHED_INDEX, dtype=np.int64)
    lr_distance = np.full(n_lr, np.nan, dtype=float)

    for _ in range(params.max_matching_iterations):
        changed = False
        for hr_index in range(n_hr):
            if _try_match_hr_to_lr_candidates(
                hr_index,
                neighbor_indices[hr_index],
                distances[hr_index],
                mass_hr,
                mass_lr,
                params.max_mass_distance,
                hr_to_lr,
                lr_to_hr,
                lr_distance,
            ):
                changed = True
        if not changed:
            break

    return HaloMatchingResult(
        lr_to_hr_index=lr_to_hr,
        hr_to_lr_index=hr_to_lr,
        lr_spatial_distance_mpc_h=lr_distance,
    )


def hr_masses_for_matched_lr(
    matching: HaloMatchingResult,
    hr_masses: np.ndarray,
) -> np.ndarray:
    """
    Map matched LR rows to HR partner masses.

    Parameters
    ----------
    matching : HaloMatchingResult
        Output of ``match_hr_to_lr_one_to_one``.
    hr_masses : np.ndarray
        HR mass array aligned with the HR catalog used for matching.

    Returns
    -------
    np.ndarray
        HR mass on matched LR rows; NaN where ``lr_to_hr_index`` is unmatched.
    """
    calibrated = np.full(len(matching.lr_to_hr_index), np.nan, dtype=float)
    matched_mask = matching.lr_to_hr_index != UNMATCHED_INDEX
    calibrated[matched_mask] = hr_masses[matching.lr_to_hr_index[matched_mask]]
    return calibrated
