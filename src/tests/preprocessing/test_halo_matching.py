"""Tests for periodic HR-LR halo matching."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.halo_matching import (
    UNMATCHED_INDEX,
    hr_masses_for_matched_lr,
    match_hr_to_lr_one_to_one,
)
from density_field_properties.preprocessing.mass_calibration_config import (
    MatchingHyperparameters,
)


def _matching_params(**overrides) -> MatchingHyperparameters:
    """
    Build matching hyperparameters with optional overrides.

    Parameters
    ----------
    **overrides
        Field overrides passed to ``MatchingHyperparameters``.

    Returns
    -------
    MatchingHyperparameters
        Parameter bundle for tests.
    """
    base = MatchingHyperparameters(
        k_neighbours=5,
        r_match_mpc_h=5.0,
        max_mass_distance=0.95,
        max_matching_iterations=5,
    )
    return base if not overrides else MatchingHyperparameters(**{**base.__dict__, **overrides})


def test_obvious_pair_matches_with_hr_mass_on_lr():
    """
    A single HR/LR pair within radius and mass gate should match mutually.
    """
    hr = pd.DataFrame(
        {
            "x": [10.0],
            "y": [10.0],
            "z": [10.0],
            "M200b": [1.0e12],
        }
    )
    lr = pd.DataFrame(
        {
            "x": [10.5],
            "y": [10.0],
            "z": [10.0],
            "M200b": [9.5e11],
        }
    )
    result = match_hr_to_lr_one_to_one(hr, lr, box_size_mpc_h=100.0, params=_matching_params())

    assert result.hr_to_lr_index[0] == 0
    assert result.lr_to_hr_index[0] == 0
    calibrated = hr_masses_for_matched_lr(result, hr["M200b"].to_numpy())
    assert calibrated[0] == pytest.approx(1.0e12)


def test_periodic_wrap_pairs_across_box_boundary():
    """
    Periodic kd-tree search should match halos separated by a box wrap.
    """
    hr = pd.DataFrame({"x": [0.5], "y": [0.0], "z": [0.0], "M200b": [2.0e12]})
    lr = pd.DataFrame({"x": [9.5], "y": [0.0], "z": [0.0], "M200b": [1.9e12]})
    result = match_hr_to_lr_one_to_one(
        hr,
        lr,
        box_size_mpc_h=10.0,
        params=_matching_params(r_match_mpc_h=2.0),
    )

    assert result.lr_to_hr_index[0] == 0
    assert result.lr_spatial_distance_mpc_h[0] == pytest.approx(1.0)


def test_mass_gate_blocks_match_when_offset_too_large():
    """
    Relative mass distance above ``max_mass_distance`` should leave LR unmatched.
    """
    hr = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0], "M200b": [1.0e12]})
    lr = pd.DataFrame({"x": [0.1], "y": [0.0], "z": [0.0], "M200b": [1.0e10]})
    result = match_hr_to_lr_one_to_one(hr, lr, box_size_mpc_h=10.0, params=_matching_params())

    assert result.lr_to_hr_index[0] == UNMATCHED_INDEX
    assert np.isnan(hr_masses_for_matched_lr(result, hr["M200b"].to_numpy())[0])


def test_displaced_hr_rematches_on_later_iteration():
    """
    A displaced HR should find another LR on a later iteration, not via recursion.
    """
    lr = pd.DataFrame(
        {
            "x": [0.0, 3.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "M200b": [1.0e12, 1.0e12],
        }
    )
    hr = pd.DataFrame(
        {
            "x": [0.2, 0.05],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "M200b": [1.0e12, 1.0e12],
        }
    )
    single_pass = match_hr_to_lr_one_to_one(
        hr,
        lr,
        box_size_mpc_h=20.0,
        params=_matching_params(max_matching_iterations=1, k_neighbours=2),
    )
    assert single_pass.hr_to_lr_index[0] == UNMATCHED_INDEX
    assert single_pass.hr_to_lr_index[1] == 0

    multi_pass = match_hr_to_lr_one_to_one(
        hr,
        lr,
        box_size_mpc_h=20.0,
        params=_matching_params(max_matching_iterations=5, k_neighbours=2),
    )
    assert multi_pass.hr_to_lr_index[1] == 0
    assert multi_pass.hr_to_lr_index[0] == 1
    assert multi_pass.lr_to_hr_index[0] == 1
    assert multi_pass.lr_to_hr_index[1] == 0


def test_closer_hr_wins_when_two_hr_share_one_lr_candidate():
    """
    Two HR halos near one LR should assign the LR to the closer HR halo.
    """
    lr = pd.DataFrame({"x": [5.0], "y": [0.0], "z": [0.0], "M200b": [1.0e12]})
    hr = pd.DataFrame(
        {
            "x": [4.0, 6.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "M200b": [1.0e12, 1.0e12],
        }
    )
    result = match_hr_to_lr_one_to_one(
        hr,
        lr,
        box_size_mpc_h=20.0,
        params=_matching_params(k_neighbours=1),
    )

    assert result.hr_to_lr_index[0] == 0
    assert result.hr_to_lr_index[1] == UNMATCHED_INDEX
    assert result.lr_to_hr_index[0] == 0
    assert result.lr_spatial_distance_mpc_h[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "hr_rows,lr_rows",
    [
        (3, 4),
        (5, 2),
    ],
)
def test_at_most_one_hr_per_lr_row(hr_rows, lr_rows):
    """
    Each LR row should appear at most once across all HR assignments.
    """
    rng = np.random.default_rng(42)
    hr = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 50.0, hr_rows),
            "y": rng.uniform(0.0, 50.0, hr_rows),
            "z": rng.uniform(0.0, 50.0, hr_rows),
            "M200b": rng.uniform(1.0e11, 1.0e12, hr_rows),
        }
    )
    lr = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 50.0, lr_rows),
            "y": rng.uniform(0.0, 50.0, lr_rows),
            "z": rng.uniform(0.0, 50.0, lr_rows),
            "M200b": rng.uniform(1.0e11, 1.0e12, lr_rows),
        }
    )
    result = match_hr_to_lr_one_to_one(hr, lr, box_size_mpc_h=100.0, params=_matching_params())

    assigned_lr = result.hr_to_lr_index[result.hr_to_lr_index != UNMATCHED_INDEX]
    assert assigned_lr.size == np.unique(assigned_lr).size
