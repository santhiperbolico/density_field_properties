"""Haloscope model fitting per log-mass bin."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from density_field_properties.haloscope.bins import mask_mass_bin
from density_field_properties.haloscope.model import ConditionalMultiVariateGaussian


def fit_models(
    training_table: pd.DataFrame,
    bin_edges: np.ndarray,
    input_features: Sequence[str],
    output_features: Sequence[str],
    min_bin_size: int,
    mass_column: str = "M200b",
    conditional_model_class=None,
) -> Dict[int, object]:
    """
    Fit one Haloscope model per log-mass bin on the training table.

    Parameters
    ----------
    training_table : pd.DataFrame
        High-resolution catalog with input and output feature columns.
    bin_edges : np.ndarray
        log10 mass bin edges.
    input_features : Sequence[str]
        Haloscope conditioning columns.
    output_features : Sequence[str]
        Target secondary-property columns.
    min_bin_size : int
        Skip bins with fewer training halos than this threshold.
    mass_column : str, optional
        Mass column used for bin assignment.
    conditional_model_class : type, optional
        Class with ``fit`` and ``predict``; defaults to Haloscope CMVG.

    Returns
    -------
    dict[int, object]
        Fitted models keyed by bin index.
    """
    if conditional_model_class is None:
        conditional_model_class = ConditionalMultiVariateGaussian

    feature_columns = list(input_features)
    target_columns = list(output_features)
    log_mass = np.log10(training_table[mass_column].to_numpy())
    models: Dict[int, object] = {}
    n_bins = len(bin_edges) - 1

    for bin_index in range(n_bins):
        low, high = bin_edges[bin_index], bin_edges[bin_index + 1]
        train_mask = mask_mass_bin(log_mass, low, high)
        train_bin = training_table.loc[train_mask]
        if len(train_bin) < min_bin_size:
            continue
        model = conditional_model_class()
        model.fit(
            train_bin[feature_columns].to_numpy(),
            train_bin[target_columns].to_numpy(),
        )
        models[bin_index] = model
    return models


def holdout_validate_sim_bins(
    halos_sim: pd.DataFrame,
    bin_edges: np.ndarray,
    input_features: Sequence[str],
    output_features: Sequence[str],
    mass_column: str = "M200b",
    test_fraction: float = 0.2,
    random_seed: int = 0,
    min_bin_size: int = 10,
    conditional_model_class=None,
) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Train Haloscope models per mass bin on a SIM train split and predict on held-out SIM.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        Training catalog with input and output feature columns.
    bin_edges : np.ndarray
        log10 mass bin edges.
    input_features : Sequence[str]
        Haloscope conditioning columns.
    output_features : Sequence[str]
        Target secondary-property columns.
    mass_column : str, optional
        Mass column used for binning.
    test_fraction : float, optional
        Fraction of SIM halos assigned to the test split.
    random_seed : int, optional
        RNG seed for the train/test split.
    min_bin_size : int, optional
        Skip bins with fewer than this many train or test halos.
    conditional_model_class : type, optional
        Class with ``fit`` and ``predict``; defaults to Haloscope CMVG.

    Returns
    -------
    list[tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]
        Per-bin tuples ``(bin_index, y_true_test, y_pred_test)``.
    """
    if conditional_model_class is None:
        conditional_model_class = ConditionalMultiVariateGaussian

    feature_columns = list(input_features)
    target_columns = list(output_features)

    rng = np.random.default_rng(random_seed)
    mask_test = rng.random(len(halos_sim)) < test_fraction
    sim_train = halos_sim.loc[~mask_test]
    sim_test = halos_sim.loc[mask_test]
    log_mass_train = np.log10(sim_train[mass_column].to_numpy())
    log_mass_test = np.log10(sim_test[mass_column].to_numpy())

    results: List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]] = []
    n_bins = len(bin_edges) - 1
    for bin_index in range(n_bins):
        low, high = bin_edges[bin_index], bin_edges[bin_index + 1]
        train_mask = mask_mass_bin(log_mass_train, low, high)
        test_mask = mask_mass_bin(log_mass_test, low, high)
        train_bin = sim_train.loc[train_mask]
        test_bin = sim_test.loc[test_mask]
        if len(train_bin) < min_bin_size or len(test_bin) < min_bin_size:
            results.append((bin_index, None, None))
            continue
        model = conditional_model_class()
        model.fit(
            train_bin[feature_columns].to_numpy(),
            train_bin[target_columns].to_numpy(),
        )
        y_pred = model.predict(test_bin[feature_columns].to_numpy())
        y_true = test_bin[target_columns].to_numpy()
        results.append((bin_index, y_true, y_pred))
    return results
