"""Mass-bin Haloscope training and FastPM enrichment (schema helpers)."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from density_field_properties.haloscope import ConditionalMultiVariateGaussian
from density_field_properties.haloscope.sim_to_fastpm.config import INPUT_FEATURES, OUTPUT_FEATURES


def default_mass_bin_edges(log_m200b_max: float) -> np.ndarray:
    """
    Default log10(M200b) bin edges for Haloscope fits.

    Parameters
    ----------
    log_m200b_max : float
        Upper edge taken from the training sample (log10 Msun/h).

    Returns
    -------
    np.ndarray
        Bin edges with length ``n_bins + 1``.
    """
    return np.array([10.0, 10.8, 11.6, 12.6, log_m200b_max])


def mask_mass_bin(log_mass: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Boolean mask for halos in one log10 mass bin ``[low, high)``.

    Parameters
    ----------
    log_mass : np.ndarray
        log10(M200b) per halo.
    low : float
        Lower bin edge in log10.
    high : float
        Upper bin edge in log10.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    return (log_mass >= low) & (log_mass < high)


def holdout_validate_sim_bins(
    halos_sim: pd.DataFrame,
    bin_edges: np.ndarray,
    mass_column: str = "M200b",
    test_fraction: float = 0.2,
    random_seed: int = 0,
    min_bin_size: int = 10,
    conditional_model_class=None,
    input_features: Optional[Sequence[str]] = None,
) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Train Haloscope models per mass bin on a SIM train split and predict on held-out SIM.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        Training catalog with ``INPUT_FEATURES`` and ``OUTPUT_FEATURES`` columns.
    bin_edges : np.ndarray
        log10 mass bin edges.
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
    input_features : Sequence[str], optional
        Haloscope conditioning columns; defaults to ``INPUT_FEATURES``.

    Returns
    -------
    list[tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]
        Per-bin tuples ``(bin_index, y_true_test, y_pred_test)``.
    """
    if conditional_model_class is None:
        conditional_model_class = ConditionalMultiVariateGaussian
    if input_features is None:
        input_features = INPUT_FEATURES
    feature_columns = list(input_features)

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
            train_bin[list(OUTPUT_FEATURES)].to_numpy(),
        )
        y_pred = model.predict(test_bin[feature_columns].to_numpy())
        y_true = test_bin[list(OUTPUT_FEATURES)].to_numpy()
        results.append((bin_index, y_true, y_pred))
    return results


def enrich_fastpm_catalog(
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    bin_edges: np.ndarray,
    mass_column_fastpm: str,
    min_bin_size: int = 10,
    conditional_model_class=None,
    input_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[int, object]]:
    """
    Fit Haloscope on full SIM per mass bin and write predictions into ``halos_fastpm``.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        SIM training table.
    halos_fastpm : pd.DataFrame
        FastPM target table; output columns are added in place on a copy.
    bin_edges : np.ndarray
        log10 mass bin edges.
    mass_column_fastpm : str
        Mass column in FastPM used for bin assignment.
    min_bin_size : int, optional
        Skip bins with too few SIM halos or no FastPM halos.
    conditional_model_class : type, optional
        Haloscope model class.
    input_features : Sequence[str], optional
        Haloscope conditioning columns; defaults to ``INPUT_FEATURES``.

    Returns
    -------
    tuple[pd.DataFrame, dict[int, object]]
        Enriched FastPM frame and fitted models keyed by bin index.
    """
    if conditional_model_class is None:
        conditional_model_class = ConditionalMultiVariateGaussian
    if input_features is None:
        input_features = INPUT_FEATURES
    feature_columns = list(input_features)

    enriched = halos_fastpm.copy()
    for feature in OUTPUT_FEATURES:
        enriched[feature] = np.nan

    log_mass_sim = np.log10(halos_sim["M200b"].to_numpy())
    log_mass_fastpm = np.log10(enriched[mass_column_fastpm].to_numpy())
    models: Dict[int, object] = {}
    n_bins = len(bin_edges) - 1

    for bin_index in range(n_bins):
        low, high = bin_edges[bin_index], bin_edges[bin_index + 1]
        sim_mask = mask_mass_bin(log_mass_sim, low, high)
        fastpm_mask = mask_mass_bin(log_mass_fastpm, low, high)
        sim_bin = halos_sim.loc[sim_mask]
        if len(sim_bin) < min_bin_size or fastpm_mask.sum() == 0:
            continue
        model = conditional_model_class()
        model.fit(
            sim_bin[feature_columns].to_numpy(),
            sim_bin[list(OUTPUT_FEATURES)].to_numpy(),
        )
        y_pred = model.predict(enriched.loc[fastpm_mask, feature_columns].to_numpy())
        for column_index, column_name in enumerate(OUTPUT_FEATURES):
            enriched.loc[fastpm_mask, column_name] = y_pred[:, column_index]
        models[bin_index] = model
    return enriched, models
