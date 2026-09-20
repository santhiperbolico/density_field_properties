"""Haloscope prediction and FastPM enrichment."""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from density_field_properties.haloscope.bins import mask_mass_bin
from density_field_properties.haloscope.training import fit_models


def predict_models(
    models: Dict[int, object],
    target_table: pd.DataFrame,
    bin_edges: np.ndarray,
    input_features: Sequence[str],
    output_features: Sequence[str],
    mass_column: str = "M200b_cal",
) -> pd.DataFrame:
    """
    Predict secondary properties on the target table using fitted bin models.

    Parameters
    ----------
    models : dict[int, object]
        Fitted Haloscope models keyed by bin index.
    target_table : pd.DataFrame
        Target catalog to enrich.
    bin_edges : np.ndarray
        log10 mass bin edges used during fit.
    input_features : Sequence[str]
        Haloscope conditioning columns.
    output_features : Sequence[str]
        Target secondary-property columns to write.
    mass_column : str, optional
        Mass column in the target table used for bin assignment.

    Returns
    -------
    pd.DataFrame
        Copy of ``target_table`` with predicted output columns filled.
    """
    enriched = target_table.copy()
    target_columns = list(output_features)
    for feature in target_columns:
        enriched[feature] = np.nan

    feature_columns = list(input_features)
    log_mass = np.log10(enriched[mass_column].to_numpy())

    for bin_index, model in models.items():
        low, high = bin_edges[bin_index], bin_edges[bin_index + 1]
        target_mask = mask_mass_bin(log_mass, low, high)
        if target_mask.sum() == 0:
            continue
        y_pred = model.predict(enriched.loc[target_mask, feature_columns].to_numpy())
        for column_index, column_name in enumerate(target_columns):
            enriched.loc[target_mask, column_name] = y_pred[:, column_index]
    return enriched


def enrich_fastpm_catalog(
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    bin_edges: np.ndarray,
    mass_column_fastpm: str,
    input_features: Sequence[str],
    output_features: Sequence[str],
    min_bin_size: int = 10,
    conditional_model_class=None,
) -> Tuple[pd.DataFrame, Dict[int, object]]:
    """
    Fit Haloscope on full SIM per mass bin and write predictions into ``halos_fastpm``.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        SIM training table.
    halos_fastpm : pd.DataFrame
        FastPM target table; output columns are added on a copy.
    bin_edges : np.ndarray
        log10 mass bin edges.
    mass_column_fastpm : str
        Mass column in FastPM used for bin assignment.
    input_features : Sequence[str]
        Haloscope conditioning columns.
    output_features : Sequence[str]
        Target secondary-property columns.
    min_bin_size : int, optional
        Skip bins with too few SIM halos or no FastPM halos.
    conditional_model_class : type, optional
        Haloscope model class.

    Returns
    -------
    tuple[pd.DataFrame, dict[int, object]]
        Enriched FastPM frame and fitted models keyed by bin index.
    """
    models = fit_models(
        halos_sim,
        bin_edges,
        input_features,
        output_features,
        min_bin_size=min_bin_size,
        mass_column="M200b",
        conditional_model_class=conditional_model_class,
    )
    enriched = predict_models(
        models,
        halos_fastpm,
        bin_edges,
        input_features,
        output_features,
        mass_column=mass_column_fastpm,
    )
    return enriched, models
