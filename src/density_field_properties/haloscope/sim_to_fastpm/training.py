"""Deprecated shim — use density_field_properties.haloscope.training and predict."""

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from density_field_properties.haloscope.bins import default_mass_bin_edges, mask_mass_bin
from density_field_properties.haloscope.predict import (
    enrich_fastpm_catalog as _enrich_fastpm_catalog,
)
from density_field_properties.haloscope.sim_to_fastpm.config import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
)
from density_field_properties.haloscope.training import (
    holdout_validate_sim_bins as _holdout_validate_sim_bins,
)

warnings.warn(
    "haloscope.sim_to_fastpm.training is deprecated; " "use density_field_properties.haloscope",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "default_mass_bin_edges",
    "enrich_fastpm_catalog",
    "holdout_validate_sim_bins",
    "mask_mass_bin",
]


def holdout_validate_sim_bins(
    halos_sim: pd.DataFrame,
    bin_edges: np.ndarray,
    mass_column: str = "M200b",
    test_fraction: float = 0.2,
    random_seed: int = 0,
    min_bin_size: int = 10,
    conditional_model_class=None,
    input_features: Optional[Sequence[str]] = None,
    output_features: Optional[Sequence[str]] = None,
):
    """
    Deprecated wrapper that defaults feature columns from sim_to_fastpm config.
    """
    if input_features is None:
        feature_columns = list(INPUT_FEATURES)
    else:
        feature_columns = list(input_features)
    if output_features is None:
        target_columns = list(OUTPUT_FEATURES)
    else:
        target_columns = list(output_features)
    return _holdout_validate_sim_bins(
        halos_sim,
        bin_edges,
        feature_columns,
        target_columns,
        mass_column=mass_column,
        test_fraction=test_fraction,
        random_seed=random_seed,
        min_bin_size=min_bin_size,
        conditional_model_class=conditional_model_class,
    )


def enrich_fastpm_catalog(
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    bin_edges: np.ndarray,
    mass_column_fastpm: str,
    min_bin_size: int = 10,
    conditional_model_class=None,
    input_features: Optional[Sequence[str]] = None,
    output_features: Optional[Sequence[str]] = None,
):
    """
    Deprecated wrapper that defaults feature columns from sim_to_fastpm config.
    """
    if input_features is None:
        feature_columns = list(INPUT_FEATURES)
    else:
        feature_columns = list(input_features)
    if output_features is None:
        target_columns = list(OUTPUT_FEATURES)
    else:
        target_columns = list(output_features)
    return _enrich_fastpm_catalog(
        halos_sim,
        halos_fastpm,
        bin_edges,
        mass_column_fastpm,
        feature_columns,
        target_columns,
        min_bin_size=min_bin_size,
        conditional_model_class=conditional_model_class,
    )
