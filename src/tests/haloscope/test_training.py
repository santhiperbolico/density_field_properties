"""Unit tests for Haloscope fit and predict helpers."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.haloscope import (
    default_mass_bin_edges,
    enrich_fastpm_catalog,
    fit_models,
    holdout_validate_sim_bins,
    predict_models,
)
from density_field_properties.haloscope.bins import mask_mass_bin


def _synthetic_catalog(n_halos: int, seed: int, env_offset: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    masses = rng.uniform(1e11, 1e13, size=n_halos)
    env = rng.normal(loc=env_offset, scale=1.0, size=n_halos)
    return pd.DataFrame(
        {
            "M200b": masses,
            "env": env,
            "cv": rng.uniform(0.5, 2.0, size=n_halos),
            "Spin": rng.uniform(0.01, 0.5, size=n_halos),
            "ca": rng.uniform(0.2, 0.9, size=n_halos),
            "ba": rng.uniform(0.2, 0.9, size=n_halos),
        }
    )


@pytest.mark.parametrize(
    "low,high,values,expected",
    [
        (1.0, 2.0, [0.9, 1.0, 1.5, 2.0], [False, True, True, False]),
    ],
)
def test_mask_mass_bin(low, high, values, expected):
    """
    Mass-bin masks must follow the closed-open interval ``[low, high)``.
    """
    mask = mask_mass_bin(np.array(values, dtype=float), low, high)
    assert mask.tolist() == expected


def test_default_mass_bin_edges_match_assembly_bias_grid():
    """
    Haloscope and assembly-bias diagnostics must share the same log-mass grid.
    """
    edges = default_mass_bin_edges(13.2)
    assert edges[0] == 11.5
    assert edges[-1] == 13.2
    assert len(edges) == 10


def test_fit_and_predict_models_on_synthetic_tables():
    """
    Fit and predict must populate output columns for overlapping mass bins.
    """
    training = _synthetic_catalog(120, seed=0)
    target = _synthetic_catalog(80, seed=1, env_offset=0.2)
    target["M200b_cal"] = target["M200b"]
    bin_edges = default_mass_bin_edges(np.log10(training["M200b"].max()))
    input_features = ("env",)
    output_features = ("cv", "Spin", "ca", "ba")

    models = fit_models(
        training,
        bin_edges,
        input_features,
        output_features,
        min_bin_size=5,
    )
    assert models

    enriched = predict_models(
        models,
        target,
        bin_edges,
        input_features,
        output_features,
        mass_column="M200b_cal",
    )
    predicted = enriched[list(output_features)].notna().all(axis=1).sum()
    assert predicted > 0


def test_enrich_fastpm_catalog_matches_fit_predict_composition():
    """
    The enrichment helper must delegate to fit and predict without losing rows.
    """
    training = _synthetic_catalog(120, seed=2)
    target = _synthetic_catalog(80, seed=3)
    bin_edges = default_mass_bin_edges(np.log10(training["M200b"].max()))
    input_features = ("env",)
    output_features = ("cv", "Spin", "ca", "ba")

    enriched, models = enrich_fastpm_catalog(
        training,
        target,
        bin_edges,
        mass_column_fastpm="M200b",
        input_features=input_features,
        output_features=output_features,
        min_bin_size=5,
    )
    assert len(enriched) == len(target)
    assert models
    assert enriched[list(output_features)].notna().any().any()


def test_holdout_validate_sim_bins_returns_per_bin_results():
    """
    Hold-out validation must return one tuple per mass bin.
    """
    training = _synthetic_catalog(200, seed=4)
    bin_edges = default_mass_bin_edges(np.log10(training["M200b"].max()))
    results = holdout_validate_sim_bins(
        training,
        bin_edges,
        input_features=("env",),
        output_features=("cv", "Spin", "ca", "ba"),
        min_bin_size=5,
        random_seed=0,
    )
    assert len(results) == len(bin_edges) - 1
