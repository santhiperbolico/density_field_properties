"""Unified Haloscope HR/LR feature table builder."""

from typing import Sequence

import pandas as pd

from density_field_properties.preprocessing.context import (
    PreprocessingContext,
    SimulationRunContext,
)
from density_field_properties.preprocessing.filters import filter_finite_input_features
from density_field_properties.preprocessing.input_features.base import InputFeatureAttacher
from density_field_properties.preprocessing.input_features.registry import resolve_attachers
from density_field_properties.preprocessing.mass_calibration import calibrate_lr_mass
from density_field_properties.preprocessing.schemas import (
    validate_hr_training_table,
    validate_lr_target_table,
)


def _apply_attachers(
    catalog: pd.DataFrame,
    attachers: Sequence[InputFeatureAttacher],
    run_context: SimulationRunContext,
) -> pd.DataFrame:
    """
    Run all input feature attachers on one catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Raw halo table for one simulation.
    attachers : Sequence[InputFeatureAttacher]
        Ordered feature attachers.
    run_context : SimulationRunContext
        Simulation-specific paths and grid parameters.

    Returns
    -------
    pd.DataFrame
        Catalog with all requested INPUT columns attached.
    """
    enriched = catalog
    for attacher in attachers:
        enriched = attacher.attach(enriched, run_context)
    return enriched


def build_feature_tables(
    halos_sim: pd.DataFrame,
    halos_fastpm: pd.DataFrame,
    input_features: Sequence[str],
    context: PreprocessingContext,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Attach INPUT features and optional LR mass calibration for Haloscope.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        High-resolution training catalog.
    halos_fastpm : pd.DataFrame
        Low-resolution target catalog.
    input_features : Sequence[str]
        Ordered Haloscope INPUT feature names.
    context : PreprocessingContext
        HR/LR simulation contexts and mass-calibration flag.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, str]
        HR table, LR table, and LR mass column for bin assignment.

    Raises
    ------
    ValueError
        If either catalog is empty after filtering finite INPUT values.
    """
    if len(halos_sim) == 0 or len(halos_fastpm) == 0:
        raise ValueError("Empty SIM or FastPM catalog after loading and host filtering.")

    attachers = resolve_attachers(input_features)
    hr_table = _apply_attachers(halos_sim, attachers, context.sim)
    lr_table = _apply_attachers(halos_fastpm, attachers, context.fastpm)

    hr_table = filter_finite_input_features(hr_table, input_features)
    lr_table = filter_finite_input_features(lr_table, input_features)
    if len(hr_table) == 0 or len(lr_table) == 0:
        raise ValueError(
            "No halos left after attaching input features; "
            "check catalogs, descriptor directories, and batch limits."
        )

    lr_table, mass_column = calibrate_lr_mass(
        lr_table,
        hr_table,
        context.calibrate_mass,
    )
    validate_hr_training_table(hr_table, input_features)
    validate_lr_target_table(
        lr_table,
        input_features,
        calibrate_mass=context.calibrate_mass,
    )
    return hr_table, lr_table, mass_column
