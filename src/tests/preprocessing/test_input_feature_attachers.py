"""Tests for concrete Haloscope input feature attachers."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.n_halos_env import (
    NHalosEnvironmentFeature,
)
from density_field_properties.preprocessing.input_features.rockstar_t_over_u import (
    RockstarTOverUFeature,
)
from density_field_properties.preprocessing.schemas import SchemaValidationError


def test_n_halos_environment_feature_adds_env_column():
    """
    NHalosEnvironmentFeature writes the env column from halo positions.
    """
    catalog = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    run_context = SimulationRunContext(boxsize_mpc_h=10.0, env_radius_mpc_h=2.0)
    enriched = NHalosEnvironmentFeature().attach(catalog, run_context)
    assert "env" in enriched.columns
    assert len(enriched) == 2


def test_rockstar_t_over_u_feature_requires_catalog_column():
    """
    RockstarTOverUFeature validates that T/|U| was loaded by the reader.
    """
    run_context = SimulationRunContext(boxsize_mpc_h=1000.0)
    with pytest.raises(SchemaValidationError, match="t_over_u"):
        RockstarTOverUFeature().attach(pd.DataFrame({"x": [1.0]}), run_context)

    catalog = pd.DataFrame({"t_over_u": [0.8, np.nan]})
    enriched = RockstarTOverUFeature().attach(catalog, run_context)
    assert "t_over_u" in enriched.columns
