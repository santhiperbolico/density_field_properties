"""Tests for N-halo environment feature attachment."""

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.n_halos_env import (
    NHalosEnvironmentFeature,
    _local_environment,
)


def test_local_environment_counts_neighbors_in_periodic_box():
    """
    Periodic KD-tree neighbor counts exclude self and use log10(1 + N).
    """
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    env = _local_environment(positions, boxsize_mpc_h=10.0, radius_mpc_h=2.0)
    assert env.shape == (2,)
    assert env[0] == pytest.approx(np.log10(2.0))


def test_n_halos_environment_feature_adds_env_column():
    """
    NHalosEnvironmentFeature writes the env column on the catalog.
    """
    catalog = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
    run_context = SimulationRunContext(boxsize_mpc_h=10.0, env_radius_mpc_h=2.0)
    enriched = NHalosEnvironmentFeature().attach(catalog, run_context)
    assert "env" in enriched.columns
    assert len(enriched) == 1
