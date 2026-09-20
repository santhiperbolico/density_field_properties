"""Tests for Haloscope input feature registry."""

import pytest

from density_field_properties.preprocessing.input_features.n_halos_env import (
    NHalosEnvironmentFeature,
)
from density_field_properties.preprocessing.input_features.registry import resolve_attachers
from density_field_properties.preprocessing.input_features.rockstar_t_over_u import (
    RockstarTOverUFeature,
)


def test_resolve_attachers_returns_instances_in_order():
    """
    Registry preserves the requested feature order.
    """
    attachers = resolve_attachers(("env", "t_over_u"))
    assert [type(item) for item in attachers] == [
        NHalosEnvironmentFeature,
        RockstarTOverUFeature,
    ]


@pytest.mark.parametrize(
    "input_features,match",
    [
        ((), "at least one"),
        (("env", "env"), "Duplicate"),
        (("unknown_feature",), "Unknown input feature"),
    ],
)
def test_resolve_attachers_rejects_invalid_requests(input_features, match):
    """
    Registry fails fast on empty, duplicate, or unknown feature names.
    """
    with pytest.raises(ValueError, match=match):
        resolve_attachers(input_features)
