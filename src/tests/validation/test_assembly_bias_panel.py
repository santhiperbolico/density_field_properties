"""Tests for assembly-bias panel helpers."""

import pytest


@pytest.mark.parametrize(
    "input_features,expected_feature_text",
    [
        (("env",), "env"),
        (("t_over_u", "tidal_anisotropy"), "T/|U| + tidal anisotropy"),
        (("env", "t_over_u"), "env + T/|U|"),
    ],
)
def test_format_assembly_bias_panel_title(input_features, expected_feature_text):
    """
    Panel titles should reflect the Haloscope INPUT features of the run.
    """
    pytest.importorskip("bigfile")
    from density_field_properties.validation.assembly_bias_panel import (
        format_assembly_bias_panel_title,
    )

    title = format_assembly_bias_panel_title(
        input_features,
        sim_delta_mode="saved CIC",
        fastpm_delta_mode="halo CIC",
    )

    assert title == (f"input: {expected_feature_text} " "(HR δ: saved CIC, FastPM δ: halo CIC)")
