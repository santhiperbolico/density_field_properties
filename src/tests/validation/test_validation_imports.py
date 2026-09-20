"""Import checks for validation and utils packages."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,symbol_names",
    [
        (
            "density_field_properties.validation.assembly_bias",
            [
                "assembly_bias_curves_for_catalog",
                "attach_paranjape_bias",
                "load_sim_matter_overdensity",
            ],
        ),
        (
            "density_field_properties.validation.plots",
            ["plot_assembly_bias_env_panel"],
        ),
        (
            "density_field_properties.validation.marginals",
            ["corner_plot_sim_validation"],
        ),
        (
            "density_field_properties.utils.stats",
            ["central_68_scatter", "median_property_vs_mass"],
        ),
        (
            "density_field_properties.utils.plotting",
            ["compare_2d_contours"],
        ),
    ],
)
def test_validation_and_utils_symbols_are_importable(module_path, symbol_names):
    """
    Each listed symbol must resolve on the canonical validation/utils modules.
    """
    if module_path == "density_field_properties.validation.assembly_bias":
        try:
            importlib.import_module("bigfile")
        except Exception:
            pytest.skip("bigfile not available")
    module = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(module, name), f"{module_path} missing {name}"
