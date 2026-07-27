"""Import checks for Haloscope and SIM-to-FastPM modules (no cluster data)."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,symbol_names",
    [
        (
            "density_field_properties.haloscope",
            ["ConditionalMultiVariateGaussian"],
        ),
        (
            "density_field_properties.haloscope.sim_to_fastpm.config",
            [
                "default_sim_hlist_path",
                "default_fastpm_list_path",
                "OUTPUT_FEATURES",
            ],
        ),
        (
            "density_field_properties.haloscope.sim_to_fastpm.pipeline",
            ["run_sim_to_fastpm_haloscope_pipeline"],
        ),
        (
            "density_field_properties.haloscope.sim_to_fastpm.load_catalogs",
            ["load_unit_sim_training_catalog", "load_fastpm_target_catalog"],
        ),
        (
            "density_field_properties.haloscope.sim_to_fastpm.training",
            ["default_mass_bin_edges", "enrich_fastpm_catalog", "holdout_validate_sim_bins"],
        ),
        (
            "density_field_properties.haloscope.sim_to_fastpm.plotting",
            ["corner_plot_sim_validation", "median_property_vs_mass"],
        ),
    ],
)
def test_haloscope_public_symbols_are_importable(module_path, symbol_names):
    """
    Each listed symbol must resolve on the module used by the notebook and CLI runner.
    """
    module = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(module, name), f"{module_path} missing {name}"


def test_run_sim_to_fastpm_haloscope_script_main_imports():
    """
    Mirror ``scripts/run_sim_to_fastpm_haloscope.py`` deferred imports.
    """
    from density_field_properties.haloscope.sim_to_fastpm.config import (
        default_fastpm_list_path,
        default_sim_hlist_path,
    )
    from density_field_properties.haloscope.sim_to_fastpm.pipeline import (
        run_sim_to_fastpm_haloscope_pipeline,
    )

    assert callable(default_sim_hlist_path)
    assert callable(default_fastpm_list_path)
    assert callable(run_sim_to_fastpm_haloscope_pipeline)
