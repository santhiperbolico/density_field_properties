"""Import checks for Haloscope and pipeline modules (no cluster data)."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,symbol_names",
    [
        (
            "density_field_properties.haloscope",
            [
                "ConditionalMultiVariateGaussian",
                "default_mass_bin_edges",
                "enrich_fastpm_catalog",
                "fit_models",
                "holdout_validate_sim_bins",
                "predict_models",
            ],
        ),
        (
            "density_field_properties.pipelines.run_defaults",
            [
                "default_sim_hlist_path",
                "default_fastpm_list_path",
                "OUTPUT_FEATURES",
            ],
        ),
        (
            "density_field_properties.preprocessing.catalog_loaders",
            ["load_unit_sim_training_catalog", "load_fastpm_target_catalog"],
        ),
        (
            "density_field_properties.validation.marginals",
            ["corner_plot_sim_validation"],
        ),
        (
            "density_field_properties.utils.stats",
            ["median_property_vs_mass"],
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
    pytest.importorskip("bigfile")
    from density_field_properties.pipelines.haloscope_enrichment import (
        run_haloscope_enrichment_pipeline,
    )
    from density_field_properties.pipelines.run_defaults import (
        default_fastpm_list_path,
        default_sim_hlist_path,
    )

    assert callable(default_sim_hlist_path)
    assert callable(default_fastpm_list_path)
    assert callable(run_haloscope_enrichment_pipeline)
