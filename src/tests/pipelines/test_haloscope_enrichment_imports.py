"""Import checks for pipeline orchestration modules."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,symbol_names",
    [
        (
            "density_field_properties.pipelines",
            [
                "run_haloscope_enrichment_pipeline",
                "write_tidal_assembly_bias_panel",
            ],
        ),
        (
            "density_field_properties.pipelines.haloscope_enrichment",
            [
                "run_haloscope_enrichment_pipeline",
            ],
        ),
        (
            "density_field_properties.validation",
            [
                "write_tidal_assembly_bias_panel",
            ],
        ),
    ],
)
def test_pipeline_public_symbols_are_importable(module_path, symbol_names):
    """
    Each listed symbol must resolve on the canonical pipelines module.
    """
    module = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(module, name), f"{module_path} missing {name}"


def test_pipelines_entrypoint_main_imports():
    """
    Mirror ``pipelines/run_haloscope_enrichment.py`` deferred imports.
    """
    from density_field_properties.pipelines.config import load_haloscope_enrichment_config
    from density_field_properties.pipelines.haloscope_enrichment import (
        run_haloscope_enrichment_pipeline,
    )
    from density_field_properties.pipelines.run_defaults import (
        default_fastpm_list_path,
        default_sim_hlist_path,
    )

    assert callable(default_sim_hlist_path)
    assert callable(default_fastpm_list_path)
    assert callable(load_haloscope_enrichment_config)
    assert callable(run_haloscope_enrichment_pipeline)
