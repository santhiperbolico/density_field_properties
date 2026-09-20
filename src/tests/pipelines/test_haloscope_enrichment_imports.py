"""Import checks for pipeline orchestration modules."""

import importlib
import warnings

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


def test_legacy_pipeline_shim_emits_deprecation_warning():
    """
    Legacy sim_to_fastpm.pipeline imports must warn but remain callable.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(
            "density_field_properties.haloscope.sim_to_fastpm.pipeline"
        )
        assert any(
            issubclass(item.category, DeprecationWarning) for item in caught
        ), "Expected DeprecationWarning from legacy pipeline shim"
        assert callable(module.run_sim_to_fastpm_haloscope_pipeline)


def test_pipelines_entrypoint_main_imports():
    """
    Mirror ``pipelines/run_haloscope_enrichment.py`` deferred imports.
    """
    from density_field_properties.haloscope.sim_to_fastpm.config import (
        default_fastpm_list_path,
        default_sim_hlist_path,
    )
    from density_field_properties.pipelines.config import load_haloscope_enrichment_config
    from density_field_properties.pipelines.haloscope_enrichment import (
        run_haloscope_enrichment_pipeline,
    )

    assert callable(default_sim_hlist_path)
    assert callable(default_fastpm_list_path)
    assert callable(load_haloscope_enrichment_config)
    assert callable(run_haloscope_enrichment_pipeline)
