"""Import checks for environment_properties and legacy shims."""

import importlib
import warnings

import pytest


@pytest.mark.parametrize(
    "module_path,symbol_names",
    [
        (
            "density_field_properties.environment_properties",
            [
                "DensityFieldInfo",
                "TidalTensorArray",
                "kgrid",
            ],
        ),
        (
            "density_field_properties.environment_properties.cic",
            [
                "DensityFieldInfo",
                "get_grid_cell",
            ],
        ),
        (
            "density_field_properties.environment_properties.tidal_tensor",
            [
                "TidalTensor",
                "TidalTensorArray",
                "TIDAL_TENSOR_PATH",
            ],
        ),
    ],
)
def test_environment_properties_public_symbols_are_importable(module_path, symbol_names):
    """
    Each listed symbol must resolve on the canonical environment_properties modules.
    """
    module = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(module, name), f"{module_path} missing {name}"


def test_legacy_density_field_utils_shim_emits_deprecation_warning():
    """
    Legacy density_field.utils imports must warn but remain callable.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("density_field_properties.density_field.utils")
        assert any(
            issubclass(item.category, DeprecationWarning) for item in caught
        ), "Expected DeprecationWarning from legacy density_field.utils shim"
        assert callable(module.get_grid_cell)


def test_legacy_tidal_tensor_shim_emits_deprecation_warning():
    """
    Legacy tidal_tensor imports must warn but remain callable.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("density_field_properties.tidal_tensor")
        assert any(
            issubclass(item.category, DeprecationWarning) for item in caught
        ), "Expected DeprecationWarning from legacy tidal_tensor shim"
        assert hasattr(module, "TidalTensorArray")
