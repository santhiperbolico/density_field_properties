"""Import checks for environment_properties modules."""

import importlib

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
