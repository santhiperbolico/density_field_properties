"""Tests for validation memory helpers."""

from unittest.mock import patch

from density_field_properties.validation.memory import release_validation_memory


@patch("density_field_properties.validation.memory.gc.collect")
@patch("matplotlib.pyplot.close")
def test_release_validation_memory_closes_figures_and_collects(mock_close, mock_collect):
    """
    Validation cleanup should close matplotlib figures and run garbage collection.
    """
    release_validation_memory()

    mock_close.assert_called_once_with("all")
    mock_collect.assert_called_once_with()
