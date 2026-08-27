import numpy as np
import pytest

from density_field_properties.density_field.utils import get_grid_cell


@pytest.mark.parametrize(
    "positions, box_size, n_grid, expected_cells",
    [
        (
            np.array([[2.5, 5.0, 7.5]]),
            10,
            5,
            np.array([[1, 2, 3]]),
        ),
        (
            np.array([[0.0, 4.0, 8.0], [9.5, 1.0, 3.0]]),
            10,
            5,
            np.array([[0, 2, 4], [4, 0, 1]]),
        ),
    ],
)
def test_get_grid_cell_maps_each_axis(positions, box_size, n_grid, expected_cells):
    cells = get_grid_cell(positions, box_size=box_size, n_grid=n_grid)

    assert cells.shape == expected_cells.shape
    np.testing.assert_array_equal(cells, expected_cells)
    assert not np.all(cells[:, 0] == cells[:, 1])
