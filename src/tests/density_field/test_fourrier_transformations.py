import pytest

from density_field_properties.density_field.fourrier_transformations import kgrid


@pytest.mark.parametrize("n_grid, box_size", [(10, 10), (10, 100), (100, 100), (512, 1000)])
def test_kgrid(n_grid, box_size):
    kx, ky, kz = kgrid(n_grid, box_size)
    assert kx.shape == (n_grid, n_grid, n_grid // 2 + 1)
