import numpy as np
import pytest

from density_field_properties.density_field.fourrier_transformations import kgrid


def kvector(boz_size, n_grid, flag=0, sparse=False):
    """
    Sujatha's Function to calculate the k-vector.

    Parameters
    ----------
    box_size : float
        Physical size of the simulation box.
    n_grid : int
        Number of grid cells per dimension.
    flag: int, default=0
        If flag=1, the k-vector is calculated using the sine function.
    sparse: bool, default=False
        If True the shape of the returned coordinate array for dimension *i*

    Returns
    -------
    k_x, k_y, k_z : ndarray
        3D arrays with the Cartesian components of the wavevector for each Fourier mode
    """
    dk = 2 * np.pi / boz_size
    kspace = np.concatenate([range(0, int(n_grid / 2)), range(-int(n_grid / 2), 0)]) * dk

    k_x, k_y, k_z = np.meshgrid(
        kspace, kspace, kspace[0 : n_grid // 2 + 1], indexing="ij", sparse=sparse
    )
    if flag == 1:
        k_x = np.sin(k_x * boz_size / n_grid)
        k_y = np.sin(k_y * boz_size / n_grid)
        k_z = np.sin(k_z * boz_size / n_grid)
    return k_x, k_y, k_z


@pytest.mark.parametrize("n_grid, box_size", [(10, 10), (10, 100), (100, 100), (512, 1000)])
def test_kgrid(n_grid, box_size):
    kx, ky, kz = kgrid(n_grid, box_size)
    assert kx.shape == (n_grid, n_grid, n_grid // 2 + 1)
    assert ky.shape == (n_grid, n_grid, n_grid // 2 + 1)
    assert kz.shape == (n_grid, n_grid, n_grid // 2 + 1)


@pytest.mark.parametrize(
    "n_grid, box_size",
    [
        (10, 10),
        (10, 100),
        (100, 100),
    ],
)
def test_kgrid_kvector(n_grid, box_size):
    kx, ky, kz = kgrid(n_grid, box_size)
    ekx, eky, ekz = kvector(box_size, n_grid)
    assert ekx == pytest.approx(kx, abs=1e-7)
    assert eky == pytest.approx(ky, abs=1e-7)
    assert ekz == pytest.approx(kz, abs=1e-7)
