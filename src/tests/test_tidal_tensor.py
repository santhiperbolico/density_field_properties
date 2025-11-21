import os
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

from density_field_properties.tidal_tensor import GAUSSIAN_SCALE_DEFAULT, NAME_HD5, TidalTensor

MASS_PARTICLE = 1.2e9
BOX_SIZE = 3


@pytest.fixture
def dm_particles() -> np.ndarray:
    data = np.array([[1, 1, 1], [2.9, 1, 0], [0, 0, 2.9], [1, 2, 0], [0, 2.9, 0], [2.9, 0, 0]])
    return data


@pytest.fixture
def density_field() -> np.ndarray:
    expected = np.zeros((3, 3, 3))

    expected[1, 1, 1] += MASS_PARTICLE

    expected[2, 1, 0] += 0.1 * MASS_PARTICLE
    expected[0, 1, 0] += 0.9 * MASS_PARTICLE

    expected[0, 0, 2] += 0.1 * MASS_PARTICLE
    expected[0, 0, 0] += 0.9 * MASS_PARTICLE

    expected[1, 2, 0] += MASS_PARTICLE

    expected[0, 2, 0] += 0.1 * MASS_PARTICLE
    expected[0, 0, 0] += 0.9 * MASS_PARTICLE

    expected[2, 0, 0] += 0.1 * MASS_PARTICLE
    expected[0, 0, 0] += 0.9 * MASS_PARTICLE
    return expected


@pytest.mark.parametrize("gaussian_scale", [None, 1.0])
def test_tidal_tensor_from_delta(gaussian_scale, density_field):
    with TemporaryDirectory() as tmpath:
        tidal_tensor = TidalTensor.from_delta(
            delta=density_field, box_size=BOX_SIZE, path=tmpath, gaussian_scale=gaussian_scale
        )
    assert isinstance(tidal_tensor, TidalTensor)
    assert isinstance(tidal_tensor.tidal_tensor, dict)
    assert len(tidal_tensor.tidal_tensor) == 9


@pytest.mark.parametrize(
    "cell_x, cell_y, cell_z, expected_size",
    [(1, 1, 1, 1), (np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), 2)],
)
def test_tidal_tensor_get_tidal_tensor(cell_x, cell_y, cell_z, expected_size, density_field):
    with TemporaryDirectory() as tmpath:
        tidal_tensor = TidalTensor.from_delta(
            delta=density_field, box_size=BOX_SIZE, path=tmpath, gaussian_scale=1.0
        )
        t_comp = tidal_tensor.get_tidal_tensor(cell_x, cell_y, cell_z)
    assert t_comp.shape == (expected_size, 3, 3)


@pytest.mark.parametrize("gaussian_scale", [None, 1.0])
def test_tidal_tensor_from_folder(gaussian_scale):
    components = [0, 3, 4, 6, 7, 8]
    gaussian_scale_name = gaussian_scale if gaussian_scale is not None else GAUSSIAN_SCALE_DEFAULT
    with TemporaryDirectory() as tmpath:
        default_values = np.zeros((3, 3, 3))
        for component in components:
            file_name = os.path.join(tmpath, f"TidalTensor_{component}_{gaussian_scale_name}.h5")
            with h5py.File(file_name, "w") as f:
                f.create_dataset(NAME_HD5, data=default_values, dtype="float32")
        tidal_tensor = TidalTensor.from_folder(path=tmpath, gaussian_scale=gaussian_scale)
    assert isinstance(tidal_tensor, TidalTensor)
    assert isinstance(tidal_tensor.tidal_tensor, dict)
    assert len(tidal_tensor.tidal_tensor) == 9
