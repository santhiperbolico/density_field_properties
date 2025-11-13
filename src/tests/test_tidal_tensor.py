import numpy as np
import pytest

from density_field_properties.tidal_tensor import TidalTensor

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


def test_tidal_tensor_fft_from_delta(dm_particles, density_field):
    tidal_tensor = TidalTensor.tidal_tensor_fft_from_delta(
        delta=density_field, box_size=BOX_SIZE, gaussian_scale=1.0, deconvolve_cic=False
    )
    assert isinstance(tidal_tensor, TidalTensor)
    assert isinstance(tidal_tensor.tidal_tensor, dict)
    assert len(tidal_tensor.tidal_tensor) == 9
