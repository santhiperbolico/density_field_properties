from tempfile import TemporaryDirectory

import numpy as np
import pytest

from density_field_properties.density_field.cic_deposit import (
    density_field_cic_main,
    load_density_field_cic,
    mass_field_cic,
    save_density_field_cic,
)
from density_field_properties.density_field.utils import DensityFieldInfo

MASS_PARTICLE = 1.2e9
BOX_SIZE = 3


@pytest.fixture
def dm_particles() -> np.ndarray:
    data = np.array([[1, 1, 1], [2.9, 1, 0], [0, 0, 2.9], [1, 2, 0], [0, 2.9, 0], [2.9, 0, 0]])
    return data


@pytest.fixture
def expected_mass_field() -> np.ndarray:
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


def test_mass_field_cic(dm_particles, expected_mass_field):
    mass_field = mass_field_cic(
        data=dm_particles, mass_particle=MASS_PARTICLE, box_size=BOX_SIZE, n_grid=3
    )
    assert np.allclose(mass_field, expected_mass_field)


def test_density_field_cic_main(dm_particles, expected_mass_field):
    with TemporaryDirectory() as tmpdirname:
        dm_particles_file = f"{tmpdirname}/dm_particles.txt"
        np.savetxt(dm_particles_file, dm_particles)
        density, density_info = density_field_cic_main(
            dm_particles_file=dm_particles_file,
            mass_particle=MASS_PARTICLE,
            box_size=BOX_SIZE,
            n_grid=3,
            batch_size=None,
        )
        assert np.allclose(density_info.n_particles, 6)
        assert np.allclose(density, expected_mass_field)


def test_density_field_cic_main_batches(dm_particles, expected_mass_field):
    with TemporaryDirectory() as tmpdirname:
        dm_particles_file = f"{tmpdirname}/dm_particles.txt"
        np.savetxt(dm_particles_file, dm_particles)
        density_batches, density_info_batches = density_field_cic_main(
            dm_particles_file=dm_particles_file,
            mass_particle=MASS_PARTICLE,
            box_size=BOX_SIZE,
            n_grid=3,
            batch_size=2,
        )
        assert np.allclose(density_info_batches.n_particles, 6)
        assert np.allclose(density_batches, expected_mass_field)


def test_save_load_densiti_field(expected_mass_field):
    with TemporaryDirectory() as tmpdirname:
        dm_particles_file = f"{tmpdirname}/dm_particles.txt"
        density_info = DensityFieldInfo(
            n_grid=3, box_size=BOX_SIZE, n_particles=6, mass_particle=MASS_PARTICLE
        )
        output_file = save_density_field_cic(
            expected_mass_field, tmpdirname, dm_particles_file, density_info
        )
        output_info_file = f"{tmpdirname}/dm_particles_density_info.txt"
        density_field, density_info = load_density_field_cic(output_file, output_info_file)
    assert np.allclose(density_info.n_particles, 6)
    assert np.allclose(density_field, expected_mass_field)
