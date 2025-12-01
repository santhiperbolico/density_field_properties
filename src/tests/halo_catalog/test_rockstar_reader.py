import os
from tempfile import TemporaryDirectory

import numpy as np

from density_field_properties.cosmology import Cosmology
from density_field_properties.halo_catalog.rockstar import (
    RockstarCatalogReader,
    read_rockstar_cosmology_header,
)


def test_read_rockstar_cosmology_header():
    txt = """#Omega_M = 0.3 ; Omega_L = 0.7; h0 = 0.6
            # Other header
            1 2 3
            """
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)
        cosmo = read_rockstar_cosmology_header(tmpfile_name)

    assert isinstance(cosmo, Cosmology)
    assert cosmo.omega_matter == 0.3
    assert cosmo.omega_lambda == 0.7
    assert cosmo.h0 == 0.6


def test_rockstar_reader_basic(tmp_path):
    txt = """# Comment
0 10 0 0 0  1 2 3 4 5  39  1.0 2.0 3.0  100.0
1 20 0 0 0  1 2 3 4 5  39  4.0 5.0 6.0  200.0
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)

        hcat = RockstarCatalogReader.read_catalog(
            tmpfile_name,
            halo_x_position=11,
            halo_y_position=12,
            halo_z_position=13,
            halo_m200b_position=14,
        )

    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([10, 20]))
    assert np.all(hcat.halo_m200b == np.array([100.0, 200.0]))


def test_rockstar_batch_generator(tmp_path):
    txt = """# Header
#Omega_M = 0.3 ; Omega_L = 0.7 ; h0 = 0.6
0 10 0 0 0  1 2 3 4 5  39  1.0 2.0 3.0  100.0
1 20 0 0 0  1 2 3 4 5  39  4.0 5.0 6.0  200.0
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)

        gen = RockstarCatalogReader.read_catalog_batch_generator(
            tmpfile_name,
            batch_size=1,
            halo_x_position=11,
            halo_y_position=12,
            halo_z_position=13,
            halo_m200b_position=14,
        )
        batch, start, end = next(gen)

    assert batch.n_halos == 1
    assert batch.halo_id[0] == 10
    assert isinstance(start, int)
    assert isinstance(end, int)
